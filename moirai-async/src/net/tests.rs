#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::*;
use crate::executor::AsyncExecutor;
use crate::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use crate::timer::{timeout, TimeoutError};
use std::future::Future;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream as StdTcpStream};
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

const BACKPRESSURE_BUFFER_BYTES: usize = 4 * 1024;
const BACKPRESSURE_CHUNK: [u8; 16 * 1024] = [0xA5; 16 * 1024];
const BACKPRESSURE_MAX_BYTES: usize = 16 * 1024 * 1024;
const READINESS_PAYLOAD: [u8; 5] = *b"ready";

fn test_config() -> TcpServerConfig {
    TcpServerConfig {
        max_connections: Some(8),
        nodelay: true,
        keep_alive: None,
        timeout: Some(Duration::from_secs(2)),
    }
}

#[test]
fn test_tcp_server_config() {
    let config = TcpServerConfig::default();
    assert_eq!(config.max_connections, Some(1000));
    assert!(config.nodelay);
    assert_eq!(config.keep_alive, Some(Duration::from_secs(300)));
    assert_eq!(config.timeout, Some(Duration::from_secs(30)));
}

#[test]
fn test_udp_config() {
    let config = UdpConfig::default();
    assert_eq!(config.buffer_size, 65536);
    assert!(!config.broadcast);
    assert!(!config.multicast);
}

#[test]
fn test_server_stats() {
    let stats = ServerStats::default();
    assert_eq!(
        stats
            .total_connections
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        stats
            .active_connections
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        stats
            .bytes_received
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        stats.bytes_sent.load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn test_udp_stats() {
    let stats = UdpStats::default();
    assert_eq!(
        stats
            .packets_sent
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        stats
            .packets_received
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        stats.bytes_sent.load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        stats
            .bytes_received
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[test]
fn test_tcp_loopback_read_write_and_stats_values() {
    futures::executor::block_on(async {
        let listener = TcpListener::bind_with_config("127.0.0.1:0", test_config())
            .await
            .expect("listener bind must succeed");
        let addr = listener.local_addr().expect("listener address must exist");

        let client = std::thread::spawn(move || {
            let mut stream = StdTcpStream::connect(addr).expect("client connection must succeed");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("client read timeout must be set");
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .expect("client write timeout must be set");
            stream
                .write_all(b"ping")
                .expect("client write must succeed");

            let mut echo = [0_u8; 4];
            stream
                .read_exact(&mut echo)
                .expect("client echo read must succeed");
            assert_eq!(&echo, b"pong");
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        let (mut stream, _peer) = accept_before(&listener, deadline).await;

        let mut inbound = [0_u8; 4];
        let bytes_read = read_before(&mut stream, &mut inbound, deadline).await;
        assert_eq!(bytes_read, 4);
        assert_eq!(&inbound, b"ping");

        let bytes_written = write_all_before(&mut stream, b"pong", deadline).await;
        assert_eq!(bytes_written, 4);
        stream.flush().await.expect("server flush must succeed");

        client.join().expect("client thread must complete");

        let stats = listener.stats();
        assert_eq!(stats.total_connections, 1);
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.bytes_received, 4);
        assert_eq!(stats.bytes_sent, 4);
    });
}

#[test]
fn test_tcp_shutdown_write_sends_eof_and_stats_values() {
    futures::executor::block_on(async {
        let listener = TcpListener::bind_with_config("127.0.0.1:0", test_config())
            .await
            .expect("listener bind must succeed");
        let addr = listener.local_addr().expect("listener address must exist");

        let client = std::thread::spawn(move || {
            let mut stream = StdTcpStream::connect(addr).expect("client connection must succeed");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("client read timeout must be set");
            let mut received = Vec::new();
            stream
                .read_to_end(&mut received)
                .expect("client must observe EOF after shutdown");
            assert_eq!(&received, b"closed");
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        let (mut stream, _peer) = accept_before(&listener, deadline).await;
        AsyncWriteExt::write_all(&mut stream, b"closed")
            .await
            .expect("server write_all must succeed");
        AsyncWriteExt::shutdown(&mut stream)
            .await
            .expect("server shutdown must succeed");

        client.join().expect("client thread must complete");

        let stats = listener.stats();
        assert_eq!(stats.total_connections, 1);
        assert_eq!(stats.bytes_sent, 6);
    });
}

#[test]
fn test_tcp_poll_write_reports_pending_under_backpressure() {
    let (listener, addr) = backpressure_listener();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();

    let server = std::thread::spawn(move || {
        let (mut stream, _peer) = listener.accept().expect("server accept must succeed");
        release_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("client must release backpressure server");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("server read timeout must be set");

        let mut drained = 0;
        let mut buf = [0_u8; 8 * 1024];
        loop {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(count) => drained += count,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::ConnectionReset
                    ) =>
                {
                    break;
                }
                Err(error) => panic!("server drain failed: {error}"),
            }
        }
        drained
    });

    let client = backpressure_client(addr);
    let mut stream = TcpStream::from_std(client).expect("moirai stream must wrap std stream");
    let written = poll_write_until_pending(&mut stream);
    assert!(written > 0);
    assert!(written <= BACKPRESSURE_MAX_BYTES);

    release_sender
        .send(())
        .expect("release signal must reach server");
    drop(stream);

    let drained = server.join().expect("server thread must join");
    assert!(drained > 0);
}

#[test]
fn test_tcp_poll_read_reports_pending_before_peer_data() {
    let (listener, addr) = readiness_listener();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();

    let server = std::thread::spawn(move || {
        let (mut stream, _peer) = listener.accept().expect("server accept must succeed");
        stream
            .set_nodelay(true)
            .expect("server nodelay must be set");
        release_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("client must release readiness server");
        stream
            .write_all(&READINESS_PAYLOAD)
            .expect("server payload write must succeed");
        stream.flush().expect("server flush must succeed");
    });

    let client = readiness_client(addr);
    let mut stream = TcpStream::from_std(client).expect("moirai stream must wrap std stream");
    let mut buf = [0_u8; READINESS_PAYLOAD.len()];

    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let pending = Pin::new(&mut stream).poll_read(&mut context, &mut buf);
    assert!(matches!(pending, Poll::Pending));

    release_sender
        .send(())
        .expect("release signal must reach server");
    let received = poll_read_until_ready(&mut stream, &mut buf);
    assert_eq!(received, READINESS_PAYLOAD.len());
    assert_eq!(buf, READINESS_PAYLOAD);

    server.join().expect("server thread must join");
}

#[test]
fn test_tcp_pending_read_future_drop_preserves_stream_payload() {
    let (listener, addr) = readiness_listener();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();

    let server = std::thread::spawn(move || {
        let (mut stream, _peer) = listener.accept().expect("server accept must succeed");
        stream
            .set_nodelay(true)
            .expect("server nodelay must be set");
        release_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("client must release readiness server");
        stream
            .write_all(&READINESS_PAYLOAD)
            .expect("server payload write must succeed");
        stream.flush().expect("server flush must succeed");
    });

    let client = readiness_client(addr);
    let mut stream = TcpStream::from_std(client).expect("moirai stream must wrap std stream");
    let mut cancelled_buf = [0_u8; READINESS_PAYLOAD.len()];

    {
        let mut future = std::pin::pin!(stream.read_exact(&mut cancelled_buf));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }

    assert_eq!(cancelled_buf, [0_u8; READINESS_PAYLOAD.len()]);

    release_sender
        .send(())
        .expect("release signal must reach server");
    let mut buf = [0_u8; READINESS_PAYLOAD.len()];
    let received = poll_read_until_ready(&mut stream, &mut buf);
    assert_eq!(received, READINESS_PAYLOAD.len());
    assert_eq!(buf, READINESS_PAYLOAD);

    server.join().expect("server thread must join");
}

#[test]
fn timeout_read_stale_socket_wake_does_not_repoll_completed_task() {
    let (listener, addr) = readiness_listener();
    let peer = StdTcpStream::connect(addr).expect("peer socket must connect");
    peer.set_nodelay(true).expect("peer nodelay must be set");
    let (accepted, _peer_addr) = listener.accept().expect("server accept must succeed");
    accepted
        .set_nodelay(true)
        .expect("accepted nodelay must be set");

    let executor = Arc::new(AsyncExecutor::new().expect("async executor must start"));
    let runner_executor = Arc::clone(&executor);
    let runner = std::thread::spawn(move || runner_executor.run());

    let handle = executor.spawn(async move {
        let mut stream = TcpStream::from_std(accepted).expect("accepted socket must wrap");
        let mut timed_out_buf = [0_u8; READINESS_PAYLOAD.len()];

        let result = timeout(Duration::from_millis(25), stream.read(&mut timed_out_buf)).await;

        assert_eq!(timed_out_buf, [0_u8; READINESS_PAYLOAD.len()]);
        assert!(
            matches!(result, Err(TimeoutError)),
            "socket read must time out before peer payload"
        );

        stream
    });

    let mut stream = futures::executor::block_on(handle);
    let events_before_write = executor
        .reactor()
        .metrics()
        .events_processed
        .load(Ordering::Relaxed);

    (&peer)
        .write_all(&READINESS_PAYLOAD)
        .expect("peer write must succeed");
    (&peer).flush().expect("peer flush must succeed");

    wait_for_reactor_event_after(&executor, events_before_write);

    let mut received = [0_u8; READINESS_PAYLOAD.len()];
    let bytes = futures::executor::block_on(stream.read(&mut received))
        .expect("returned stream must still read peer payload");
    assert_eq!(bytes, READINESS_PAYLOAD.len());
    assert_eq!(received, READINESS_PAYLOAD);

    executor.stop().expect("executor stop must wake reactor");
    runner
        .join()
        .expect("executor thread must not panic on stale socket wake")
        .expect("executor run must stop cleanly");
}

#[test]
fn test_udp_loopback_send_recv_and_stats_values() {
    futures::executor::block_on(async {
        let sender = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("sender bind must succeed");
        let receiver = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("receiver bind must succeed");
        let target = receiver.local_addr().expect("receiver address must exist");
        let source = sender.local_addr().expect("sender address must exist");

        let sent = sender
            .send_to(b"datagram", target)
            .await
            .expect("udp send must succeed");
        assert_eq!(sent, 8);

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut buf = [0_u8; 16];
        let (received, peer) = loop {
            match receiver.recv_from(&mut buf).await {
                Ok(received) => break received,
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    std::thread::yield_now();
                }
                Err(error) => panic!("udp receive failed: {error}"),
            }
        };

        assert_eq!(received, 8);
        assert_eq!(peer, source);
        assert_eq!(&buf[..received], b"datagram");

        let send_stats = sender.stats();
        assert_eq!(send_stats.packets_sent, 1);
        assert_eq!(send_stats.bytes_sent, 8);
        assert_eq!(send_stats.packets_received, 0);
        assert_eq!(send_stats.bytes_received, 0);

        let recv_stats = receiver.stats();
        assert_eq!(recv_stats.packets_sent, 0);
        assert_eq!(recv_stats.bytes_sent, 0);
        assert_eq!(recv_stats.packets_received, 1);
        assert_eq!(recv_stats.bytes_received, 8);
    });
}

#[test]
fn connection_pool_assigns_unique_ids_and_removes_by_id() {
    let pool = ConnectionPool::new(Some(4));
    let addr: SocketAddr = "127.0.0.1:5000".parse().unwrap();

    let id_a = pool.add_connection(addr);
    let id_b = pool.add_connection(addr); // same peer address, distinct connection
    assert_ne!(
        id_a, id_b,
        "ids must be unique even for an identical peer addr"
    );
    assert_eq!(pool.connection_count(), 2);

    // Removing one connection by id leaves the other intact (no address collision).
    assert!(pool.remove_connection(id_a));
    assert_eq!(pool.connection_count(), 1);
    assert!(
        !pool.remove_connection(id_a),
        "double remove must be a no-op"
    );
    assert!(pool.remove_connection(id_b));
    assert_eq!(pool.connection_count(), 0);

    // The retained ConnectionInfo carries the captured peer address.
    let id_c = pool.add_connection(addr);
    assert_eq!(pool.get_active_connections()[&id_c].peer_addr, addr);
}

#[test]
fn connection_pool_reservation_accounting_is_balanced() {
    let pool = ConnectionPool::new(Some(2));
    let addr: SocketAddr = "127.0.0.1:5001".parse().unwrap();

    assert!(pool.try_reserve());
    assert!(pool.try_reserve());
    assert!(!pool.try_reserve(), "two reservations exhaust the cap of 2");

    // Cancelling a reservation (the cancel-leak path) restores capacity exactly.
    pool.cancel_reservation();
    assert!(pool.has_capacity());
    assert!(pool.try_reserve());

    // Converting reservations into tracked connections releases them so the cap
    // continues to reflect reserved + active, never double-counting.
    let _id0 = pool.add_connection_reserved(addr);
    let _id1 = pool.add_connection_reserved(addr);
    assert_eq!(pool.connection_count(), 2);
    assert!(!pool.has_capacity());
}

#[test]
fn listener_accept_cancellation_does_not_leak_reservations() {
    futures::executor::block_on(async {
        let config = TcpServerConfig {
            max_connections: Some(2),
            nodelay: true,
            keep_alive: None,
            timeout: Some(Duration::from_secs(2)),
        };
        let listener = TcpListener::bind_with_config("127.0.0.1:0", config)
            .await
            .expect("listener bind must succeed");
        let addr = listener.local_addr().expect("listener address must exist");

        // Poll-then-drop several pending accept futures. Each reserves a slot on
        // first poll and is cancelled while `inner.accept()` is pending. Without
        // the RAII reservation guard these would leak and permanently exhaust the
        // cap of 2, making every real accept below fail with WouldBlock.
        for _ in 0..5 {
            let mut accept = std::pin::pin!(listener.accept());
            let waker = futures::task::noop_waker();
            let mut context = Context::from_waker(&waker);
            assert!(matches!(
                Future::poll(accept.as_mut(), &mut context),
                Poll::Pending
            ));
        }

        // The full cap must still be available: accept two real connections.
        let client_a = std::thread::spawn(move || StdTcpStream::connect(addr).unwrap());
        let deadline = Instant::now() + Duration::from_secs(2);
        let (stream_a, _peer_a) = accept_before(&listener, deadline).await;
        let conn_a = client_a.join().unwrap();

        let client_b = std::thread::spawn(move || StdTcpStream::connect(addr).unwrap());
        let (stream_b, _peer_b) = accept_before(&listener, deadline).await;
        let conn_b = client_b.join().unwrap();

        assert_eq!(listener.stats().active_connections, 2);

        // Dropping a tracked stream untracks exactly that connection by id, even
        // though its peer socket may be torn down, restoring one slot.
        drop(stream_a);
        drop(conn_a);
        assert_eq!(listener.stats().active_connections, 1);

        // The freed slot is reusable: a third connection now accepts.
        let client_c = std::thread::spawn(move || StdTcpStream::connect(addr).unwrap());
        let (stream_c, _peer_c) = accept_before(&listener, deadline).await;
        let conn_c = client_c.join().unwrap();
        assert_eq!(listener.stats().active_connections, 2);

        drop(stream_b);
        drop(stream_c);
        drop((conn_b, conn_c));
    });
}

async fn accept_before(listener: &TcpListener, deadline: Instant) -> (TcpStream, SocketAddr) {
    loop {
        match listener.accept().await {
            Ok(accepted) => break accepted,
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                std::thread::yield_now();
            }
            Err(error) => panic!("listener accept failed: {error}"),
        }
    }
}

async fn read_before(stream: &mut TcpStream, buf: &mut [u8], deadline: Instant) -> usize {
    loop {
        match stream.read(buf).await {
            Ok(bytes_read) => break bytes_read,
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                std::thread::yield_now();
            }
            Err(error) => panic!("server read failed: {error}"),
        }
    }
}

async fn write_all_before(stream: &mut TcpStream, buf: &[u8], deadline: Instant) -> usize {
    let mut bytes_written = 0;
    while bytes_written < buf.len() {
        match stream.write(&buf[bytes_written..]).await {
            Ok(0) => panic!("server write made no progress"),
            Ok(written) => bytes_written += written,
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                std::thread::yield_now();
            }
            Err(error) => panic!("server write failed: {error}"),
        }
    }
    bytes_written
}

fn backpressure_listener() -> (std::net::TcpListener, SocketAddr) {
    let socket = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )
    .expect("server socket must be created");
    socket
        .set_reuse_address(true)
        .expect("server socket reuse must be set");
    socket
        .set_recv_buffer_size(BACKPRESSURE_BUFFER_BYTES)
        .expect("server receive buffer must be bounded");
    socket
        .bind(&socket2::SockAddr::from(SocketAddrV4::new(
            Ipv4Addr::LOCALHOST,
            0,
        )))
        .expect("server socket must bind");
    socket.listen(1).expect("server socket must listen");
    let listener: std::net::TcpListener = socket.into();
    let addr = listener.local_addr().expect("server address must exist");
    (listener, addr)
}

fn backpressure_client(addr: SocketAddr) -> StdTcpStream {
    let socket = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )
    .expect("client socket must be created");
    socket
        .set_send_buffer_size(BACKPRESSURE_BUFFER_BYTES)
        .expect("client send buffer must be bounded");
    socket
        .connect(&socket2::SockAddr::from(addr))
        .expect("client socket must connect");
    let stream: StdTcpStream = socket.into();
    stream
        .set_nodelay(true)
        .expect("client nodelay must be set");
    stream
}

fn readiness_listener() -> (std::net::TcpListener, SocketAddr) {
    let listener = std::net::TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
        .expect("server socket must bind");
    let addr = listener.local_addr().expect("server address must exist");
    (listener, addr)
}

fn readiness_client(addr: SocketAddr) -> StdTcpStream {
    let stream = StdTcpStream::connect(addr).expect("client socket must connect");
    stream
        .set_nodelay(true)
        .expect("client nodelay must be set");
    stream
}

fn poll_write_until_pending(stream: &mut TcpStream) -> usize {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let mut written = 0;

    loop {
        match Pin::new(&mut *stream).poll_write(&mut context, &BACKPRESSURE_CHUNK) {
            Poll::Ready(Ok(0)) => panic!("poll_write made no progress"),
            Poll::Ready(Ok(count)) => {
                written += count;
                assert!(written <= BACKPRESSURE_MAX_BYTES);
            }
            Poll::Ready(Err(error)) if error.kind() == io::ErrorKind::WouldBlock => break written,
            Poll::Ready(Err(error)) => panic!("poll_write failed: {error}"),
            Poll::Pending => break written,
        }
    }
}

fn poll_read_until_ready(stream: &mut TcpStream, buf: &mut [u8]) -> usize {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut received = 0;

    while received < buf.len() {
        match Pin::new(&mut *stream).poll_read(&mut context, &mut buf[received..]) {
            Poll::Ready(Ok(0)) => panic!("poll_read reached EOF before payload"),
            Poll::Ready(Ok(count)) => received += count,
            Poll::Ready(Err(error))
                if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                std::thread::yield_now();
            }
            Poll::Ready(Err(error)) => panic!("poll_read failed: {error}"),
            Poll::Pending if Instant::now() < deadline => std::thread::yield_now(),
            Poll::Pending => panic!("poll_read did not become ready before deadline"),
        }
    }

    received
}

fn wait_for_reactor_event_after(executor: &AsyncExecutor, previous_events: u64) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while executor
        .reactor()
        .metrics()
        .events_processed
        .load(Ordering::Relaxed)
        <= previous_events
    {
        assert!(
            Instant::now() < deadline,
            "reactor must process the peer read-readiness event"
        );
        std::thread::yield_now();
    }
}
