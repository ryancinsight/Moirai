#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod connect;
mod datagram;
mod resolve;
mod stream;

use super::*;
use crate::executor::AsyncExecutor;
use crate::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use crate::timer::{TimeoutError, timeout};
use std::future::Future;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream as StdTcpStream};
use std::pin::Pin;
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
    assert_eq!(config.buffers, UdpBufferConfig::default());
    assert!(!config.broadcast);
    assert_eq!(config.multicast, None);
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
