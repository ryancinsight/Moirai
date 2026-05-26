//! Async TCP write-readiness comparison under bounded socket buffers.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_async::io::AsyncWrite as MoiraiAsyncWrite;
use socket2::{Domain, Protocol, SockAddr, Socket, Type};
use std::io::{self, Read};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream as StdTcpStream};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::runtime::Builder;

const SAMPLE_SIZE: usize = 20;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const SOCKET_BUFFER_BYTES: usize = 4 * 1024;
const WRITE_CHUNK: [u8; 16 * 1024] = [0xA5; 16 * 1024];
const MAX_WRITTEN_BYTES: usize = 16 * 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(2);

fn listener_with_bounded_receive_buffer() -> (std::net::TcpListener, SocketAddr) {
    let socket =
        Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).expect("server socket");
    socket.set_reuse_address(true).expect("server socket reuse");
    socket
        .set_recv_buffer_size(SOCKET_BUFFER_BYTES)
        .expect("server receive buffer");
    socket
        .bind(&SockAddr::from(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)))
        .expect("server bind");
    socket.listen(1).expect("server listen");
    let listener: std::net::TcpListener = socket.into();
    let addr = listener.local_addr().expect("server local address");
    (listener, addr)
}

fn client_with_bounded_send_buffer(addr: SocketAddr) -> StdTcpStream {
    let socket =
        Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).expect("client socket");
    socket
        .set_send_buffer_size(SOCKET_BUFFER_BYTES)
        .expect("client send buffer");
    socket
        .connect(&SockAddr::from(addr))
        .expect("client connect");
    let stream: StdTcpStream = socket.into();
    stream.set_nodelay(true).expect("client nodelay");
    stream
        .set_nonblocking(true)
        .expect("client nonblocking mode");
    stream
}

fn spawn_backpressure_server() -> (SocketAddr, std::sync::mpsc::Sender<()>, JoinHandle<usize>) {
    let (listener, addr) = listener_with_bounded_receive_buffer();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let handle = thread::spawn(move || {
        let (mut stream, _peer) = listener.accept().expect("server accept");
        release_receiver
            .recv_timeout(IO_TIMEOUT)
            .expect("client must release backpressure server");
        stream
            .set_read_timeout(Some(IO_TIMEOUT))
            .expect("server read timeout");

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
    (addr, release_sender, handle)
}

fn poll_moirai_until_backpressured(stream: &mut moirai_async::net::TcpStream) -> usize {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let mut written = 0;

    loop {
        match Pin::new(&mut *stream).poll_write(&mut context, &WRITE_CHUNK) {
            Poll::Ready(Ok(0)) => panic!("moirai poll_write made no progress"),
            Poll::Ready(Ok(count)) => {
                written += count;
                assert!(written <= MAX_WRITTEN_BYTES);
            }
            Poll::Ready(Err(error)) if error.kind() == io::ErrorKind::WouldBlock => break written,
            Poll::Ready(Err(error)) => panic!("moirai poll_write failed: {error}"),
            Poll::Pending => break written,
        }
    }
}

fn poll_tokio_until_backpressured(stream: &mut tokio::net::TcpStream) -> usize {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let mut written = 0;

    loop {
        match tokio::io::AsyncWrite::poll_write(Pin::new(&mut *stream), &mut context, &WRITE_CHUNK)
        {
            Poll::Ready(Ok(0)) => panic!("tokio poll_write made no progress"),
            Poll::Ready(Ok(count)) => {
                written += count;
                assert!(written <= MAX_WRITTEN_BYTES);
            }
            Poll::Ready(Err(error)) if error.kind() == io::ErrorKind::WouldBlock => break written,
            Poll::Ready(Err(error)) => panic!("tokio poll_write failed: {error}"),
            Poll::Pending => break written,
        }
    }
}

fn moirai_tcp_write_backpressure_once() -> usize {
    let (addr, release_sender, server) = spawn_backpressure_server();
    let client = client_with_bounded_send_buffer(addr);
    let mut stream =
        moirai_async::net::TcpStream::from_std(client).expect("moirai stream from std");
    let written = poll_moirai_until_backpressured(&mut stream);
    assert!(written > 0);
    release_sender.send(()).expect("release server");
    drop(stream);
    assert!(server.join().expect("server join") > 0);
    written
}

fn tokio_tcp_write_backpressure_once(runtime: &tokio::runtime::Runtime) -> usize {
    let (addr, release_sender, server) = spawn_backpressure_server();
    let client = client_with_bounded_send_buffer(addr);
    let mut stream = {
        let _guard = runtime.enter();
        tokio::net::TcpStream::from_std(client).expect("tokio stream from std")
    };
    runtime
        .block_on(stream.writable())
        .expect("tokio stream must reach initial writable readiness");
    let written = {
        let _guard = runtime.enter();
        poll_tokio_until_backpressured(&mut stream)
    };
    assert!(written > 0);
    release_sender.send(()).expect("release server");
    drop(stream);
    assert!(server.join().expect("server join") > 0);
    written
}

fn async_tcp_backpressure_comparison(c: &mut Criterion) {
    let runtime = Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("tokio benchmark runtime must build");

    let moirai_expected = moirai_tcp_write_backpressure_once();
    let tokio_expected = tokio_tcp_write_backpressure_once(&runtime);
    assert!(moirai_expected > 0);
    assert!(tokio_expected > 0);

    let mut group = c.benchmark_group("async_tcp_write_backpressure");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", WRITE_CHUNK.len()),
        &WRITE_CHUNK.len(),
        |b, _| b.iter(|| black_box(moirai_tcp_write_backpressure_once())),
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", WRITE_CHUNK.len()),
        &WRITE_CHUNK.len(),
        |b, _| b.iter(|| black_box(tokio_tcp_write_backpressure_once(black_box(&runtime)))),
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = async_tcp_backpressure_comparison
}
criterion_main!(benches);
