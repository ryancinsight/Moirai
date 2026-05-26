//! Async TCP read-readiness comparison against Tokio.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_async::io::AsyncRead as MoiraiAsyncRead;
use std::io::{self, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream as StdTcpStream};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::io::ReadBuf;
use tokio::runtime::Builder;

const SAMPLE_SIZE: usize = 20;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const READINESS_PAYLOAD_LEN: usize = 5;
const READINESS_PAYLOAD: [u8; READINESS_PAYLOAD_LEN] = *b"ready";
const IO_TIMEOUT: Duration = Duration::from_secs(2);

fn spawn_readiness_server() -> (SocketAddr, std::sync::mpsc::Sender<()>, JoinHandle<()>) {
    let listener = std::net::TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
        .expect("server socket bind");
    let addr = listener.local_addr().expect("server local address");
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let handle = thread::spawn(move || {
        let (mut stream, _peer) = listener.accept().expect("server accept");
        stream.set_nodelay(true).expect("server nodelay");
        release_receiver
            .recv_timeout(IO_TIMEOUT)
            .expect("client must release readiness server");
        stream
            .write_all(&READINESS_PAYLOAD)
            .expect("server payload write");
        stream.flush().expect("server flush");
    });
    (addr, release_sender, handle)
}

fn readiness_client(addr: SocketAddr) -> StdTcpStream {
    let stream = StdTcpStream::connect(addr).expect("client connect");
    stream.set_nodelay(true).expect("client nodelay");
    stream
        .set_nonblocking(true)
        .expect("client nonblocking mode");
    stream
}

fn poll_moirai_before_peer_data(stream: &mut moirai_async::net::TcpStream) {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let mut buf = [0_u8; READINESS_PAYLOAD_LEN];

    let pending = Pin::new(stream).poll_read(&mut context, &mut buf);
    assert!(matches!(pending, Poll::Pending));
}

fn poll_tokio_before_peer_data(stream: &mut tokio::net::TcpStream) {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let mut buf = [0_u8; READINESS_PAYLOAD_LEN];
    let mut read_buf = ReadBuf::new(&mut buf);

    let pending = tokio::io::AsyncRead::poll_read(Pin::new(stream), &mut context, &mut read_buf);
    assert!(matches!(pending, Poll::Pending));
}

fn read_moirai_payload(stream: &mut moirai_async::net::TcpStream) -> [u8; READINESS_PAYLOAD_LEN] {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let deadline = Instant::now() + IO_TIMEOUT;
    let mut buf = [0_u8; READINESS_PAYLOAD_LEN];
    let mut received = 0;

    while received < buf.len() {
        match Pin::new(&mut *stream).poll_read(&mut context, &mut buf[received..]) {
            Poll::Ready(Ok(0)) => panic!("moirai read reached EOF before payload"),
            Poll::Ready(Ok(count)) => received += count,
            Poll::Ready(Err(error))
                if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                thread::yield_now();
            }
            Poll::Ready(Err(error)) => panic!("moirai read failed: {error}"),
            Poll::Pending if Instant::now() < deadline => thread::yield_now(),
            Poll::Pending => panic!("moirai read did not become ready before deadline"),
        }
    }

    buf
}

fn read_tokio_payload(
    runtime: &tokio::runtime::Runtime,
    stream: &tokio::net::TcpStream,
) -> [u8; READINESS_PAYLOAD_LEN] {
    runtime
        .block_on(async {
            tokio::time::timeout(IO_TIMEOUT, async {
                let mut buf = [0_u8; READINESS_PAYLOAD_LEN];
                let mut received = 0;

                while received < buf.len() {
                    stream.readable().await?;
                    match stream.try_read(&mut buf[received..]) {
                        Ok(0) => {
                            return Err(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "tokio read reached EOF before payload",
                            ));
                        }
                        Ok(count) => received += count,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
                        Err(error) => return Err(error),
                    }
                }

                Ok::<[u8; READINESS_PAYLOAD_LEN], io::Error>(buf)
            })
            .await
        })
        .expect("tokio read readiness timeout")
        .expect("tokio read readiness payload")
}

fn moirai_tcp_read_readiness_once() -> usize {
    let (addr, release_sender, server) = spawn_readiness_server();
    let client = readiness_client(addr);
    let mut stream =
        moirai_async::net::TcpStream::from_std(client).expect("moirai stream from std");

    poll_moirai_before_peer_data(&mut stream);
    release_sender.send(()).expect("release server");
    let buf = read_moirai_payload(&mut stream);
    assert_eq!(buf, READINESS_PAYLOAD);
    server.join().expect("server join");
    buf.len()
}

fn tokio_tcp_read_readiness_once(runtime: &tokio::runtime::Runtime) -> usize {
    let (addr, release_sender, server) = spawn_readiness_server();
    let client = readiness_client(addr);
    let mut stream = {
        let _guard = runtime.enter();
        tokio::net::TcpStream::from_std(client).expect("tokio stream from std")
    };

    {
        let _guard = runtime.enter();
        poll_tokio_before_peer_data(&mut stream);
    }

    release_sender.send(()).expect("release server");
    let buf = read_tokio_payload(runtime, &stream);
    assert_eq!(buf, READINESS_PAYLOAD);
    server.join().expect("server join");
    buf.len()
}

fn async_tcp_readiness_comparison(c: &mut Criterion) {
    let runtime = Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio benchmark runtime must build");

    assert_eq!(moirai_tcp_read_readiness_once(), READINESS_PAYLOAD.len());
    assert_eq!(
        tokio_tcp_read_readiness_once(&runtime),
        READINESS_PAYLOAD.len()
    );

    let mut group = c.benchmark_group("async_tcp_read_readiness");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", READINESS_PAYLOAD.len()),
        &READINESS_PAYLOAD.len(),
        |b, _| b.iter(|| black_box(moirai_tcp_read_readiness_once())),
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", READINESS_PAYLOAD.len()),
        &READINESS_PAYLOAD.len(),
        |b, _| b.iter(|| black_box(tokio_tcp_read_readiness_once(black_box(&runtime)))),
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
    targets = async_tcp_readiness_comparison
}
criterion_main!(benches);
