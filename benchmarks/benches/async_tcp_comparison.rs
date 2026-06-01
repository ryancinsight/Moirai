//! Async TCP facade comparison benchmarks against Tokio TCP sockets.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_async::io::{AsyncReadExt as MoiraiAsyncReadExt, AsyncWriteExt as MoiraiAsyncWriteExt};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream as StdTcpStream};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::runtime::Builder;

const SAMPLE_SIZE: usize = 20;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const CLIENT_PAYLOAD: &[u8] = b"moirai-tcp-loopback-ping";
const SERVER_PAYLOAD: &[u8] = b"moirai-tcp-loopback-pong";
const SHUTDOWN_PAYLOAD: &[u8] = b"moirai-tcp-shutdown";
const IO_TIMEOUT: Duration = Duration::from_secs(2);

fn client_echo_roundtrip(addr: SocketAddr) -> [u8; SERVER_PAYLOAD.len()] {
    let mut stream = StdTcpStream::connect(addr).expect("std TCP client must connect");
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .expect("std TCP client read timeout must be set");
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .expect("std TCP client write timeout must be set");
    stream
        .set_nodelay(true)
        .expect("std TCP client nodelay must be set");
    stream
        .write_all(CLIENT_PAYLOAD)
        .expect("std TCP client request write must succeed");

    let mut echo = [0_u8; SERVER_PAYLOAD.len()];
    stream
        .read_exact(&mut echo)
        .expect("std TCP client echo read must succeed");
    assert_eq!(&echo, SERVER_PAYLOAD);
    echo
}

fn client_read_to_end(addr: SocketAddr) -> Vec<u8> {
    let mut stream = StdTcpStream::connect(addr).expect("std TCP read client must connect");
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .expect("std TCP read client timeout must be set");
    stream
        .set_nodelay(true)
        .expect("std TCP read client nodelay must be set");

    let mut received = Vec::with_capacity(SHUTDOWN_PAYLOAD.len());
    stream
        .read_to_end(&mut received)
        .expect("std TCP read client must observe EOF");
    assert_eq!(&received, SHUTDOWN_PAYLOAD);
    received
}

fn spawn_echo_server() -> (SocketAddr, JoinHandle<usize>) {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").expect("std TCP echo server must bind");
    let addr = listener.local_addr().expect("std TCP echo server address");
    let handle = thread::spawn(move || {
        let (mut stream, _peer) = listener
            .accept()
            .expect("std TCP echo server must accept one client");
        stream
            .set_read_timeout(Some(IO_TIMEOUT))
            .expect("std TCP echo server read timeout must be set");
        stream
            .set_write_timeout(Some(IO_TIMEOUT))
            .expect("std TCP echo server write timeout must be set");
        stream
            .set_nodelay(true)
            .expect("std TCP echo server nodelay must be set");

        let mut completed = 0_usize;
        loop {
            let mut request = [0_u8; CLIENT_PAYLOAD.len()];
            match stream.read_exact(&mut request) {
                Ok(()) => {
                    assert_eq!(&request, CLIENT_PAYLOAD);
                    stream
                        .write_all(SERVER_PAYLOAD)
                        .expect("std TCP echo server response write must succeed");
                    completed += 1;
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::UnexpectedEof
                            | io::ErrorKind::ConnectionAborted
                            | io::ErrorKind::ConnectionReset
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::WouldBlock
                    ) =>
                {
                    break;
                }
                Err(error) => panic!("std TCP echo server failed: {error}"),
            }
        }
        completed
    });
    (addr, handle)
}

fn moirai_tcp_shutdown_once(
    runtime: &moirai::Moirai,
    listener: &moirai_async::net::TcpListener,
) -> usize {
    let addr = listener
        .local_addr()
        .expect("moirai shutdown listener address");
    let client = thread::spawn(move || client_read_to_end(addr));

    runtime
        .block_on(async {
            let (mut stream, _peer) = listener.accept().await?;
            stream.set_nodelay(true)?;
            MoiraiAsyncWriteExt::write_all(&mut stream, SHUTDOWN_PAYLOAD).await?;
            MoiraiAsyncWriteExt::shutdown(&mut stream).await
        })
        .expect("moirai TCP shutdown must succeed");

    let received = client
        .join()
        .expect("std TCP shutdown client thread must complete");
    assert_eq!(&received, SHUTDOWN_PAYLOAD);
    received.len()
}

fn tokio_tcp_shutdown_once(
    runtime: &tokio::runtime::Runtime,
    listener: &tokio::net::TcpListener,
) -> usize {
    let addr = listener
        .local_addr()
        .expect("tokio shutdown listener address");
    let client = thread::spawn(move || client_read_to_end(addr));

    runtime
        .block_on(async {
            let (mut stream, _peer) = listener.accept().await?;
            stream.set_nodelay(true)?;
            tokio::io::AsyncWriteExt::write_all(&mut stream, SHUTDOWN_PAYLOAD).await?;
            tokio::io::AsyncWriteExt::shutdown(&mut stream).await
        })
        .expect("tokio TCP shutdown must succeed");

    let received = client
        .join()
        .expect("std TCP shutdown client thread must complete");
    assert_eq!(&received, SHUTDOWN_PAYLOAD);
    received.len()
}

fn moirai_tcp_stream_echo_once(
    runtime: &moirai::Moirai,
    stream: &mut moirai_async::net::TcpStream,
) -> [u8; SERVER_PAYLOAD.len()] {
    let mut echo = [0_u8; SERVER_PAYLOAD.len()];
    runtime
        .block_on(async {
            MoiraiAsyncWriteExt::write_all(stream, CLIENT_PAYLOAD).await?;
            stream.flush().await?;
            MoiraiAsyncReadExt::read_exact(stream, &mut echo).await
        })
        .expect("moirai TCP stream echo must succeed");
    assert_eq!(&echo, SERVER_PAYLOAD);
    echo
}

fn tokio_tcp_stream_echo_once(
    runtime: &tokio::runtime::Runtime,
    stream: &mut tokio::net::TcpStream,
) -> [u8; SERVER_PAYLOAD.len()] {
    let mut echo = [0_u8; SERVER_PAYLOAD.len()];
    runtime
        .block_on(async {
            stream.write_all(CLIENT_PAYLOAD).await?;
            stream.flush().await?;
            stream.read_exact(&mut echo).await?;
            Ok::<(), io::Error>(())
        })
        .expect("tokio TCP stream echo must succeed");
    assert_eq!(&echo, SERVER_PAYLOAD);
    echo
}

fn moirai_tcp_echo_roundtrip(
    runtime: &moirai::Moirai,
    listener: &moirai_async::net::TcpListener,
) -> [u8; CLIENT_PAYLOAD.len()] {
    let addr = listener.local_addr().expect("moirai listener address");
    let client = thread::spawn(move || client_echo_roundtrip(addr));

    let mut request = [0_u8; CLIENT_PAYLOAD.len()];
    runtime
        .block_on(async {
            let (mut stream, _peer) = listener.accept().await?;
            stream.set_nodelay(true)?;
            let read = stream.read(&mut request).await?;
            assert_eq!(read, CLIENT_PAYLOAD.len());
            assert_eq!(&request, CLIENT_PAYLOAD);

            let mut written = 0;
            while written < SERVER_PAYLOAD.len() {
                let count = stream.write(&SERVER_PAYLOAD[written..]).await?;
                assert_ne!(count, 0);
                written += count;
            }
            stream.flush().await
        })
        .expect("moirai TCP echo roundtrip must succeed");

    let echo = client.join().expect("std TCP client thread must complete");
    assert_eq!(&echo, SERVER_PAYLOAD);
    request
}

fn tokio_tcp_echo_roundtrip(
    runtime: &tokio::runtime::Runtime,
    listener: &tokio::net::TcpListener,
) -> [u8; CLIENT_PAYLOAD.len()] {
    let addr = listener.local_addr().expect("tokio listener address");
    let client = thread::spawn(move || client_echo_roundtrip(addr));

    let mut request = [0_u8; CLIENT_PAYLOAD.len()];
    runtime
        .block_on(async {
            let (mut stream, _peer) = listener.accept().await?;
            stream.set_nodelay(true)?;
            stream.read_exact(&mut request).await?;
            assert_eq!(&request, CLIENT_PAYLOAD);
            stream.write_all(SERVER_PAYLOAD).await?;
            stream.flush().await
        })
        .expect("tokio TCP echo roundtrip must succeed");

    let echo = client.join().expect("std TCP client thread must complete");
    assert_eq!(&echo, SERVER_PAYLOAD);
    request
}

fn async_tcp_comparison(c: &mut Criterion) {
    let runtime = Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("tokio benchmark runtime must build");
    let moirai_runtime = moirai::Moirai::new().expect("moirai benchmark runtime must build");

    let moirai_listener = moirai_runtime
        .block_on(moirai_async::net::TcpListener::bind("127.0.0.1:0"))
        .expect("moirai TCP listener must bind");
    let tokio_listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("tokio TCP listener must bind");
    let moirai_shutdown_listener = moirai_runtime
        .block_on(moirai_async::net::TcpListener::bind("127.0.0.1:0"))
        .expect("moirai TCP shutdown listener must bind");
    let tokio_shutdown_listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("tokio TCP shutdown listener must bind");
    let moirai_expected = moirai_tcp_echo_roundtrip(&moirai_runtime, &moirai_listener);
    let tokio_expected = tokio_tcp_echo_roundtrip(&runtime, &tokio_listener);
    assert_eq!(&moirai_expected, CLIENT_PAYLOAD);
    assert_eq!(&tokio_expected, CLIENT_PAYLOAD);
    let moirai_shutdown_expected =
        moirai_tcp_shutdown_once(&moirai_runtime, &moirai_shutdown_listener);
    let tokio_shutdown_expected = tokio_tcp_shutdown_once(&runtime, &tokio_shutdown_listener);
    assert_eq!(moirai_shutdown_expected, SHUTDOWN_PAYLOAD.len());
    assert_eq!(tokio_shutdown_expected, SHUTDOWN_PAYLOAD.len());

    let mut group = c.benchmark_group("async_tcp_loopback_echo");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", CLIENT_PAYLOAD.len()),
        &CLIENT_PAYLOAD.len(),
        |b, _| {
            b.iter(|| {
                black_box(moirai_tcp_echo_roundtrip(
                    black_box(&moirai_runtime),
                    black_box(&moirai_listener),
                ))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", CLIENT_PAYLOAD.len()),
        &CLIENT_PAYLOAD.len(),
        |b, _| {
            b.iter(|| {
                black_box(tokio_tcp_echo_roundtrip(
                    black_box(&runtime),
                    black_box(&tokio_listener),
                ))
            })
        },
    );
    group.finish();

    let (moirai_stream_addr, moirai_stream_server) = spawn_echo_server();
    let (tokio_stream_addr, tokio_stream_server) = spawn_echo_server();
    let mut moirai_stream = moirai_runtime
        .block_on(moirai_async::net::TcpStream::connect(
            &moirai_stream_addr.to_string(),
        ))
        .expect("moirai TCP stream must connect");
    let mut tokio_stream = runtime
        .block_on(tokio::net::TcpStream::connect(tokio_stream_addr))
        .expect("tokio TCP stream must connect");
    moirai_stream
        .set_nodelay(true)
        .expect("moirai TCP stream nodelay must be set");
    tokio_stream
        .set_nodelay(true)
        .expect("tokio TCP stream nodelay must be set");

    let moirai_stream_expected = moirai_tcp_stream_echo_once(&moirai_runtime, &mut moirai_stream);
    let tokio_stream_expected = tokio_tcp_stream_echo_once(&runtime, &mut tokio_stream);
    assert_eq!(&moirai_stream_expected, SERVER_PAYLOAD);
    assert_eq!(&tokio_stream_expected, SERVER_PAYLOAD);

    let mut stream_group = c.benchmark_group("async_tcp_stream_echo");
    stream_group.sample_size(SAMPLE_SIZE);
    stream_group.bench_with_input(
        BenchmarkId::new("moirai", CLIENT_PAYLOAD.len()),
        &CLIENT_PAYLOAD.len(),
        |b, _| {
            b.iter(|| {
                black_box(moirai_tcp_stream_echo_once(
                    black_box(&moirai_runtime),
                    black_box(&mut moirai_stream),
                ))
            })
        },
    );
    stream_group.bench_with_input(
        BenchmarkId::new("tokio", CLIENT_PAYLOAD.len()),
        &CLIENT_PAYLOAD.len(),
        |b, _| {
            b.iter(|| {
                black_box(tokio_tcp_stream_echo_once(
                    black_box(&runtime),
                    black_box(&mut tokio_stream),
                ))
            })
        },
    );
    stream_group.finish();

    drop(moirai_stream);
    drop(tokio_stream);
    assert_ne!(
        moirai_stream_server
            .join()
            .expect("moirai echo server thread must join"),
        0
    );
    assert_ne!(
        tokio_stream_server
            .join()
            .expect("tokio echo server thread must join"),
        0
    );

    let mut shutdown_group = c.benchmark_group("async_tcp_write_shutdown");
    shutdown_group.sample_size(SAMPLE_SIZE);
    shutdown_group.bench_with_input(
        BenchmarkId::new("moirai", SHUTDOWN_PAYLOAD.len()),
        &SHUTDOWN_PAYLOAD.len(),
        |b, _| {
            b.iter(|| {
                black_box(moirai_tcp_shutdown_once(
                    black_box(&moirai_runtime),
                    black_box(&moirai_shutdown_listener),
                ))
            })
        },
    );
    shutdown_group.bench_with_input(
        BenchmarkId::new("tokio", SHUTDOWN_PAYLOAD.len()),
        &SHUTDOWN_PAYLOAD.len(),
        |b, _| {
            b.iter(|| {
                black_box(tokio_tcp_shutdown_once(
                    black_box(&runtime),
                    black_box(&tokio_shutdown_listener),
                ))
            })
        },
    );
    shutdown_group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = async_tcp_comparison
}
criterion_main!(benches);
