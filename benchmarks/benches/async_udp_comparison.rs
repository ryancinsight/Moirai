//! Async UDP facade comparison benchmarks against Tokio UDP sockets.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::io;
use std::net::UdpSocket as StdUdpSocket;
use std::time::{Duration, Instant};
use tokio::runtime::Builder;

const SAMPLE_SIZE: usize = 30;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const PAYLOAD: &[u8] = b"moirai-udp-loopback-payload";
const RECV_TIMEOUT: Duration = Duration::from_secs(2);

fn std_sender() -> StdUdpSocket {
    let socket = StdUdpSocket::bind("127.0.0.1:0").expect("std UDP sender must bind");
    socket
        .set_nonblocking(false)
        .expect("std UDP sender must remain blocking");
    socket
}

fn recv_moirai_payload(
    runtime: &moirai::Moirai,
    receiver: &moirai_async::net::UdpSocket,
    sender: &StdUdpSocket,
    buf: &mut [u8; PAYLOAD.len()],
) -> [u8; PAYLOAD.len()] {
    sender
        .send_to(
            PAYLOAD,
            receiver.local_addr().expect("moirai receiver address"),
        )
        .expect("std sender must send to moirai receiver");

    let deadline = Instant::now() + RECV_TIMEOUT;
    loop {
        match runtime.block_on(receiver.recv_from(buf)) {
            Ok((received, _peer)) => {
                assert_eq!(received, PAYLOAD.len());
                return *buf;
            }
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                std::thread::yield_now();
            }
            Err(error) => panic!("moirai UDP receive failed: {error}"),
        }
    }
}

fn recv_tokio_payload(
    runtime: &tokio::runtime::Runtime,
    receiver: &tokio::net::UdpSocket,
    sender: &StdUdpSocket,
    buf: &mut [u8; PAYLOAD.len()],
) -> [u8; PAYLOAD.len()] {
    sender
        .send_to(
            PAYLOAD,
            receiver.local_addr().expect("tokio receiver address"),
        )
        .expect("std sender must send to tokio receiver");

    let received = runtime
        .block_on(receiver.recv_from(buf))
        .expect("tokio UDP receive must succeed")
        .0;
    assert_eq!(received, PAYLOAD.len());
    *buf
}

fn async_udp_comparison(c: &mut Criterion) {
    let runtime = Builder::new_current_thread()
        .enable_io()
        .build()
        .expect("tokio benchmark runtime must build");
    let moirai_runtime = moirai::Moirai::new().expect("moirai benchmark runtime must build");

    let moirai_receiver = moirai_runtime
        .block_on(moirai_async::net::UdpSocket::bind("127.0.0.1:0"))
        .expect("moirai UDP receiver must bind");
    let tokio_receiver = runtime
        .block_on(tokio::net::UdpSocket::bind("127.0.0.1:0"))
        .expect("tokio UDP receiver must bind");
    let moirai_sender = std_sender();
    let tokio_sender = std_sender();

    let mut moirai_buf = [0_u8; PAYLOAD.len()];
    let mut tokio_buf = [0_u8; PAYLOAD.len()];
    let moirai_expected = recv_moirai_payload(
        &moirai_runtime,
        &moirai_receiver,
        &moirai_sender,
        &mut moirai_buf,
    );
    let tokio_expected =
        recv_tokio_payload(&runtime, &tokio_receiver, &tokio_sender, &mut tokio_buf);
    assert_eq!(moirai_expected, PAYLOAD);
    assert_eq!(tokio_expected, PAYLOAD);

    let mut group = c.benchmark_group("async_udp_loopback_recv_from");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", PAYLOAD.len()),
        &PAYLOAD.len(),
        |b, _| {
            let mut buf = [0_u8; PAYLOAD.len()];
            b.iter(|| {
                black_box(recv_moirai_payload(
                    black_box(&moirai_runtime),
                    black_box(&moirai_receiver),
                    black_box(&moirai_sender),
                    black_box(&mut buf),
                ))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("tokio", PAYLOAD.len()),
        &PAYLOAD.len(),
        |b, _| {
            let mut buf = [0_u8; PAYLOAD.len()];
            b.iter(|| {
                black_box(recv_tokio_payload(
                    black_box(&runtime),
                    black_box(&tokio_receiver),
                    black_box(&tokio_sender),
                    black_box(&mut buf),
                ))
            })
        },
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
    targets = async_udp_comparison
}
criterion_main!(benches);
