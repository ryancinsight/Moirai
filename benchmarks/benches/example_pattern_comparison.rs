//! Criterion benchmarks for example-level Rayon, Tokio, and Moirai patterns.
//!
//! The runnable examples print one-off timings. This benchmark keeps the same
//! workload shapes but runs them under Criterion so regressions can be compared
//! with distributions instead of a single wall-clock sample.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai::Moirai;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::time::Duration;
use tokio::sync::mpsc;

const BENCHMARK_SAMPLE_SIZE: usize = 10;
const BENCHMARK_MEASUREMENT_SECONDS: u64 = 1;
const BENCHMARK_WARM_UP_MILLIS: u64 = 250;
const WORKER_THREADS: usize = 4;

const RAYON_ITEMS: usize = 65_536;
const RAYON_ROUNDS: usize = 16;
const TOKIO_FANOUT_TASKS: usize = 256;
const TOKIO_FANOUT_ROUNDS: usize = 12;
const CHANNEL_ITEMS: usize = 40_000;
const MULTI_PRODUCER_ITEMS: usize = 60_000;
const PRODUCERS: usize = 4;
const CHANNEL_CAPACITY: usize = 512;
const CHANNEL_ROUNDS: usize = 8;

#[derive(Clone, Copy)]
enum Work {
    Item(usize),
    Stop,
}

#[inline]
fn rayon_transform(index: usize) -> u64 {
    let mut value = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);

    for round in 0..RAYON_ROUNDS {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        value = value.wrapping_add(round as u64);
    }

    value
}

#[inline]
fn fanout_delay(index: usize) -> Duration {
    Duration::from_millis(((index % 4) + 1) as u64)
}

#[inline]
fn fanout_payload(index: usize) -> u64 {
    let mut value = (index as u64).wrapping_mul(0x0000_0100_0000_01b3);

    for round in 0..TOKIO_FANOUT_ROUNDS {
        value ^= value.rotate_left(13);
        value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
        value ^= value >> 33;
        value = value.wrapping_add(round as u64);
    }

    value
}

#[inline]
fn channel_transform(index: usize) -> u64 {
    let mut value = (index as u64).wrapping_add(0x517c_c1b7_2722_0a95);

    for round in 0..CHANNEL_ROUNDS {
        value ^= value.rotate_right(17);
        value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        value ^= value >> 29;
        value = value.wrapping_add(round as u64);
    }

    value
}

fn producer_range(producer: usize, count: usize) -> std::ops::Range<usize> {
    let base = count / PRODUCERS;
    let remainder = count % PRODUCERS;
    let start = producer * base + producer.min(remainder);
    let len = base + usize::from(producer < remainder);
    start..(start + len)
}

fn verify_checksum(actual: u64, expected: u64) -> u64 {
    assert_eq!(actual, expected);
    black_box(actual)
}

fn rayon_reference(count: usize) -> u64 {
    (0..count)
        .map(rayon_transform)
        .fold(0_u64, u64::wrapping_add)
}

fn fanout_reference(count: usize) -> u64 {
    (0..count)
        .map(fanout_payload)
        .fold(0_u64, u64::wrapping_add)
}

fn channel_reference(count: usize) -> u64 {
    (0..count)
        .map(channel_transform)
        .fold(0_u64, u64::wrapping_add)
}

fn rayon_parallel_iterator(pool: &ThreadPool, count: usize) -> u64 {
    pool.install(|| {
        (0..count)
            .into_par_iter()
            .map(rayon_transform)
            .reduce(|| 0_u64, u64::wrapping_add)
    })
}

fn moirai_indexed_reduce(runtime: &Moirai, count: usize) -> u64 {
    runtime
        .map_reduce_indexed(count, 0_u64, rayon_transform, u64::wrapping_add)
        .expect("Moirai map/reduce should complete")
}

fn tokio_fanout(tokio: &tokio::runtime::Runtime, count: usize) -> u64 {
    tokio.block_on(async move {
        let handles = (0..count)
            .map(|index| {
                tokio::spawn(async move {
                    tokio::time::sleep(fanout_delay(index)).await;
                    fanout_payload(index)
                })
            })
            .collect::<Vec<_>>();

        let mut checksum = 0_u64;
        for handle in handles {
            checksum = checksum.wrapping_add(handle.await.expect("Tokio task should complete"));
        }
        checksum
    })
}

fn moirai_fanout(runtime: &Moirai, count: usize) -> u64 {
    let handles = (0..count)
        .map(|index| {
            runtime.spawn_async(async move {
                moirai::sleep(fanout_delay(index)).await;
                fanout_payload(index)
            })
        })
        .collect::<Vec<_>>();

    handles.into_iter().fold(0_u64, |acc, handle| {
        let value = handle
            .join()
            .expect("Moirai task should be joinable")
            .expect("Moirai task should complete");
        acc.wrapping_add(value)
    })
}

fn tokio_single_producer_channel(tokio: &tokio::runtime::Runtime, count: usize) -> u64 {
    tokio.block_on(async move {
        let (tx, mut rx) = mpsc::channel::<Work>(CHANNEL_CAPACITY);
        let consumer = tokio::spawn(async move {
            let mut checksum = 0_u64;
            while let Some(work) = rx.recv().await {
                match work {
                    Work::Item(index) => checksum = checksum.wrapping_add(channel_transform(index)),
                    Work::Stop => break,
                }
            }
            checksum
        });

        let producer = tokio::spawn(async move {
            for index in 0..count {
                tx.send(Work::Item(index))
                    .await
                    .expect("Tokio channel should accept item");
            }
            tx.send(Work::Stop)
                .await
                .expect("Tokio channel should accept stop marker");
        });

        producer.await.expect("Tokio producer should complete");
        consumer.await.expect("Tokio consumer should complete")
    })
}

fn moirai_single_producer_mpmc(runtime: &Moirai, count: usize) -> u64 {
    let (tx, rx) = runtime.bounded_channel::<Work>(CHANNEL_CAPACITY);
    let consumer = runtime.spawn_fn(move || {
        let mut checksum = 0_u64;
        while let Work::Item(index) = rx.recv().expect("Moirai channel should receive work") {
            checksum = checksum.wrapping_add(channel_transform(index));
        }
        checksum
    });

    let producer = runtime.spawn_fn(move || {
        for index in 0..count {
            tx.send(Work::Item(index))
                .expect("Moirai channel should accept item");
        }
        tx.send(Work::Stop)
            .expect("Moirai channel should accept stop marker");
    });

    producer
        .join()
        .expect("Moirai producer should be joinable")
        .expect("Moirai producer should complete");
    consumer
        .join()
        .expect("Moirai consumer should be joinable")
        .expect("Moirai consumer should complete")
}

fn tokio_multi_producer_channel(tokio: &tokio::runtime::Runtime, count: usize) -> u64 {
    tokio.block_on(async move {
        let (tx, mut rx) = mpsc::channel::<Work>(CHANNEL_CAPACITY);
        let consumer = tokio::spawn(async move {
            let mut checksum = 0_u64;
            while let Some(work) = rx.recv().await {
                match work {
                    Work::Item(index) => checksum = checksum.wrapping_add(channel_transform(index)),
                    Work::Stop => break,
                }
            }
            checksum
        });

        let producers = (0..PRODUCERS)
            .map(|producer| {
                let tx = tx.clone();
                tokio::spawn(async move {
                    for index in producer_range(producer, count) {
                        tx.send(Work::Item(index))
                            .await
                            .expect("Tokio channel should accept item");
                    }
                })
            })
            .collect::<Vec<_>>();

        for producer in producers {
            producer.await.expect("Tokio producer should complete");
        }
        tx.send(Work::Stop)
            .await
            .expect("Tokio channel should accept stop marker");
        drop(tx);

        consumer.await.expect("Tokio consumer should complete")
    })
}

fn moirai_multi_producer_mpmc(runtime: &Moirai, count: usize) -> u64 {
    let (tx, rx) = runtime.bounded_channel::<Work>(CHANNEL_CAPACITY);
    let consumer = runtime.spawn_fn(move || {
        let mut checksum = 0_u64;
        while let Work::Item(index) = rx.recv().expect("Moirai channel should receive work") {
            checksum = checksum.wrapping_add(channel_transform(index));
        }
        checksum
    });

    let producers = (0..PRODUCERS)
        .map(|producer| {
            let tx = tx.clone();
            runtime.spawn_fn(move || {
                for index in producer_range(producer, count) {
                    tx.send(Work::Item(index))
                        .expect("Moirai channel should accept item");
                }
            })
        })
        .collect::<Vec<_>>();

    for producer in producers {
        producer
            .join()
            .expect("Moirai producer should be joinable")
            .expect("Moirai producer should complete");
    }
    tx.send(Work::Stop)
        .expect("Moirai channel should accept stop marker");
    drop(tx);

    consumer
        .join()
        .expect("Moirai consumer should be joinable")
        .expect("Moirai consumer should complete")
}

fn bench_rayon_patterns(c: &mut Criterion) {
    let expected = rayon_reference(RAYON_ITEMS);
    let rayon = ThreadPoolBuilder::new()
        .num_threads(WORKER_THREADS)
        .build()
        .expect("Rayon pool must start");
    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");

    let mut group = c.benchmark_group("example_rayon_patterns");
    group.sample_size(BENCHMARK_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS));

    group.bench_function("rayon_parallel_iterator", |bench| {
        bench.iter(|| verify_checksum(rayon_parallel_iterator(&rayon, RAYON_ITEMS), expected));
    });

    group.bench_function("moirai_indexed_reduce", |bench| {
        bench.iter(|| verify_checksum(moirai_indexed_reduce(&moirai, RAYON_ITEMS), expected));
    });

    group.finish();
    moirai.shutdown();
}

fn bench_tokio_fanout(c: &mut Criterion) {
    let expected = fanout_reference(TOKIO_FANOUT_TASKS);
    let moirai = Moirai::builder()
        .worker_threads(WORKER_THREADS)
        .build()
        .expect("Moirai runtime must start");
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");

    let mut group = c.benchmark_group("example_tokio_fanout");
    group.sample_size(BENCHMARK_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS));

    group.bench_function("tokio_spawn_sleep", |bench| {
        bench.iter(|| verify_checksum(tokio_fanout(&tokio, TOKIO_FANOUT_TASKS), expected));
    });

    group.bench_function("moirai_spawn_async_sleep", |bench| {
        bench.iter(|| verify_checksum(moirai_fanout(&moirai, TOKIO_FANOUT_TASKS), expected));
    });

    group.finish();
    moirai.shutdown();
}

fn bench_channel_patterns(c: &mut Criterion) {
    let single_expected = channel_reference(CHANNEL_ITEMS);
    let multi_expected = channel_reference(MULTI_PRODUCER_ITEMS);
    let moirai = Moirai::builder()
        .worker_threads(PRODUCERS + 1)
        .build()
        .expect("Moirai runtime must start");
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(PRODUCERS + 1)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");

    let mut group = c.benchmark_group("example_channel_patterns");
    group.sample_size(BENCHMARK_SAMPLE_SIZE);
    group.measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS));
    group.warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS));

    group.bench_function("tokio_single_producer_mpsc", |bench| {
        bench.iter(|| {
            verify_checksum(
                tokio_single_producer_channel(&tokio, CHANNEL_ITEMS),
                single_expected,
            )
        });
    });

    group.bench_function("moirai_single_producer_mpmc", |bench| {
        bench.iter(|| {
            verify_checksum(
                moirai_single_producer_mpmc(&moirai, CHANNEL_ITEMS),
                single_expected,
            )
        });
    });

    group.bench_function("tokio_multi_producer_mpsc", |bench| {
        bench.iter(|| {
            verify_checksum(
                tokio_multi_producer_channel(&tokio, MULTI_PRODUCER_ITEMS),
                multi_expected,
            )
        });
    });

    group.bench_function("moirai_multi_producer_mpmc", |bench| {
        bench.iter(|| {
            verify_checksum(
                moirai_multi_producer_mpmc(&moirai, MULTI_PRODUCER_ITEMS),
                multi_expected,
            )
        });
    });

    group.finish();
    moirai.shutdown();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(BENCHMARK_SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(BENCHMARK_MEASUREMENT_SECONDS))
        .warm_up_time(Duration::from_millis(BENCHMARK_WARM_UP_MILLIS))
        .without_plots();
    targets = bench_rayon_patterns, bench_tokio_fanout, bench_channel_patterns
}

criterion_main!(benches);
