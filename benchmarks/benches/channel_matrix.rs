//! Focused bounded-channel benchmark matrix.
//!
//! This isolates channel transport cost across producer counts and capacities.
//! The broader example benchmark includes runtime scheduling and payload work;
//! this matrix keeps the payload to integer transfer plus checksum validation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use moirai_core::channel::mpmc;
use std::thread;
use std::time::Duration;
use tokio::sync::mpsc;

const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_MILLIS: u64 = 250;
const WARM_UP_MILLIS: u64 = 100;
const ITEM_COUNT: usize = 8_192;
const PRODUCER_COUNTS: [usize; 3] = [1, 4, 8];
const CAPACITIES: [usize; 3] = [1, 512, 4_096];

fn expected_sum(count: usize) -> u64 {
    let count = count as u64;
    count.saturating_sub(1).wrapping_mul(count) / 2
}

fn producer_range(producer: usize, producers: usize, count: usize) -> std::ops::Range<usize> {
    let base = count / producers;
    let remainder = count % producers;
    let start = producer * base + producer.min(remainder);
    let len = base + usize::from(producer < remainder);
    start..(start + len)
}

fn verify_sum(actual: u64, expected: u64) -> u64 {
    assert_eq!(actual, expected);
    black_box(actual)
}

fn tokio_mpsc_sum(
    runtime: &tokio::runtime::Runtime,
    producers: usize,
    capacity: usize,
    count: usize,
) -> u64 {
    runtime.block_on(async move {
        let (tx, mut rx) = mpsc::channel::<usize>(capacity);

        let consumer = tokio::spawn(async move {
            let mut sum = 0_u64;
            while let Some(value) = rx.recv().await {
                sum = sum.wrapping_add(value as u64);
            }
            sum
        });

        let producer_handles = (0..producers)
            .map(|producer| {
                let tx = tx.clone();
                tokio::spawn(async move {
                    for item in producer_range(producer, producers, count) {
                        tx.send(item)
                            .await
                            .expect("Tokio bounded channel should accept item");
                    }
                })
            })
            .collect::<Vec<_>>();

        drop(tx);

        for producer in producer_handles {
            producer.await.expect("Tokio producer should complete");
        }

        consumer.await.expect("Tokio consumer should complete")
    })
}

fn moirai_mpmc_sum(producers: usize, capacity: usize, count: usize) -> u64 {
    let (tx, rx) = mpmc::<usize>(capacity);

    let consumer = thread::spawn(move || {
        let mut sum = 0_u64;
        for _ in 0..count {
            let item = rx
                .recv()
                .expect("Moirai bounded channel should receive item");
            sum = sum.wrapping_add(item as u64);
        }
        sum
    });

    let producer_handles = (0..producers)
        .map(|producer| {
            let tx = tx.clone();
            thread::spawn(move || {
                for item in producer_range(producer, producers, count) {
                    tx.send(item)
                        .expect("Moirai bounded channel should accept item");
                }
            })
        })
        .collect::<Vec<_>>();

    drop(tx);

    for producer in producer_handles {
        producer.join().expect("Moirai producer should complete");
    }

    consumer.join().expect("Moirai consumer should complete")
}

fn bench_channel_matrix(c: &mut Criterion) {
    let expected = expected_sum(ITEM_COUNT);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(PRODUCER_COUNTS[PRODUCER_COUNTS.len() - 1] + 1)
        .enable_all()
        .build()
        .expect("Tokio runtime must start");

    let mut group = c.benchmark_group("bounded_channel_matrix");
    group.sample_size(SAMPLE_SIZE);
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.throughput(Throughput::Elements(ITEM_COUNT as u64));

    for producers in PRODUCER_COUNTS {
        for capacity in CAPACITIES {
            let label = format!("p{producers}_c{capacity}");

            group.bench_with_input(
                BenchmarkId::new("tokio_mpsc", &label),
                &(producers, capacity),
                |bench, &(producers, capacity)| {
                    bench.iter(|| {
                        verify_sum(
                            tokio_mpsc_sum(&runtime, producers, capacity, ITEM_COUNT),
                            expected,
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("moirai_mpmc", &label),
                &(producers, capacity),
                |bench, &(producers, capacity)| {
                    bench.iter(|| {
                        verify_sum(moirai_mpmc_sum(producers, capacity, ITEM_COUNT), expected)
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = bench_channel_matrix
}

criterion_main!(benches);
