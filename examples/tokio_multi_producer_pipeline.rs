//! Compare cloned-producer Tokio MPSC with Moirai MPMC.
//!
//! Tokio's `mpsc::Sender` is commonly cloned across producer tasks while one
//! receiver owns the resource. This example uses the same multi-producer shape
//! with Moirai's bounded MPMC channel and checks both paths against the same
//! deterministic checksum.

use moirai::Moirai;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const ITEMS: usize = 120_000;
const PRODUCERS: usize = 4;
const CHANNEL_CAPACITY: usize = 512;
const ROUNDS: usize = 8;

#[derive(Clone, Copy)]
enum Work {
    Item(usize),
    Stop,
}

#[derive(Clone, Copy)]
struct TimedChecksum {
    checksum: u64,
    elapsed: Duration,
}

#[inline]
fn transform(index: usize) -> u64 {
    let mut value = (index as u64).wrapping_add(0x6a09_e667_f3bc_c909);

    for round in 0..ROUNDS {
        value ^= value.rotate_left(11);
        value = value.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        value ^= value >> 31;
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

fn reference_checksum(count: usize) -> u64 {
    (0..count).map(transform).fold(0_u64, u64::wrapping_add)
}

async fn tokio_multi_producer(count: usize) -> TimedChecksum {
    let (tx, mut rx) = mpsc::channel::<Work>(CHANNEL_CAPACITY);

    let start = Instant::now();
    let consumer = tokio::spawn(async move {
        let mut checksum = 0_u64;
        while let Some(work) = rx.recv().await {
            match work {
                Work::Item(index) => checksum = checksum.wrapping_add(transform(index)),
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
                        .expect("Tokio channel should accept producer item");
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

    let checksum = consumer.await.expect("Tokio consumer should complete");

    TimedChecksum {
        checksum,
        elapsed: start.elapsed(),
    }
}

fn moirai_multi_producer(runtime: &Moirai, count: usize) -> TimedChecksum {
    let (tx, rx) = runtime.bounded_channel::<Work>(CHANNEL_CAPACITY);

    let start = Instant::now();
    let consumer = runtime.spawn_fn(move || {
        let mut checksum = 0_u64;
        loop {
            match rx.recv().expect("Moirai channel should receive work") {
                Work::Item(index) => checksum = checksum.wrapping_add(transform(index)),
                Work::Stop => break,
            }
        }
        checksum
    });

    let producers = (0..PRODUCERS)
        .map(|producer| {
            let tx = tx.clone();
            runtime.spawn_fn(move || {
                for index in producer_range(producer, count) {
                    tx.send(Work::Item(index))
                        .expect("Moirai channel should accept producer item");
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

    let checksum = consumer
        .join()
        .expect("Moirai consumer should be joinable")
        .expect("Moirai consumer should complete");

    TimedChecksum {
        checksum,
        elapsed: start.elapsed(),
    }
}

fn print_result(label: &str, result: TimedChecksum, reference: u64) {
    assert_eq!(
        result.checksum, reference,
        "{label} produced a checksum that differs from the sequential reference"
    );
    println!(
        "{label:<24} checksum={:#018x} elapsed={:?}",
        result.checksum, result.elapsed
    );
}

fn print_ratio(label: &str, baseline: Duration, candidate: Duration) {
    let ratio = candidate.as_secs_f64() / baseline.as_secs_f64();
    println!("{label:<24} candidate/baseline={ratio:.3}x");
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Moirai::builder().worker_threads(PRODUCERS + 1).build()?;
    let reference = reference_checksum(ITEMS);

    println!("=== Tokio Multi-Producer MPSC vs Moirai MPMC ===");
    println!(
        "items={ITEMS} producers={PRODUCERS} capacity={CHANNEL_CAPACITY} payload_rounds={ROUNDS}"
    );
    println!("sequential_reference checksum={reference:#018x}");

    let tokio = tokio_multi_producer(ITEMS).await;
    let moirai = moirai_multi_producer(&runtime, ITEMS);

    print_result("tokio cloned mpsc", tokio, reference);
    print_result("moirai cloned mpmc", moirai, reference);
    print_ratio("moirai vs tokio", tokio.elapsed, moirai.elapsed);

    runtime.shutdown();
    Ok(())
}
