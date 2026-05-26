//! Compare Tokio bounded MPSC message passing with Moirai bounded channels.
//!
//! Tokio's channel tutorial presents message passing as the normal way to let
//! one task own a resource while other tasks submit commands. This example uses
//! that shape for a bounded producer/consumer pipeline and compares it with
//! Moirai's bounded channel plus blocking runtime tasks.

use moirai::Moirai;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const ITEMS: usize = 100_000;
const CHANNEL_CAPACITY: usize = 256;
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
    let mut value = (index as u64).wrapping_add(0x517c_c1b7_2722_0a95);

    for round in 0..ROUNDS {
        value ^= value.rotate_right(17);
        value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        value ^= value >> 29;
        value = value.wrapping_add(round as u64);
    }

    value
}

fn reference_checksum(count: usize) -> u64 {
    (0..count).map(transform).fold(0_u64, u64::wrapping_add)
}

async fn tokio_mpsc(count: usize) -> TimedChecksum {
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
    let checksum = consumer.await.expect("Tokio consumer should complete");

    TimedChecksum {
        checksum,
        elapsed: start.elapsed(),
    }
}

fn moirai_bounded_channel(runtime: &Moirai, count: usize) -> TimedChecksum {
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
    let checksum = consumer
        .join()
        .expect("Moirai consumer should be joinable")
        .expect("Moirai consumer should complete");

    TimedChecksum {
        checksum,
        elapsed: start.elapsed(),
    }
}

fn moirai_spsc_channel(runtime: &Moirai, count: usize) -> TimedChecksum {
    let (tx, rx) = moirai_core::channel::spsc::<Work>(CHANNEL_CAPACITY);

    let start = Instant::now();
    let consumer = runtime.spawn_fn(move || {
        let mut checksum = 0_u64;
        loop {
            match rx.recv().expect("Moirai SPSC channel should receive work") {
                Work::Item(index) => checksum = checksum.wrapping_add(transform(index)),
                Work::Stop => break,
            }
        }
        checksum
    });

    let producer = runtime.spawn_fn(move || {
        for index in 0..count {
            tx.send(Work::Item(index))
                .expect("Moirai SPSC channel should accept item");
        }
        tx.send(Work::Stop)
            .expect("Moirai SPSC channel should accept stop marker");
    });

    producer
        .join()
        .expect("Moirai SPSC producer should be joinable")
        .expect("Moirai SPSC producer should complete");
    let checksum = consumer
        .join()
        .expect("Moirai SPSC consumer should be joinable")
        .expect("Moirai SPSC consumer should complete");

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
    let runtime = Moirai::new()?;
    let reference = reference_checksum(ITEMS);

    println!("=== Tokio MPSC Pipeline vs Moirai Bounded Channel ===");
    println!("items={ITEMS} capacity={CHANNEL_CAPACITY} payload_rounds={ROUNDS}");
    println!("sequential_reference checksum={reference:#018x}");

    let tokio = tokio_mpsc(ITEMS).await;
    let moirai_mpmc = moirai_bounded_channel(&runtime, ITEMS);
    let moirai_spsc = moirai_spsc_channel(&runtime, ITEMS);

    print_result("tokio mpsc", tokio, reference);
    print_result("moirai bounded mpmc", moirai_mpmc, reference);
    print_result("moirai bounded spsc", moirai_spsc, reference);
    print_ratio("moirai mpmc vs tokio", tokio.elapsed, moirai_mpmc.elapsed);
    print_ratio("moirai spsc vs tokio", tokio.elapsed, moirai_spsc.elapsed);

    runtime.shutdown();
    Ok(())
}
