//! Compare Tokio task fanout with Moirai async task fanout.
//!
//! Tokio examples commonly use `tokio::spawn` to fan out independent async work
//! and then await each join handle. This example mirrors that shape with
//! `Moirai::spawn_async`, using deterministic delays and checksums so both
//! runtimes are compared against the same observable result.

use moirai::Moirai;
use std::time::{Duration, Instant};
use tokio::time::sleep as tokio_sleep;

const TASKS: usize = 512;
const ROUNDS: usize = 12;

#[derive(Clone, Copy)]
struct TimedChecksum {
    checksum: u64,
    elapsed: Duration,
}

#[inline]
fn delay_for(index: usize) -> Duration {
    Duration::from_millis(((index % 4) + 1) as u64)
}

#[inline]
fn payload(index: usize) -> u64 {
    let mut value = (index as u64).wrapping_mul(0x0001_0000_0001_b3);

    for round in 0..ROUNDS {
        value ^= value.rotate_left(13);
        value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
        value ^= value >> 33;
        value = value.wrapping_add(round as u64);
    }

    value
}

fn reference_checksum(count: usize) -> u64 {
    (0..count).map(payload).fold(0_u64, u64::wrapping_add)
}

async fn tokio_fanout(count: usize) -> TimedChecksum {
    let start = Instant::now();
    let handles = (0..count)
        .map(|index| {
            tokio::spawn(async move {
                tokio_sleep(delay_for(index)).await;
                payload(index)
            })
        })
        .collect::<Vec<_>>();

    let mut checksum = 0_u64;
    for handle in handles {
        let value = handle.await.expect("Tokio task should complete");
        checksum = checksum.wrapping_add(value);
    }

    TimedChecksum {
        checksum,
        elapsed: start.elapsed(),
    }
}

fn moirai_fanout(runtime: &Moirai, count: usize) -> TimedChecksum {
    let start = Instant::now();
    let handles = (0..count)
        .map(|index| {
            runtime.spawn_async(async move {
                moirai::sleep(delay_for(index)).await;
                payload(index)
            })
        })
        .collect::<Vec<_>>();

    let checksum = handles.into_iter().fold(0_u64, |acc, handle| {
        let value = handle
            .join()
            .expect("Moirai task should be joinable")
            .expect("Moirai task should complete");
        acc.wrapping_add(value)
    });

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
        "{label:<20} checksum={:#018x} elapsed={:?}",
        result.checksum, result.elapsed
    );
}

fn print_ratio(label: &str, baseline: Duration, candidate: Duration) {
    let ratio = candidate.as_secs_f64() / baseline.as_secs_f64();
    println!("{label:<20} candidate/baseline={ratio:.3}x");
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reference = reference_checksum(TASKS);
    let runtime = Moirai::new()?;

    println!("=== Tokio Task Fanout vs Moirai Async Fanout ===");
    println!("tasks={TASKS} payload_rounds={ROUNDS}");
    println!("sequential_reference checksum={reference:#018x}");

    let tokio = tokio_fanout(TASKS).await;
    let moirai = moirai_fanout(&runtime, TASKS);

    print_result("tokio spawn", tokio, reference);
    print_result("moirai spawn_async", moirai, reference);
    print_ratio("moirai vs tokio", tokio.elapsed, moirai.elapsed);

    runtime.shutdown();
    Ok(())
}
