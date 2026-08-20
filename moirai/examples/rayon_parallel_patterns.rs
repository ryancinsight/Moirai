//! Compare common Rayon data-parallel patterns with Moirai's indexed reduction.
//!
//! Rayon is commonly used by changing iterator entry points to `par_iter` or
//! `into_par_iter`, and by using `rayon::join` for recursive divide-and-conquer
//! work. This example runs both patterns against the same deterministic CPU
//! workload and checks them against Moirai's indexed map/reduce result.

use moirai::Moirai;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::time::{Duration, Instant};

const ITEMS: usize = 65_536;
const ROUNDS: usize = 16;
const JOIN_LEAF: usize = 2_048;

#[derive(Clone, Copy)]
struct TimedChecksum {
    checksum: u64,
    elapsed: Duration,
}

#[inline]
fn cpu_transform(index: usize) -> u64 {
    let mut value = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);

    for round in 0..ROUNDS {
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        value = value.wrapping_add(round as u64);
    }

    value
}

fn sequential_reference(count: usize) -> u64 {
    (0..count).map(cpu_transform).fold(0_u64, u64::wrapping_add)
}

fn timed<F>(operation: F) -> TimedChecksum
where
    F: FnOnce() -> u64,
{
    let start = Instant::now();
    let checksum = operation();
    TimedChecksum {
        checksum,
        elapsed: start.elapsed(),
    }
}

fn rayon_parallel_iterator(pool: &ThreadPool, count: usize) -> TimedChecksum {
    timed(|| {
        pool.install(|| {
            (0..count)
                .into_par_iter()
                .map(cpu_transform)
                .reduce(|| 0_u64, u64::wrapping_add)
        })
    })
}

fn rayon_join_range(start: usize, end: usize) -> u64 {
    if end - start <= JOIN_LEAF {
        return (start..end)
            .map(cpu_transform)
            .fold(0_u64, u64::wrapping_add);
    }

    let midpoint = start + ((end - start) / 2);
    let (left, right) = rayon::join(
        || rayon_join_range(start, midpoint),
        || rayon_join_range(midpoint, end),
    );
    left.wrapping_add(right)
}

fn rayon_join(pool: &ThreadPool, count: usize) -> TimedChecksum {
    timed(|| pool.install(|| rayon_join_range(0, count)))
}

fn moirai_indexed_reduce(runtime: &Moirai, count: usize) -> TimedChecksum {
    timed(|| {
        runtime
            .map_reduce_indexed(count, 0_u64, cpu_transform, u64::wrapping_add)
            .expect("Moirai indexed map/reduce should complete")
    })
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let worker_threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);
    let rayon = ThreadPoolBuilder::new()
        .num_threads(worker_threads)
        .build()?;
    let runtime = Moirai::builder().worker_threads(worker_threads).build()?;
    let reference = sequential_reference(ITEMS);

    println!("=== Rayon Patterns vs Moirai Indexed Reduction ===");
    println!("items={ITEMS} rounds_per_item={ROUNDS} worker_threads={worker_threads}");
    println!("sequential_reference checksum={reference:#018x}");

    let rayon_iter = rayon_parallel_iterator(&rayon, ITEMS);
    let rayon_joined = rayon_join(&rayon, ITEMS);
    let moirai_reduce = moirai_indexed_reduce(&runtime, ITEMS);

    print_result("rayon into_par_iter", rayon_iter, reference);
    print_result("rayon join", rayon_joined, reference);
    print_result("moirai map_reduce", moirai_reduce, reference);

    print_ratio(
        "moirai vs rayon iter",
        rayon_iter.elapsed,
        moirai_reduce.elapsed,
    );
    print_ratio(
        "moirai vs rayon join",
        rayon_joined.elapsed,
        moirai_reduce.elapsed,
    );

    runtime.shutdown();
    Ok(())
}
