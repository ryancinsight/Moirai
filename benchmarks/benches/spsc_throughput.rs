//! SPSC ring throughput: one producer thread, one consumer thread.
//!
//! The measured quantity is the steady-state cost of a send/receive pair when
//! the queue is neither full nor empty — the regime where the cached-index
//! optimisation applies. The capacity is deliberately larger than the burst so
//! the producer rarely blocks, because a queue that is constantly full measures
//! backoff rather than the ring itself.

use std::hint::black_box;
use std::thread;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use moirai_core::channel::spsc;

/// Push `count` values through a channel of `capacity` and return their sum, so
/// the compiler cannot discard the transfer.
fn round_trip(capacity: usize, count: usize) -> u64 {
    let (tx, rx) = spsc::<u64>(capacity);

    let producer = thread::spawn(move || {
        for value in 0..count as u64 {
            // A closed channel would mean the consumer died; let the join report it.
            if tx.send(black_box(value)).is_err() {
                break;
            }
        }
    });

    let mut sum = 0_u64;
    for _ in 0..count {
        match rx.recv() {
            Ok(value) => sum = sum.wrapping_add(value),
            Err(_) => break,
        }
    }

    producer.join().expect("producer thread must not panic");
    sum
}

fn bench_spsc_throughput(c: &mut Criterion) {
    const COUNT: usize = 100_000;

    let mut group = c.benchmark_group("spsc_throughput");
    group.throughput(Throughput::Elements(COUNT as u64));

    // Two regimes: a deep queue the producer never fills, and a shallow one that
    // forces the cache to be refreshed often, which is where the optimisation
    // should show no benefit and must show no harm.
    for capacity in [64_usize, 8192] {
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            &capacity,
            |b, &capacity| {
                b.iter(|| black_box(round_trip(capacity, COUNT)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_spsc_throughput);
criterion_main!(benches);
