//! Resize-gate cost of batch stealing, measured from both sides.
//!
//! `ChaseLevDeque::steal_batch` moves up to sixteen items behind the deque's
//! resize gate, a `SeqCst` counter every thief shares. Entering the gate once
//! per item and entering it once per batch differ in *where* the cost lands:
//! per-item entry pays contended read-modify-writes on one line for every
//! element, while per-batch entry holds the gate longer and so makes a
//! concurrent `resize` — which spins until the gate is empty — wait behind a
//! whole batch.
//!
//! Both sides are therefore measured, because a batch-throughput win that
//! starves the owner's growth path is not a win:
//!
//! - `thief_drain` times K thieves draining a pre-filled deque. No resize can
//!   run, so this isolates the per-item gate traffic.
//! - `owner_growth_under_thieves` times only the owner's push loop while K
//!   thieves batch-steal, starting from the minimum capacity so the pushes
//!   force repeated resizes against live thieves. This is the side the hoist
//!   can regress.
//!
//! Both rows assert exactly-once transfer over their payloads: every pushed
//! value is either stolen or still resident, counted once.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use moirai_scheduler::{ChaseLevDeque, DeferredReclaim, DequeCapacity, StealResult};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::{Duration, Instant},
};

/// Thief counts. Contention on the shared gate counter is the whole subject, so
/// the ladder starts at two thieves; a single-thief row measures thread
/// placement on this class of hybrid core more than it measures the gate.
const THIEF_COUNTS: &[usize] = &[2, 4, 8];

/// Items drained per `thief_drain` iteration. Large enough that thread start-up
/// does not dominate, small enough to keep the row inside the suite budget.
const DRAIN_ITEMS: usize = 4_096;

/// Pushes per `owner_growth_under_thieves` iteration. From the 16-slot minimum
/// this doubles the buffer nine times, so the row measures nine resizes racing
/// live thieves rather than a single lucky one.
const GROWTH_PUSHES: usize = 8_192;

const MIN_CAPACITY: usize = 16;

fn capacity(requested: usize) -> DequeCapacity<usize> {
    DequeCapacity::try_from(requested).expect("benchmark capacity must be representable")
}

/// Closed-form sum of `1..=count`, the value every payload total must match.
fn expected_sum(count: usize) -> usize {
    count.wrapping_mul(count.wrapping_add(1)) / 2
}

/// Drains `DRAIN_ITEMS` pre-pushed values with `thieves` batch stealers and
/// returns the wall time from barrier release to the last thief finishing.
///
/// The deque is filled before the barrier, so its capacity never grows during
/// the timed region: what is measured is gate traffic between thieves.
fn thief_drain(thieves: usize) -> Duration {
    let mut owner: ChaseLevDeque<usize, DeferredReclaim> =
        ChaseLevDeque::new(capacity(DRAIN_ITEMS.next_power_of_two()));
    for value in 1..=DRAIN_ITEMS {
        owner.push(black_box(value));
    }

    let taken = Arc::new(AtomicUsize::new(0));
    let sum = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(thieves + 1));

    let workers: Vec<_> = (0..thieves)
        .map(|_| {
            let stealer = owner.stealer();
            let taken = Arc::clone(&taken);
            let sum = Arc::clone(&sum);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let mut local_count = 0usize;
                let mut local_sum = 0usize;
                while taken.load(Ordering::Relaxed) < DRAIN_ITEMS {
                    match stealer.steal_batch() {
                        StealResult::Success(batch) => {
                            let mut batch_count = 0usize;
                            for value in batch {
                                local_sum = local_sum.wrapping_add(value);
                                batch_count += 1;
                            }
                            local_count += batch_count;
                            taken.fetch_add(batch_count, Ordering::Relaxed);
                        }
                        StealResult::Empty | StealResult::Retry => std::hint::spin_loop(),
                    }
                }
                sum.fetch_add(local_sum, Ordering::Relaxed);
                black_box(local_count);
            })
        })
        .collect();

    barrier.wait();
    let started = Instant::now();
    for worker in workers {
        worker.join().expect("thief thread must not panic");
    }
    let elapsed = started.elapsed();

    assert_eq!(
        taken.load(Ordering::Relaxed),
        DRAIN_ITEMS,
        "every pushed value must be stolen exactly once"
    );
    assert_eq!(
        sum.load(Ordering::Relaxed),
        expected_sum(DRAIN_ITEMS),
        "stolen payloads must sum to the closed form"
    );
    elapsed
}

/// Times only the owner's `GROWTH_PUSHES` pushes, from the minimum capacity,
/// while `thieves` batch stealers run against the same deque.
///
/// Every push that fills the buffer calls `resize`, which spins until the steal
/// gate is empty, so this row is the owner's exposure to how long a thief holds
/// that gate.
fn owner_growth_under_thieves(thieves: usize) -> Duration {
    let mut owner: ChaseLevDeque<usize, DeferredReclaim> =
        ChaseLevDeque::new(capacity(MIN_CAPACITY));
    let stop = Arc::new(AtomicBool::new(false));
    let taken = Arc::new(AtomicUsize::new(0));
    let sum = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(thieves + 1));

    let workers: Vec<_> = (0..thieves)
        .map(|_| {
            let stealer = owner.stealer();
            let stop = Arc::clone(&stop);
            let taken = Arc::clone(&taken);
            let sum = Arc::clone(&sum);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let mut local_count = 0usize;
                let mut local_sum = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    match stealer.steal_batch() {
                        StealResult::Success(batch) => {
                            for value in batch {
                                local_sum = local_sum.wrapping_add(value);
                                local_count += 1;
                            }
                        }
                        StealResult::Empty | StealResult::Retry => std::hint::spin_loop(),
                    }
                }
                taken.fetch_add(local_count, Ordering::Relaxed);
                sum.fetch_add(local_sum, Ordering::Relaxed);
            })
        })
        .collect();

    barrier.wait();
    let started = Instant::now();
    for value in 1..=GROWTH_PUSHES {
        owner.push(black_box(value));
    }
    let elapsed = started.elapsed();

    stop.store(true, Ordering::Relaxed);
    for worker in workers {
        worker.join().expect("thief thread must not panic");
    }

    // The owner drains whatever the thieves left; stolen plus resident must
    // account for every pushed value exactly once.
    let mut resident_count = 0usize;
    let mut resident_sum = 0usize;
    while let Some(value) = owner.pop() {
        resident_sum = resident_sum.wrapping_add(value);
        resident_count += 1;
    }

    assert_eq!(
        taken.load(Ordering::Relaxed) + resident_count,
        GROWTH_PUSHES,
        "stolen and resident items must account for every push exactly once"
    );
    assert_eq!(
        sum.load(Ordering::Relaxed).wrapping_add(resident_sum),
        expected_sum(GROWTH_PUSHES),
        "stolen and resident payloads must sum to the closed form"
    );
    elapsed
}

pub(super) fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("steal_batch_gate");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(2));
    group.warm_up_time(Duration::from_millis(300));

    for &thieves in THIEF_COUNTS {
        group.throughput(Throughput::Elements(DRAIN_ITEMS as u64));
        group.bench_with_input(
            BenchmarkId::new("thief_drain", thieves),
            &thieves,
            |b, &thieves| {
                b.iter_custom(|iterations| (0..iterations).map(|_| thief_drain(thieves)).sum());
            },
        );

        group.throughput(Throughput::Elements(GROWTH_PUSHES as u64));
        group.bench_with_input(
            BenchmarkId::new("owner_growth_under_thieves", thieves),
            &thieves,
            |b, &thieves| {
                b.iter_custom(|iterations| {
                    (0..iterations)
                        .map(|_| owner_growth_under_thieves(thieves))
                        .sum()
                });
            },
        );
    }

    group.finish();
}
