//! Initial-capacity comparison for the scheduler's production local deque.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use moirai_scheduler::{ChaseLevDeque, DequeCapacity};
use std::time::Duration;

const CAPACITIES: &[usize] = &[16, 32, 64, 128, 256];
const WARM_ITEMS: usize = 15;
const BURST_ITEMS: usize = 257;

#[repr(transparent)]
struct QueuePayload([usize; 16]);

const _: () = assert!(core::mem::size_of::<QueuePayload>() == 16 * core::mem::size_of::<usize>());

fn deque_capacity(requested: usize) -> DequeCapacity<QueuePayload> {
    DequeCapacity::try_from(requested).expect("benchmark capacity must be representable")
}

fn expected_sum(count: usize) -> usize {
    count.wrapping_mul(count.wrapping_add(1)) / 2
}

fn verify_sum(sum: usize, count: usize) -> usize {
    assert_eq!(sum, expected_sum(count));
    black_box(sum)
}

fn batch_sum(deque: &mut ChaseLevDeque<QueuePayload>, count: usize) -> usize {
    for value in 0..count {
        deque.push(black_box(QueuePayload([value.wrapping_add(1); 16])));
    }

    let mut sum = 0usize;
    while let Some(payload) = deque.pop() {
        sum = sum.wrapping_add(payload.0[0]);
    }
    sum
}

fn cold_burst_sum(capacity: usize, count: usize) -> usize {
    let mut deque = ChaseLevDeque::new(deque_capacity(capacity));
    batch_sum(&mut deque, count)
}

pub(super) fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_queue_initial_capacity");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(250));

    for &capacity in CAPACITIES {
        let mut deque = ChaseLevDeque::new(deque_capacity(capacity));
        group.throughput(Throughput::Elements(WARM_ITEMS as u64));
        group.bench_with_input(
            BenchmarkId::new("warm_no_growth", capacity),
            &capacity,
            |b, _| b.iter(|| verify_sum(batch_sum(&mut deque, WARM_ITEMS), WARM_ITEMS)),
        );

        group.throughput(Throughput::Elements(BURST_ITEMS as u64));
        group.bench_with_input(
            BenchmarkId::new("cold_growth_burst", capacity),
            &capacity,
            |b, &capacity| {
                b.iter(|| verify_sum(cold_burst_sum(capacity, BURST_ITEMS), BURST_ITEMS));
            },
        );
    }

    group.finish();
}
