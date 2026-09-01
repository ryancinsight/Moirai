//! Owned execution-context iterator comparison against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::{MoiraiIterator, ParallelContext};
use rayon::prelude::*;
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const WARM_UP_MILLIS: u64 = 100;
const MEASUREMENT_MILLIS: u64 = 300;
const WORK_ITEMS: usize = 512;
const ASYNC_WORK_ITEMS: usize = 1_024;

fn source_data() -> Vec<u64> {
    (0..WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(13).wrapping_add(5))
        .collect()
}

fn moirai_parallel_context_map(context: &ParallelContext, data: Vec<u64>) -> u64 {
    context
        .execute_iter(data, |value| value.wrapping_mul(3).wrapping_add(1))
        .expect("parallel context map must complete")
        .into_iter()
        .sum()
}

fn rayon_owned_map(data: Vec<u64>) -> u64 {
    data.into_par_iter()
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .sum()
}

fn moirai_parallel_context_async_map(data: Vec<u64>) -> u64 {
    futures::executor::block_on(async {
        MoiraiIterator::parallel(data)
            .map_async(|value| async move { value.wrapping_mul(5).wrapping_add(3) })
            .await
            .collect()
            .await
            .into_iter()
            .sum()
    })
}

fn execution_context_comparison(c: &mut Criterion) {
    let data = source_data();
    let context = ParallelContext::with_chunk_size(WORK_ITEMS);

    assert_eq!(
        moirai_parallel_context_map(&context, data.clone()),
        rayon_owned_map(data.clone())
    );

    let mut group = c.benchmark_group("execution_context_owned_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| {
            black_box(moirai_parallel_context_map(
                &context,
                black_box(input.clone()),
            ))
        })
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_owned_map(black_box(input.clone()))))
    });
    group.finish();

    let async_data = (0..ASYNC_WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(19).wrapping_add(7))
        .collect::<Vec<_>>();
    let expected = async_data
        .iter()
        .copied()
        .map(|value| value.wrapping_mul(5).wrapping_add(3))
        .sum::<u64>();
    assert_eq!(
        moirai_parallel_context_async_map(async_data.clone()),
        expected
    );

    let mut group = c.benchmark_group("execution_context_parallel_async_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(
        BenchmarkId::new("moirai", ASYNC_WORK_ITEMS),
        &async_data,
        |b, input| {
            b.iter(|| black_box(moirai_parallel_context_async_map(black_box(input.clone()))))
        },
    );
    group.finish();
}

criterion_group!(benches, execution_context_comparison);
criterion_main!(benches);
