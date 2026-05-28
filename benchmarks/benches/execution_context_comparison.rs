//! Owned execution-context iterator comparison against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::ParallelContext;
use rayon::prelude::*;
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const WARM_UP_MILLIS: u64 = 100;
const MEASUREMENT_MILLIS: u64 = 300;
const WORK_ITEMS: usize = 512;

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
}

criterion_group!(benches, execution_context_comparison);
criterion_main!(benches);
