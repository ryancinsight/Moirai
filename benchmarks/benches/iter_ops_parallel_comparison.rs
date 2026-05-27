//! Scoped `iter_ops::ParallelIter` comparison against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::iter_ops::ParallelIter;
use rayon::prelude::*;
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const WARM_UP_MILLIS: u64 = 100;
const MEASUREMENT_MILLIS: u64 = 300;
const WORK_ITEMS: usize = 8_192;

fn source_data() -> Vec<u64> {
    (0..WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(31).wrapping_add(7))
        .collect()
}

fn moirai_parallel_map(data: Vec<u64>) -> u64 {
    ParallelIter::new(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .into_iter()
        .sum()
}

fn rayon_parallel_map(data: Vec<u64>) -> u64 {
    data.into_par_iter()
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .sum()
}

fn moirai_parallel_reduce(data: Vec<u64>) -> u64 {
    ParallelIter::new(data).reduce(0_u64, |accumulator, value| accumulator.wrapping_add(*value))
}

fn rayon_parallel_reduce(data: Vec<u64>) -> u64 {
    data.into_par_iter()
        .reduce(|| 0_u64, |left, right| left.wrapping_add(right))
}

fn iter_ops_parallel_comparison(c: &mut Criterion) {
    let data = source_data();

    assert_eq!(
        moirai_parallel_map(data.clone()),
        rayon_parallel_map(data.clone())
    );
    assert_eq!(
        moirai_parallel_reduce(data.clone()),
        rayon_parallel_reduce(data.clone())
    );

    let mut map_group = c.benchmark_group("iter_ops_parallel_map");
    map_group.sample_size(SAMPLE_SIZE);
    map_group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    map_group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    map_group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_parallel_map(black_box(input.clone()))))
    });
    map_group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_parallel_map(black_box(input.clone()))))
    });
    map_group.finish();

    let mut reduce_group = c.benchmark_group("iter_ops_parallel_reduce");
    reduce_group.sample_size(SAMPLE_SIZE);
    reduce_group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    reduce_group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    reduce_group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_parallel_reduce(black_box(input.clone()))))
    });
    reduce_group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_parallel_reduce(black_box(input.clone()))))
    });
    reduce_group.finish();
}

criterion_group!(benches, iter_ops_parallel_comparison);
criterion_main!(benches);
