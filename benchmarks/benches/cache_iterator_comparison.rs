//! Borrowed cache iterator comparison against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::cache::CacheIterExt;
use rayon::prelude::*;
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const WARM_UP_MILLIS: u64 = 100;
const MEASUREMENT_MILLIS: u64 = 300;
const WORK_ITEMS: usize = 1_024;
const LARGE_WORK_ITEMS: usize = 32_768;

fn source_data() -> Vec<u64> {
    (0..WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(17).wrapping_add(11))
        .collect()
}

fn large_source_data() -> Vec<u64> {
    (0..LARGE_WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(29).wrapping_add(13))
        .collect()
}

fn moirai_zero_copy_map(data: &[u64]) -> u64 {
    data.zero_copy_par_iter()
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .into_iter()
        .sum()
}

fn rayon_borrowed_map(data: &[u64]) -> u64 {
    data.par_iter()
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .sum()
}

fn moirai_zero_copy_reduce(data: &[u64]) -> u64 {
    data.zero_copy_par_iter()
        .reduce(|left, right| left.wrapping_add(*right))
        .unwrap_or(0)
}

fn rayon_borrowed_reduce(data: &[u64]) -> u64 {
    data.par_iter()
        .copied()
        .reduce(|| 0_u64, |left, right| left.wrapping_add(right))
}

fn moirai_zero_copy_large_reduce(data: &[u64]) -> u64 {
    data.zero_copy_par_iter()
        .reduce(|left, right| left.wrapping_add(*right))
        .unwrap_or(0)
}

fn rayon_borrowed_large_reduce(data: &[u64]) -> u64 {
    data.par_iter()
        .copied()
        .reduce(|| 0_u64, |left, right| left.wrapping_add(right))
}

fn cache_iterator_comparison(c: &mut Criterion) {
    let data = source_data();
    let large_data = large_source_data();

    assert_eq!(moirai_zero_copy_map(&data), rayon_borrowed_map(&data));
    assert_eq!(moirai_zero_copy_reduce(&data), rayon_borrowed_reduce(&data));
    assert_eq!(
        moirai_zero_copy_large_reduce(&large_data),
        rayon_borrowed_large_reduce(&large_data)
    );

    let mut map_group = c.benchmark_group("cache_iterator_zero_copy_map");
    map_group.sample_size(SAMPLE_SIZE);
    map_group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    map_group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    map_group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_zero_copy_map(black_box(input))))
    });
    map_group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_borrowed_map(black_box(input))))
    });
    map_group.finish();

    let mut reduce_group = c.benchmark_group("cache_iterator_zero_copy_reduce");
    reduce_group.sample_size(SAMPLE_SIZE);
    reduce_group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    reduce_group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    reduce_group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_zero_copy_reduce(black_box(input))))
    });
    reduce_group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_borrowed_reduce(black_box(input))))
    });
    reduce_group.finish();

    let mut large_reduce_group = c.benchmark_group("cache_iterator_zero_copy_large_reduce");
    large_reduce_group.sample_size(SAMPLE_SIZE);
    large_reduce_group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    large_reduce_group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    large_reduce_group.bench_with_input(
        BenchmarkId::new("moirai", LARGE_WORK_ITEMS),
        &large_data,
        |b, input| b.iter(|| black_box(moirai_zero_copy_large_reduce(black_box(input)))),
    );
    large_reduce_group.bench_with_input(
        BenchmarkId::new("rayon", LARGE_WORK_ITEMS),
        &large_data,
        |b, input| b.iter(|| black_box(rayon_borrowed_large_reduce(black_box(input)))),
    );
    large_reduce_group.finish();
}

criterion_group!(benches, cache_iterator_comparison);
criterion_main!(benches);
