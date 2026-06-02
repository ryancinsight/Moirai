//! Benchmarks for the synchronous data-parallel primitives, with sequential and
//! rayon baselines for differential comparison.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_parallel::{ParallelSlice, ParallelSliceMut};
use rayon::prelude::*;

fn bench_map_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_reduce_sum_f64");
    for &n in &[10_000usize, 1_000_000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

        group.bench_with_input(BenchmarkId::new("sequential", n), &data, |b, d| {
            b.iter(|| black_box(d.iter().copied().sum::<f64>()));
        });
        group.bench_with_input(BenchmarkId::new("rayon", n), &data, |b, d| {
            b.iter(|| black_box(d.par_iter().copied().sum::<f64>()));
        });
        group.bench_with_input(BenchmarkId::new("moirai", n), &data, |b, d| {
            b.iter(|| black_box(d.par().map_reduce(0.0f64, |&x| x, |a, b| a + b)));
        });
    }
    group.finish();
}

fn bench_for_each_mut(c: &mut Criterion) {
    let mut group = c.benchmark_group("for_each_mut_scale");
    for &n in &[10_000usize, 1_000_000] {
        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, &n| {
            let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            b.iter(|| data.iter_mut().for_each(|x| *x = black_box(*x) * 1.000_001));
        });
        group.bench_with_input(BenchmarkId::new("rayon", n), &n, |b, &n| {
            let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            b.iter(|| data.par_iter_mut().for_each(|x| *x = black_box(*x) * 1.000_001));
        });
        group.bench_with_input(BenchmarkId::new("moirai", n), &n, |b, &n| {
            let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            b.iter(|| data.par_mut().for_each(|x| *x = black_box(*x) * 1.000_001));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_map_reduce, bench_for_each_mut);
criterion_main!(benches);
