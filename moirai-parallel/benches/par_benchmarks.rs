//! Benchmarks for the synchronous data-parallel primitives, with sequential and
//! rayon baselines for differential comparison.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_parallel::{join_with, Parallel, ParallelSlice, ParallelSliceMut};
use rayon::prelude::*;

fn closed_form_sum(n: u64) -> u64 {
    n.saturating_sub(1).saturating_mul(n) / 2
}

fn sum_range(n: u64) -> u64 {
    (0..n).map(black_box).sum()
}

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
            b.iter(|| {
                data.par_iter_mut()
                    .for_each(|x| *x = black_box(*x) * 1.000_001)
            });
        });
        group.bench_with_input(BenchmarkId::new("moirai", n), &n, |b, &n| {
            let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            b.iter(|| data.par_mut().for_each(|x| *x = black_box(*x) * 1.000_001));
        });
    }
    group.finish();
}

fn bench_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_sum_pair");
    for &n in &[100_000u64, 1_000_000] {
        let expected = closed_form_sum(n).wrapping_add(closed_form_sum(n / 2));
        let sequential = sum_range(n).wrapping_add(sum_range(n / 2));
        let rayon = {
            let (left, right) = rayon::join(|| sum_range(n), || sum_range(n / 2));
            left.wrapping_add(right)
        };
        let moirai = {
            let (left, right) =
                join_with::<Parallel, _, _, _, _>(|| sum_range(n), || sum_range(n / 2));
            left.wrapping_add(right)
        };
        assert_eq!(sequential, expected);
        assert_eq!(rayon, expected);
        assert_eq!(moirai, expected);

        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, &n| {
            b.iter(|| {
                black_box(sum_range(n).wrapping_add(sum_range(n / 2)));
            });
        });
        group.bench_with_input(BenchmarkId::new("rayon", n), &n, |b, &n| {
            b.iter(|| {
                let (left, right) = rayon::join(|| sum_range(n), || sum_range(n / 2));
                black_box(left.wrapping_add(right));
            });
        });
        group.bench_with_input(BenchmarkId::new("moirai", n), &n, |b, &n| {
            b.iter(|| {
                let (left, right) =
                    join_with::<Parallel, _, _, _, _>(|| sum_range(n), || sum_range(n / 2));
                black_box(left.wrapping_add(right));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_map_reduce, bench_for_each_mut, bench_join);
criterion_main!(benches);
