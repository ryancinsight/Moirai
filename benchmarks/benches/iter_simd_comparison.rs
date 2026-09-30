//! Generic iterator SIMD-surface benchmarks.

#[path = "common/time_model.rs"]
mod time_model;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::simd_iter::SimdSliceIter;

const WORK_ITEMS: usize = 32 * 1024;

fn data() -> (Vec<u64>, Vec<u64>) {
    let left = (0..WORK_ITEMS)
        .map(|index| (index as u64).wrapping_mul(31).wrapping_add(7))
        .collect();
    let right = (0..WORK_ITEMS)
        .map(|index| (index as u64).wrapping_mul(17).wrapping_add(11))
        .collect();
    (left, right)
}

fn generic_add(left: &[u64], right: &[u64]) -> Vec<u64> {
    SimdSliceIter::new(left).add_slice(right)
}

fn scalar_add(left: &[u64], right: &[u64]) -> Vec<u64> {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(left, right)| left + right)
        .collect()
}

fn generic_dot(left: &[u64], right: &[u64]) -> u64 {
    SimdSliceIter::new(left).dot(right)
}

fn scalar_dot(left: &[u64], right: &[u64]) -> u64 {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .fold(0_u64, |acc, (left, right)| acc + left * right)
}

fn iter_simd_comparison(c: &mut Criterion) {
    let (left, right) = data();

    assert_eq!(generic_add(&left, &right), scalar_add(&left, &right));
    assert_eq!(generic_dot(&left, &right), scalar_dot(&left, &right));

    let mut group = c.benchmark_group("iter_simd_generic_add");
    group.bench_with_input(
        BenchmarkId::new("generic", WORK_ITEMS),
        &(&left, &right),
        |b, (left, right)| {
            b.iter(|| black_box(generic_add(black_box(left), black_box(right))));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("scalar", WORK_ITEMS),
        &(&left, &right),
        |b, (left, right)| {
            b.iter(|| black_box(scalar_add(black_box(left), black_box(right))));
        },
    );
    group.finish();

    let mut group = c.benchmark_group("iter_simd_generic_dot");
    group.bench_with_input(
        BenchmarkId::new("generic", WORK_ITEMS),
        &(&left, &right),
        |b, (left, right)| {
            b.iter(|| black_box(generic_dot(black_box(left), black_box(right))));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("scalar", WORK_ITEMS),
        &(&left, &right),
        |b, (left, right)| {
            b.iter(|| black_box(scalar_dot(black_box(left), black_box(right))));
        },
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = time_model::criterion();
    targets = iter_simd_comparison
}
criterion_main!(benches);
