//! Owned NUMA iterator comparison against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::numa::{NumaContext, NumaPolicy};
use rayon::prelude::*;
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const WARM_UP_MILLIS: u64 = 100;
const MEASUREMENT_MILLIS: u64 = 300;
const WORK_ITEMS: usize = 512;

fn source_data() -> Vec<u64> {
    (0..WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(17).wrapping_add(11))
        .collect()
}

fn moirai_numa_context_map(context: &NumaContext, data: Vec<u64>) -> u64 {
    context
        .execute_iter(data, |value| value.wrapping_mul(5).wrapping_add(3))
        .expect("NUMA context map must complete")
        .into_iter()
        .sum()
}

fn rayon_owned_map(data: Vec<u64>) -> u64 {
    data.into_par_iter()
        .map(|value| value.wrapping_mul(5).wrapping_add(3))
        .sum()
}

fn numa_context_comparison(c: &mut Criterion) {
    let data = source_data();
    let context = NumaContext::new(NumaPolicy::Local);

    assert_eq!(
        moirai_numa_context_map(&context, data.clone()),
        rayon_owned_map(data.clone())
    );

    let mut group = c.benchmark_group("numa_context_owned_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_numa_context_map(&context, black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_owned_map(black_box(input.clone()))))
    });
    group.finish();
}

criterion_group!(benches, numa_context_comparison);
criterion_main!(benches);
