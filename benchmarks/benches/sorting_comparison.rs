//! Slice parallel sorting comparison benchmarks against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::ParallelSliceMut as MoiraiParallelSliceMut;
use rayon::slice::ParallelSliceMut as RayonParallelSliceMut;
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const WORK_ITEMS: usize = 10_000;

fn generate_random_data() -> Vec<i32> {
    let mut seed: u64 = 54321;
    let mut random_u32 = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed as u32
    };

    let mut data = Vec::with_capacity(WORK_ITEMS);
    for _ in 0..WORK_ITEMS {
        data.push((random_u32() % 100_000) as i32);
    }
    data
}

fn sorting_comparison(c: &mut Criterion) {
    let data = generate_random_data();

    // Verify correctness first
    let mut moirai_stable = data.clone();
    MoiraiParallelSliceMut::par_sort(moirai_stable.as_mut_slice());
    let mut rayon_stable = data.clone();
    RayonParallelSliceMut::par_sort(rayon_stable.as_mut_slice());
    assert_eq!(moirai_stable, rayon_stable);

    let mut moirai_unstable = data.clone();
    MoiraiParallelSliceMut::par_sort_unstable(moirai_unstable.as_mut_slice());
    let mut rayon_unstable = data.clone();
    RayonParallelSliceMut::par_sort_unstable(rayon_unstable.as_mut_slice());
    assert_eq!(moirai_unstable, rayon_unstable);

    // Stable Sorting Group
    let mut group = c.benchmark_group("parallel_sorting_stable");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter_with_setup(
            || input.clone(),
            |mut v| {
                MoiraiParallelSliceMut::par_sort(v.as_mut_slice());
                black_box(v);
            },
        )
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter_with_setup(
            || input.clone(),
            |mut v| {
                RayonParallelSliceMut::par_sort(v.as_mut_slice());
                black_box(v);
            },
        )
    });
    group.finish();

    // Unstable Sorting Group
    let mut group = c.benchmark_group("parallel_sorting_unstable");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter_with_setup(
            || input.clone(),
            |mut v| {
                MoiraiParallelSliceMut::par_sort_unstable(v.as_mut_slice());
                black_box(v);
            },
        )
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter_with_setup(
            || input.clone(),
            |mut v| {
                RayonParallelSliceMut::par_sort_unstable(v.as_mut_slice());
                black_box(v);
            },
        )
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = sorting_comparison
}
criterion_main!(benches);
