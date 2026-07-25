//! Slice parallel sorting comparison benchmarks against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::ParallelSliceMut as MoiraiParallelSliceMut;
use rayon::slice::ParallelSliceMut as RayonParallelSliceMut;
use std::time::Duration;

const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;

/// Small input: recursion stays shallow, so this row measures per-call overhead
/// rather than the parallel decomposition.
const WORK_ITEMS: usize = 10_000;

/// Large input: `WORK_ITEMS` forks only a handful of times, which is why it
/// could not observe the fork ceiling ADR-022 removes. At this size the stable
/// sort decomposes into ~2000 leaves (`len / 2048`) and the unstable sort into
/// ~250, both far past any worker count, so the row measures how much of the
/// work tree the runtime actually spreads.
const LARGE_ITEMS: usize = 4_000_000;

fn generate_random_data(items: usize) -> Vec<i32> {
    let mut seed: u64 = 54321;
    let mut random_u32 = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed as u32
    };

    let mut data = Vec::with_capacity(items);
    for _ in 0..items {
        data.push((random_u32() % 100_000) as i32);
    }
    data
}

/// Assert the two implementations agree before either is timed: a benchmark of
/// a wrong sort measures nothing.
fn assert_agrees_with_rayon(data: &[i32]) {
    let mut moirai_stable = data.to_vec();
    MoiraiParallelSliceMut::par_sort(moirai_stable.as_mut_slice());
    let mut rayon_stable = data.to_vec();
    RayonParallelSliceMut::par_sort(rayon_stable.as_mut_slice());
    assert_eq!(moirai_stable, rayon_stable);

    let mut moirai_unstable = data.to_vec();
    MoiraiParallelSliceMut::par_sort_unstable(moirai_unstable.as_mut_slice());
    let mut rayon_unstable = data.to_vec();
    RayonParallelSliceMut::par_sort_unstable(rayon_unstable.as_mut_slice());
    assert_eq!(moirai_unstable, rayon_unstable);
}

fn bench_group(
    c: &mut Criterion,
    group_name: &str,
    items: usize,
    data: &[i32],
    moirai_sort: fn(&mut [i32]),
    rayon_sort: fn(&mut [i32]),
) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", items), &data, |b, input| {
        b.iter_with_setup(
            || input.to_vec(),
            |mut v| {
                moirai_sort(v.as_mut_slice());
                black_box(v);
            },
        )
    });
    group.bench_with_input(BenchmarkId::new("rayon", items), &data, |b, input| {
        b.iter_with_setup(
            || input.to_vec(),
            |mut v| {
                rayon_sort(v.as_mut_slice());
                black_box(v);
            },
        )
    });
    group.finish();
}

fn bench_size(c: &mut Criterion, items: usize, stable_group: &str, unstable_group: &str) {
    let data = generate_random_data(items);
    assert_agrees_with_rayon(&data);

    bench_group(
        c,
        stable_group,
        items,
        &data,
        MoiraiParallelSliceMut::par_sort,
        RayonParallelSliceMut::par_sort,
    );
    bench_group(
        c,
        unstable_group,
        items,
        &data,
        MoiraiParallelSliceMut::par_sort_unstable,
        RayonParallelSliceMut::par_sort_unstable,
    );
}

fn sorting_comparison(c: &mut Criterion) {
    bench_size(
        c,
        WORK_ITEMS,
        "parallel_sorting_stable",
        "parallel_sorting_unstable",
    );
    bench_size(
        c,
        LARGE_ITEMS,
        "parallel_sorting_stable_large",
        "parallel_sorting_unstable_large",
    );
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
