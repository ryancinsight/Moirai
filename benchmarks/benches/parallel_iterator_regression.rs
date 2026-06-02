//! Focused parallel iterator regression benchmarks against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::parallel::IndexedParallelIterator as MoiraiIndexedParallelIterator;
use moirai_iter::parallel::IntoParallelIterator as MoiraiIntoParallelIterator;
use moirai_iter::parallel::IntoParallelRefIterator as MoiraiIntoParallelRefIterator;
use moirai_iter::parallel::ParallelIterator as MoiraiParallelIterator;
use rayon::iter::IndexedParallelIterator as RayonIndexedParallelIterator;
use rayon::prelude::*;
use std::time::Duration;

const SAMPLE_SIZE: usize = 20;
const WARM_UP_MILLIS: u64 = 200;
const MEASUREMENT_MILLIS: u64 = 500;
const INPUT_SIZES: [usize; 3] = [1_024, 32_768, 131_072];
const CHUNK_SIZE: usize = 256;
const STEP_SIZE: usize = 3;
const NESTED_CHUNK: usize = 64;

fn source_data(len: usize) -> Vec<u64> {
    (0..len as u64)
        .map(|value| value.wrapping_mul(17).wrapping_add(11))
        .collect()
}

fn right_data(len: usize) -> Vec<u64> {
    (0..len as u64)
        .map(|value| value.wrapping_mul(31).wrapping_add(7))
        .collect()
}

fn nested_data(len: usize) -> Vec<Vec<u64>> {
    source_data(len)
        .chunks(NESTED_CHUNK)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn checksum(values: &[u64]) -> u64 {
    values
        .iter()
        .fold(0_u64, |accumulator, value| accumulator.wrapping_add(*value))
}

fn moirai_map_reduce(data: Vec<u64>) -> u64 {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .filter(|value| value % 5 != 0)
        .sum::<u64>()
}

fn rayon_map_reduce(data: Vec<u64>) -> u64 {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .filter(|value| value % 5 != 0)
        .sum::<u64>()
}

fn moirai_zip_filter_collect(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(left)
        .zip_eq(MoiraiIntoParallelIterator::into_par_iter(right))
        .map(|(left, right)| left.wrapping_mul(3) ^ right.wrapping_mul(5))
        .filter(|value| value & 3 != 0)
        .collect::<Vec<_>>()
}

fn rayon_zip_filter_collect(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(left)
        .zip_eq(rayon::prelude::IntoParallelIterator::into_par_iter(right))
        .map(|(left, right)| left.wrapping_mul(3) ^ right.wrapping_mul(5))
        .filter(|value| value & 3 != 0)
        .collect::<Vec<_>>()
}

fn moirai_borrowed_positions(data: &Vec<u64>) -> Vec<usize> {
    MoiraiIntoParallelRefIterator::par_iter(data)
        .positions(|value| *value % 97 == 0)
        .collect::<Vec<_>>()
}

fn rayon_borrowed_positions(data: &Vec<u64>) -> Vec<usize> {
    rayon::iter::IntoParallelRefIterator::par_iter(data)
        .positions(|value| *value % 97 == 0)
        .collect::<Vec<_>>()
}

fn moirai_borrowed_copied_reduce(data: &Vec<u64>) -> u64 {
    MoiraiIntoParallelRefIterator::par_iter(data)
        .copied()
        .map(|value| value.wrapping_mul(19).wrapping_add(23))
        .filter(|value| value & 7 != 0)
        .sum::<u64>()
}

fn rayon_borrowed_copied_reduce(data: &Vec<u64>) -> u64 {
    rayon::iter::IntoParallelRefIterator::par_iter(data)
        .copied()
        .map(|value| value.wrapping_mul(19).wrapping_add(23))
        .filter(|value| value & 7 != 0)
        .sum::<u64>()
}

fn moirai_collect_into_existing(data: Vec<u64>) -> u64 {
    let mut output = Vec::with_capacity(data.len() + 16);
    output.push(u64::MAX);
    MoiraiIndexedParallelIterator::collect_into_vec(
        MoiraiIntoParallelIterator::into_par_iter(data),
        &mut output,
    );
    checksum(&output)
}

fn rayon_collect_into_existing(data: Vec<u64>) -> u64 {
    let mut output = Vec::with_capacity(data.len() + 16);
    output.push(u64::MAX);
    RayonIndexedParallelIterator::collect_into_vec(
        rayon::prelude::IntoParallelIterator::into_par_iter(data),
        &mut output,
    );
    checksum(&output)
}

fn moirai_nested_flatten_reduce(data: Vec<Vec<u64>>) -> u64 {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .flatten_iter()
        .map(|value| value.wrapping_mul(13).wrapping_add(5))
        .filter(|value| value % 11 != 0)
        .sum::<u64>()
}

fn rayon_nested_flatten_reduce(data: Vec<Vec<u64>>) -> u64 {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .flatten_iter()
        .map(|value| value.wrapping_mul(13).wrapping_add(5))
        .filter(|value| value % 11 != 0)
        .sum::<u64>()
}

fn moirai_chunked_map_reduce(data: Vec<u64>) -> u64 {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .chunks(CHUNK_SIZE)
        .map(|chunk| {
            chunk
                .into_iter()
                .map(|value| value.wrapping_mul(29).wrapping_add(3))
                .filter(|value| value % 13 != 0)
                .sum::<u64>()
        })
        .sum::<u64>()
}

fn rayon_chunked_map_reduce(data: Vec<u64>) -> u64 {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .chunks(CHUNK_SIZE)
        .map(|chunk| {
            chunk
                .into_iter()
                .map(|value| value.wrapping_mul(29).wrapping_add(3))
                .filter(|value| value % 13 != 0)
                .sum::<u64>()
        })
        .sum::<u64>()
}

fn moirai_indexed_step_interleave(left: Vec<u64>, right: Vec<u64>) -> u64 {
    MoiraiIntoParallelIterator::into_par_iter(left)
        .step_by(STEP_SIZE)
        .interleave(right)
        .enumerate()
        .map(|(index, value)| value.wrapping_mul((index as u64).wrapping_add(1)))
        .sum::<u64>()
}

fn rayon_indexed_step_interleave(left: Vec<u64>, right: Vec<u64>) -> u64 {
    rayon::prelude::IntoParallelIterator::into_par_iter(left)
        .step_by(STEP_SIZE)
        .interleave(rayon::prelude::IntoParallelIterator::into_par_iter(right))
        .enumerate()
        .map(|(index, value)| value.wrapping_mul((index as u64).wrapping_add(1)))
        .sum::<u64>()
}

fn moirai_partition_unzip(data: Vec<u64>) -> (u64, u64, usize, usize) {
    let (left, right): (Vec<u64>, Vec<u64>) = MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| {
            (
                value.wrapping_mul(37).wrapping_add(5),
                value.wrapping_mul(41).wrapping_add(17),
            )
        })
        .unzip();
    let (accepted, rejected): (Vec<u64>, Vec<u64>) =
        MoiraiIntoParallelIterator::into_par_iter(left).partition(|value| value & 1 == 0);

    (
        checksum(&accepted),
        checksum(&right),
        accepted.len(),
        rejected.len(),
    )
}

fn rayon_partition_unzip(data: Vec<u64>) -> (u64, u64, usize, usize) {
    let (left, right): (Vec<u64>, Vec<u64>) =
        rayon::prelude::IntoParallelIterator::into_par_iter(data)
            .map(|value| {
                (
                    value.wrapping_mul(37).wrapping_add(5),
                    value.wrapping_mul(41).wrapping_add(17),
                )
            })
            .unzip();
    let (accepted, rejected): (Vec<u64>, Vec<u64>) =
        rayon::prelude::IntoParallelIterator::into_par_iter(left).partition(|value| value & 1 == 0);

    (
        checksum(&accepted),
        checksum(&right),
        accepted.len(),
        rejected.len(),
    )
}

fn moirai_position_find(data: Vec<u64>) -> (Option<usize>, Option<u64>, Option<u64>) {
    let position = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(43).wrapping_add(11))
        .position_last(|value| value % 4_099 == 0);
    let first = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .find_first(|value| *value % 7_919 == 0);
    let last =
        MoiraiIntoParallelIterator::into_par_iter(data).find_last(|value| *value % 7_919 == 0);

    (position, first, last)
}

fn rayon_position_find(data: Vec<u64>) -> (Option<usize>, Option<u64>, Option<u64>) {
    let position = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(43).wrapping_add(11))
        .position_last(|value| value % 4_099 == 0);
    let first = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .find_first(|value| *value % 7_919 == 0);
    let last = rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .find_last(|value| *value % 7_919 == 0);

    (position, first, last)
}

fn parallel_iterator_regression(c: &mut Criterion) {
    let mut map_reduce = c.benchmark_group("parallel_iterator_map_reduce_sizes");
    for len in INPUT_SIZES {
        let data = source_data(len);
        assert_eq!(
            moirai_map_reduce(data.clone()),
            rayon_map_reduce(data.clone())
        );
        map_reduce.bench_with_input(BenchmarkId::new("moirai", len), &data, |b, input| {
            b.iter(|| black_box(moirai_map_reduce(black_box(input.clone()))))
        });
        map_reduce.bench_with_input(BenchmarkId::new("rayon", len), &data, |b, input| {
            b.iter(|| black_box(rayon_map_reduce(black_box(input.clone()))))
        });
    }
    map_reduce.finish();

    let mut zip_collect = c.benchmark_group("parallel_iterator_zip_filter_collect_sizes");
    for len in INPUT_SIZES {
        let left = source_data(len);
        let right = right_data(len);
        assert_eq!(
            moirai_zip_filter_collect(left.clone(), right.clone()),
            rayon_zip_filter_collect(left.clone(), right.clone())
        );
        zip_collect.bench_function(BenchmarkId::new("moirai", len), |b| {
            b.iter(|| {
                black_box(moirai_zip_filter_collect(
                    black_box(left.clone()),
                    black_box(right.clone()),
                ))
            })
        });
        zip_collect.bench_function(BenchmarkId::new("rayon", len), |b| {
            b.iter(|| {
                black_box(rayon_zip_filter_collect(
                    black_box(left.clone()),
                    black_box(right.clone()),
                ))
            })
        });
    }
    zip_collect.finish();

    let mut positions = c.benchmark_group("parallel_iterator_borrowed_positions_sizes");
    for len in INPUT_SIZES {
        let data = source_data(len);
        assert_eq!(
            moirai_borrowed_positions(&data),
            rayon_borrowed_positions(&data)
        );
        positions.bench_with_input(BenchmarkId::new("moirai", len), &data, |b, input| {
            b.iter(|| black_box(moirai_borrowed_positions(black_box(input))))
        });
        positions.bench_with_input(BenchmarkId::new("rayon", len), &data, |b, input| {
            b.iter(|| black_box(rayon_borrowed_positions(black_box(input))))
        });
    }
    positions.finish();

    let mut borrowed_reduce = c.benchmark_group("parallel_iterator_borrowed_copied_reduce_sizes");
    for len in INPUT_SIZES {
        let data = source_data(len);
        assert_eq!(
            moirai_borrowed_copied_reduce(&data),
            rayon_borrowed_copied_reduce(&data)
        );
        borrowed_reduce.bench_with_input(BenchmarkId::new("moirai", len), &data, |b, input| {
            b.iter(|| black_box(moirai_borrowed_copied_reduce(black_box(input))))
        });
        borrowed_reduce.bench_with_input(BenchmarkId::new("rayon", len), &data, |b, input| {
            b.iter(|| black_box(rayon_borrowed_copied_reduce(black_box(input))))
        });
    }
    borrowed_reduce.finish();

    let mut collect_existing = c.benchmark_group("parallel_iterator_collect_into_existing_sizes");
    for len in INPUT_SIZES {
        let data = source_data(len);
        assert_eq!(
            moirai_collect_into_existing(data.clone()),
            rayon_collect_into_existing(data.clone())
        );
        collect_existing.bench_with_input(BenchmarkId::new("moirai", len), &data, |b, input| {
            b.iter(|| black_box(moirai_collect_into_existing(black_box(input.clone()))))
        });
        collect_existing.bench_with_input(BenchmarkId::new("rayon", len), &data, |b, input| {
            b.iter(|| black_box(rayon_collect_into_existing(black_box(input.clone()))))
        });
    }
    collect_existing.finish();

    let mut nested = c.benchmark_group("parallel_iterator_nested_flatten_reduce_sizes");
    for len in INPUT_SIZES {
        let data = nested_data(len);
        assert_eq!(
            moirai_nested_flatten_reduce(data.clone()),
            rayon_nested_flatten_reduce(data.clone())
        );
        nested.bench_with_input(BenchmarkId::new("moirai", len), &data, |b, input| {
            b.iter(|| black_box(moirai_nested_flatten_reduce(black_box(input.clone()))))
        });
        nested.bench_with_input(BenchmarkId::new("rayon", len), &data, |b, input| {
            b.iter(|| black_box(rayon_nested_flatten_reduce(black_box(input.clone()))))
        });
    }
    nested.finish();

    let mut chunks = c.benchmark_group("parallel_iterator_chunked_map_reduce_sizes");
    for len in INPUT_SIZES {
        let data = source_data(len);
        assert_eq!(
            moirai_chunked_map_reduce(data.clone()),
            rayon_chunked_map_reduce(data.clone())
        );
        chunks.bench_with_input(BenchmarkId::new("moirai", len), &data, |b, input| {
            b.iter(|| black_box(moirai_chunked_map_reduce(black_box(input.clone()))))
        });
        chunks.bench_with_input(BenchmarkId::new("rayon", len), &data, |b, input| {
            b.iter(|| black_box(rayon_chunked_map_reduce(black_box(input.clone()))))
        });
    }
    chunks.finish();

    let mut step_interleave = c.benchmark_group("parallel_iterator_indexed_step_interleave_sizes");
    for len in INPUT_SIZES {
        let left = source_data(len);
        let right = right_data(len);
        assert_eq!(
            moirai_indexed_step_interleave(left.clone(), right.clone()),
            rayon_indexed_step_interleave(left.clone(), right.clone())
        );
        step_interleave.bench_function(BenchmarkId::new("moirai", len), |b| {
            b.iter(|| {
                black_box(moirai_indexed_step_interleave(
                    black_box(left.clone()),
                    black_box(right.clone()),
                ))
            })
        });
        step_interleave.bench_function(BenchmarkId::new("rayon", len), |b| {
            b.iter(|| {
                black_box(rayon_indexed_step_interleave(
                    black_box(left.clone()),
                    black_box(right.clone()),
                ))
            })
        });
    }
    step_interleave.finish();

    let mut partition_unzip = c.benchmark_group("parallel_iterator_partition_unzip_sizes");
    for len in INPUT_SIZES {
        let data = source_data(len);
        assert_eq!(
            moirai_partition_unzip(data.clone()),
            rayon_partition_unzip(data.clone())
        );
        partition_unzip.bench_with_input(BenchmarkId::new("moirai", len), &data, |b, input| {
            b.iter(|| black_box(moirai_partition_unzip(black_box(input.clone()))))
        });
        partition_unzip.bench_with_input(BenchmarkId::new("rayon", len), &data, |b, input| {
            b.iter(|| black_box(rayon_partition_unzip(black_box(input.clone()))))
        });
    }
    partition_unzip.finish();

    let mut position_find = c.benchmark_group("parallel_iterator_position_find_sizes");
    for len in INPUT_SIZES {
        let data = source_data(len);
        assert_eq!(
            moirai_position_find(data.clone()),
            rayon_position_find(data.clone())
        );
        position_find.bench_with_input(BenchmarkId::new("moirai", len), &data, |b, input| {
            b.iter(|| black_box(moirai_position_find(black_box(input.clone()))))
        });
        position_find.bench_with_input(BenchmarkId::new("rayon", len), &data, |b, input| {
            b.iter(|| black_box(rayon_position_find(black_box(input.clone()))))
        });
    }
    position_find.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_millis(MEASUREMENT_MILLIS))
        .warm_up_time(Duration::from_millis(WARM_UP_MILLIS))
        .without_plots();
    targets = parallel_iterator_regression
}
criterion_main!(benches);
