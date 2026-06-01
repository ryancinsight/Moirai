//! Iterator adapter comparison benchmarks against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use moirai_iter::parallel::Either as MoiraiEither;
use moirai_iter::parallel::IndexedParallelIterator as MoiraiIndexedParallelIterator;
use moirai_iter::parallel::IntoParallelIterator as MoiraiIntoParallelIterator;
use moirai_iter::parallel::IntoParallelRefIterator as MoiraiIntoParallelRefIterator;
use moirai_iter::parallel::ParallelIterator as MoiraiParallelIterator;
use rayon::iter::Either as RayonEither;
use rayon::iter::IndexedParallelIterator as RayonIndexedParallelIterator;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const SAMPLE_SIZE: usize = 30;
const MEASUREMENT_MILLIS: u64 = 750;
const WARM_UP_MILLIS: u64 = 250;
const WORK_ITEMS: usize = 32_768;
const CHUNK_SIZE: usize = 64;

struct NonCloneBenchValue {
    value: u64,
}

fn source_data() -> Vec<u64> {
    (0..WORK_ITEMS as u64).collect()
}

fn pair_source_data() -> Vec<(u64, u64)> {
    (0..WORK_ITEMS as u64)
        .map(|value| {
            (
                value.wrapping_mul(3).wrapping_add(1),
                value.wrapping_mul(5).wrapping_add(7),
            )
        })
        .collect()
}

fn non_clone_source_data() -> Vec<NonCloneBenchValue> {
    (0..WORK_ITEMS as u64)
        .map(|value| NonCloneBenchValue {
            value: value.wrapping_mul(17).wrapping_add(11),
        })
        .collect()
}

fn moirai_indexed_boundary<Owned, Empty, Range>(
    owned: &Owned,
    empty: &Empty,
    range: &Range,
) -> (usize, bool, usize)
where
    Owned: MoiraiIndexedParallelIterator,
    Empty: MoiraiIndexedParallelIterator,
    Range: MoiraiIndexedParallelIterator,
{
    let owned_len = MoiraiIndexedParallelIterator::len(owned);
    let empty_flag = MoiraiIndexedParallelIterator::is_empty(empty);
    let range_len = MoiraiIndexedParallelIterator::len(range);
    (owned_len, empty_flag, range_len)
}

fn rayon_indexed_boundary<Owned, Empty, Range>(
    owned: &Owned,
    empty: &Empty,
    range: &Range,
) -> (usize, bool, usize)
where
    Owned: RayonIndexedParallelIterator,
    Empty: RayonIndexedParallelIterator,
    Range: RayonIndexedParallelIterator,
{
    let owned_len = RayonIndexedParallelIterator::len(owned);
    let empty_flag = RayonIndexedParallelIterator::len(empty) == 0;
    let range_len = RayonIndexedParallelIterator::len(range);
    (owned_len, empty_flag, range_len)
}

fn collect_checksum(values: &[u64]) -> u64 {
    values
        .iter()
        .fold(0_u64, |acc, value| acc.wrapping_add(*value))
}

fn moirai_collect_into_vec_pipeline(data: Vec<u64>, output: &mut Vec<u64>) -> u64 {
    MoiraiIndexedParallelIterator::collect_into_vec(
        MoiraiIntoParallelIterator::into_par_iter(data),
        output,
    );
    collect_checksum(output)
}

fn rayon_collect_into_vec_pipeline(data: Vec<u64>, output: &mut Vec<u64>) -> u64 {
    RayonIndexedParallelIterator::collect_into_vec(
        rayon::prelude::IntoParallelIterator::into_par_iter(data),
        output,
    );
    collect_checksum(output)
}

fn moirai_unzip_into_vecs_pipeline(
    data: Vec<(u64, u64)>,
    left: &mut Vec<u64>,
    right: &mut Vec<u64>,
) -> (u64, u64) {
    MoiraiIndexedParallelIterator::unzip_into_vecs(
        MoiraiIntoParallelIterator::into_par_iter(data),
        left,
        right,
    );
    (collect_checksum(left), collect_checksum(right))
}

fn rayon_unzip_into_vecs_pipeline(
    data: Vec<(u64, u64)>,
    left: &mut Vec<u64>,
    right: &mut Vec<u64>,
) -> (u64, u64) {
    RayonIndexedParallelIterator::unzip_into_vecs(
        rayon::prelude::IntoParallelIterator::into_par_iter(data),
        left,
        right,
    );
    (collect_checksum(left), collect_checksum(right))
}

fn moirai_indexed_pipeline(data: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3))
        .enumerate()
        .filter(|(index, _)| index % 2 == 0)
        .map(|(_, value)| value)
        .take(WORK_ITEMS / 2)
        .skip(64)
        .collect::<Vec<_>>()
}

fn rayon_indexed_pipeline(data: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3))
        .enumerate()
        .filter(|(index, _)| index % 2 == 0)
        .map(|(_, value)| value)
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 2)
        .skip(64)
        .collect()
}

fn moirai_filter_flat_pipeline(data: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .filter_map(|value| (value % 3 != 0).then_some(value.wrapping_mul(3)))
        .flat_map_iter(|value| [value, value.wrapping_add(1)])
        .collect::<Vec<_>>()
}

fn rayon_filter_flat_pipeline(data: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .filter_map(|value| (value % 3 != 0).then_some(value.wrapping_mul(3)))
        .flat_map_iter(|value| [value, value.wrapping_add(1)])
        .collect::<Vec<_>>()
}

fn nested_source_data() -> Vec<Vec<u64>> {
    source_data()
        .chunks(CHUNK_SIZE)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn moirai_flatten_pipeline(data: Vec<Vec<u64>>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .flatten_iter()
        .map(|value| value.wrapping_mul(13).wrapping_add(5))
        .filter(|value| value % 7 != 0)
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>()
}

fn rayon_flatten_pipeline(data: Vec<Vec<u64>>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .flatten_iter()
        .map(|value| value.wrapping_mul(13).wrapping_add(5))
        .filter(|value| value % 7 != 0)
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 2)
        .collect()
}

fn moirai_take_skip_any_pipeline(data: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| {
            black_box(value);
            17_u64
        })
        .take_any(WORK_ITEMS / 2)
        .skip_any(128)
        .collect::<Vec<_>>()
}

fn rayon_take_skip_any_pipeline(data: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| {
            black_box(value);
            17_u64
        })
        .take_any(WORK_ITEMS / 2)
        .collect::<Vec<_>>()
        .into_iter()
        .skip(128)
        .collect()
}

fn moirai_take_skip_any_while_pipeline(data: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    let taken = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .take_any_while(|value| *value != u64::MAX)
        .collect::<Vec<_>>();
    let skipped = MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .skip_any_while(|value| *value == u64::MAX)
        .collect::<Vec<_>>();

    (taken, skipped)
}

fn rayon_take_skip_any_while_pipeline(data: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    let taken = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .take_any_while(|value| *value != u64::MAX)
        .collect::<Vec<_>>();
    let skipped = rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .skip_any_while(|value| *value == u64::MAX)
        .collect::<Vec<_>>();

    (taken, skipped)
}

fn moirai_map_state_pipeline(data: Vec<u64>) -> (Vec<u64>, u64, Vec<u64>, u64) {
    let with_checksum = Arc::new(AtomicU64::new(0));
    let with = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map_with(Arc::clone(&with_checksum), |state, value| {
            state.fetch_add(value, Ordering::Relaxed);
            value.wrapping_mul(3).wrapping_add(1)
        })
        .filter(|value| value % 2 == 0)
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>();

    let init_checksum = Arc::new(AtomicU64::new(0));
    let init_sink = Arc::clone(&init_checksum);
    let init = MoiraiIntoParallelIterator::into_par_iter(data)
        .map_init(
            || Arc::clone(&init_sink),
            |state, value| {
                state.fetch_add(value, Ordering::Relaxed);
                value.wrapping_mul(5).wrapping_add(7)
            },
        )
        .filter(|value| value % 3 != 0)
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>();

    (
        with,
        with_checksum.load(Ordering::Relaxed),
        init,
        init_checksum.load(Ordering::Relaxed),
    )
}

fn rayon_map_state_pipeline(data: Vec<u64>) -> (Vec<u64>, u64, Vec<u64>, u64) {
    let with_checksum = Arc::new(AtomicU64::new(0));
    let with = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map_with(Arc::clone(&with_checksum), |state, value| {
            state.fetch_add(value, Ordering::Relaxed);
            value.wrapping_mul(3).wrapping_add(1)
        })
        .filter(|value| value % 2 == 0)
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>();

    let init_checksum = Arc::new(AtomicU64::new(0));
    let init_sink = Arc::clone(&init_checksum);
    let init = rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map_init(
            || Arc::clone(&init_sink),
            |state, value| {
                state.fetch_add(value, Ordering::Relaxed);
                value.wrapping_mul(5).wrapping_add(7)
            },
        )
        .filter(|value| value % 3 != 0)
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>();

    (
        with,
        with_checksum.load(Ordering::Relaxed),
        init,
        init_checksum.load(Ordering::Relaxed),
    )
}

fn moirai_update_pipeline(data: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .update(|value| {
            *value = value.wrapping_mul(7).wrapping_add(3);
        })
        .filter(|value| value % 5 != 0)
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>()
}

fn rayon_update_pipeline(data: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .update(|value| {
            *value = value.wrapping_mul(7).wrapping_add(3);
        })
        .filter(|value| value % 5 != 0)
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 2)
        .collect()
}

fn moirai_while_some_pipeline(data: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| Some(value.wrapping_mul(3).wrapping_add(1)))
        .while_some()
        .collect::<Vec<_>>()
}

fn rayon_while_some_pipeline(data: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| Some(value.wrapping_mul(3).wrapping_add(1)))
        .while_some()
        .collect::<Vec<_>>()
}

fn moirai_try_for_each_pipeline(data: Vec<u64>) -> Result<u64, u64> {
    let checksum = AtomicU64::new(0);
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .try_for_each(|value| {
            checksum.fetch_add(value, Ordering::Relaxed);
            Ok::<(), u64>(())
        })?;
    Ok(checksum.load(Ordering::Relaxed))
}

fn rayon_try_for_each_pipeline(data: Vec<u64>) -> Result<u64, u64> {
    let checksum = AtomicU64::new(0);
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .try_for_each(|value| {
            checksum.fetch_add(value, Ordering::Relaxed);
            Ok::<(), u64>(())
        })?;
    Ok(checksum.load(Ordering::Relaxed))
}

fn moirai_for_each_state_pipeline(data: Vec<u64>) -> (u64, u64) {
    let with_checksum = Arc::new(AtomicU64::new(0));
    MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .for_each_with(Arc::clone(&with_checksum), |state, value| {
            state.fetch_add(value, Ordering::Relaxed);
        });

    let init_checksum = Arc::new(AtomicU64::new(0));
    let init_sink = Arc::clone(&init_checksum);
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(5).wrapping_add(7))
        .for_each_init(
            || Arc::clone(&init_sink),
            |state, value| {
                state.fetch_add(value, Ordering::Relaxed);
            },
        );

    (
        with_checksum.load(Ordering::Relaxed),
        init_checksum.load(Ordering::Relaxed),
    )
}

fn rayon_for_each_state_pipeline(data: Vec<u64>) -> (u64, u64) {
    let with_checksum = Arc::new(AtomicU64::new(0));
    rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .for_each_with(Arc::clone(&with_checksum), |state, value| {
            state.fetch_add(value, Ordering::Relaxed);
        });

    let init_checksum = Arc::new(AtomicU64::new(0));
    let init_sink = Arc::clone(&init_checksum);
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(5).wrapping_add(7))
        .for_each_init(
            || Arc::clone(&init_sink),
            |state, value| {
                state.fetch_add(value, Ordering::Relaxed);
            },
        );

    (
        with_checksum.load(Ordering::Relaxed),
        init_checksum.load(Ordering::Relaxed),
    )
}

fn moirai_try_for_each_state_pipeline(data: Vec<u64>) -> Result<(u64, u64), u64> {
    let with_checksum = Arc::new(AtomicU64::new(0));
    MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(3))
        .try_for_each_with(Arc::clone(&with_checksum), |state, value| {
            state.fetch_add(value, Ordering::Relaxed);
            Ok::<(), u64>(())
        })?;

    let init_checksum = Arc::new(AtomicU64::new(0));
    let init_sink = Arc::clone(&init_checksum);
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(11).wrapping_add(5))
        .try_for_each_init(
            || Arc::clone(&init_sink),
            |state, value| {
                state.fetch_add(value, Ordering::Relaxed);
                Ok::<(), u64>(())
            },
        )?;

    Ok((
        with_checksum.load(Ordering::Relaxed),
        init_checksum.load(Ordering::Relaxed),
    ))
}

fn rayon_try_for_each_state_pipeline(data: Vec<u64>) -> Result<(u64, u64), u64> {
    let with_checksum = Arc::new(AtomicU64::new(0));
    rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(3))
        .try_for_each_with(Arc::clone(&with_checksum), |state, value| {
            state.fetch_add(value, Ordering::Relaxed);
            Ok::<(), u64>(())
        })?;

    let init_checksum = Arc::new(AtomicU64::new(0));
    let init_sink = Arc::clone(&init_checksum);
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(11).wrapping_add(5))
        .try_for_each_init(
            || Arc::clone(&init_sink),
            |state, value| {
                state.fetch_add(value, Ordering::Relaxed);
                Ok::<(), u64>(())
            },
        )?;

    Ok((
        with_checksum.load(Ordering::Relaxed),
        init_checksum.load(Ordering::Relaxed),
    ))
}

fn moirai_try_reduce_pipeline(data: Vec<u64>) -> Result<u64, u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| Ok::<u64, u64>(value.wrapping_mul(3).wrapping_add(1)))
        .try_reduce(
            || 0_u64,
            |left, right| Ok::<u64, u64>(left.wrapping_add(right)),
        )
}

fn rayon_try_reduce_pipeline(data: Vec<u64>) -> Result<u64, u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| Ok::<u64, u64>(value.wrapping_mul(3).wrapping_add(1)))
        .try_reduce(
            || 0_u64,
            |left, right| Ok::<u64, u64>(left.wrapping_add(right)),
        )
}

fn moirai_try_reduce_with_pipeline(data: Vec<u64>) -> Option<Result<u64, u64>> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| Ok::<u64, u64>(value.wrapping_mul(3).wrapping_add(1)))
        .try_reduce_with(|left, right| Ok::<u64, u64>(left.wrapping_add(right)))
}

fn rayon_try_reduce_with_pipeline(data: Vec<u64>) -> Option<Result<u64, u64>> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| Ok::<u64, u64>(value.wrapping_mul(3).wrapping_add(1)))
        .try_reduce_with(|left, right| Ok::<u64, u64>(left.wrapping_add(right)))
}

fn moirai_chain_rev_pipeline(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(left)
        .chain(MoiraiIntoParallelIterator::into_par_iter(right))
        .rev()
        .take(WORK_ITEMS)
        .collect::<Vec<_>>()
}

fn rayon_chain_rev_pipeline(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(left)
        .chain(rayon::prelude::IntoParallelIterator::into_par_iter(right))
        .rev()
        .take(WORK_ITEMS)
        .collect()
}

fn moirai_zip_eq_pipeline(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(left)
        .zip_eq(MoiraiIntoParallelIterator::into_par_iter(right))
        .map(|(left, right)| left.wrapping_mul(3).wrapping_add(right.wrapping_mul(5)))
        .filter(|value| value % 7 != 0)
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>()
}

fn rayon_zip_eq_pipeline(left: Vec<u64>, right: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(left)
        .zip_eq(rayon::prelude::IntoParallelIterator::into_par_iter(right))
        .map(|(left, right)| left.wrapping_mul(3).wrapping_add(right.wrapping_mul(5)))
        .filter(|value| value % 7 != 0)
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 2)
        .collect()
}

fn moirai_interleave_pipeline(
    left: Vec<u64>,
    right: Vec<u64>,
    right_short: Vec<u64>,
) -> (Vec<u64>, Vec<u64>) {
    let full = MoiraiIndexedParallelIterator::interleave(
        MoiraiIntoParallelIterator::into_par_iter(left.clone()),
        right,
    )
    .collect::<Vec<_>>();
    let shortest = MoiraiIndexedParallelIterator::interleave_shortest(
        MoiraiIntoParallelIterator::into_par_iter(left),
        right_short,
    )
    .collect::<Vec<_>>();
    (full, shortest)
}

fn rayon_interleave_pipeline(
    left: Vec<u64>,
    right: Vec<u64>,
    right_short: Vec<u64>,
) -> (Vec<u64>, Vec<u64>) {
    let full = RayonIndexedParallelIterator::interleave(
        rayon::prelude::IntoParallelIterator::into_par_iter(left.clone()),
        right,
    )
    .collect::<Vec<_>>();
    let shortest = RayonIndexedParallelIterator::interleave_shortest(
        rayon::prelude::IntoParallelIterator::into_par_iter(left),
        right_short,
    )
    .collect::<Vec<_>>();
    (full, shortest)
}

fn moirai_step_by_pipeline(data: Vec<u64>) -> Vec<u64> {
    MoiraiIndexedParallelIterator::step_by(MoiraiIntoParallelIterator::into_par_iter(data), 3)
        .map(|value| value.wrapping_mul(11).wrapping_add(5))
        .collect::<Vec<_>>()
}

fn rayon_step_by_pipeline(data: Vec<u64>) -> Vec<u64> {
    RayonIndexedParallelIterator::step_by(
        rayon::prelude::IntoParallelIterator::into_par_iter(data),
        3,
    )
    .map(|value| value.wrapping_mul(11).wrapping_add(5))
    .collect::<Vec<_>>()
}

fn moirai_intersperse_pipeline(data: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .filter(|value| value % 3 != 0)
        .take(WORK_ITEMS / 2)
        .intersperse(u64::MAX)
        .collect::<Vec<_>>()
}

fn rayon_intersperse_pipeline(data: Vec<u64>) -> Vec<u64> {
    let prefix = rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .filter(|value| value % 3 != 0)
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>();
    rayon::prelude::IntoParallelIterator::into_par_iter(prefix)
        .intersperse(u64::MAX)
        .collect()
}

fn moirai_inspect_chunks_pipeline(data: Vec<u64>) -> Vec<u64> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .inspect(|value| {
            black_box(*value);
        })
        .panic_fuse()
        .chunks(CHUNK_SIZE)
        .map(|chunk| chunk.into_iter().sum::<u64>())
        .collect::<Vec<_>>()
}

fn rayon_inspect_chunks_pipeline(data: Vec<u64>) -> Vec<u64> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .inspect(|value| {
            black_box(*value);
        })
        .panic_fuse()
        .chunks(CHUNK_SIZE)
        .map(|chunk| chunk.into_iter().sum::<u64>())
        .collect()
}

fn moirai_partition_pipeline(data: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    MoiraiIntoParallelIterator::into_par_iter(data).partition(|value| value % 2 == 0)
}

fn rayon_partition_pipeline(data: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    rayon::prelude::IntoParallelIterator::into_par_iter(data).partition(|value| value % 2 == 0)
}

fn moirai_terminal_reducer_pipeline(data: Vec<u64>) -> (u64, Option<u64>, Option<u64>) {
    let sum = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3))
        .filter(|value| value % 2 == 0)
        .sum::<u64>();
    let min = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .min();
    let max = MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .max();
    (sum, min, max)
}

fn rayon_terminal_reducer_pipeline(data: Vec<u64>) -> (u64, Option<u64>, Option<u64>) {
    let sum = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3))
        .filter(|value| value % 2 == 0)
        .sum::<u64>();
    let min = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .min();
    let max = rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .max();
    (sum, min, max)
}

fn moirai_ordered_reducer_pipeline(
    data: Vec<u64>,
) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    let min_by = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .min_by(|left, right| left.cmp(right));
    let max_by = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .max_by(|left, right| left.cmp(right));
    let min_by_key = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .min_by_key(|value| value.reverse_bits());
    let max_by_key = MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .max_by_key(|value| value.reverse_bits());
    (min_by, max_by, min_by_key, max_by_key)
}

fn rayon_ordered_reducer_pipeline(
    data: Vec<u64>,
) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    let min_by = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .min_by(|left, right| left.cmp(right));
    let max_by = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .max_by(|left, right| left.cmp(right));
    let min_by_key = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .min_by_key(|value| value.reverse_bits());
    let max_by_key = rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(7).wrapping_add(11))
        .max_by_key(|value| value.reverse_bits());
    (min_by, max_by, min_by_key, max_by_key)
}

fn moirai_find_map_pipeline(
    data: Vec<u64>,
) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    let first_target = WORK_ITEMS as u64 - 2;
    let any_target = WORK_ITEMS as u64 - 1;
    let first = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .find_map_first(|value| {
            (value == first_target.wrapping_mul(3).wrapping_add(1)).then_some(value)
        });
    let any = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(5).wrapping_add(7))
        .find_map_any(|value| {
            (value == any_target.wrapping_mul(5).wrapping_add(7)).then_some(value)
        });
    let last =
        MoiraiIntoParallelIterator::into_par_iter(data.clone()).find_last(|value| value % 7 == 3);
    let last_map = MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(11).wrapping_add(17))
        .find_map_last(|value| (value % 7 == 3).then_some(value));
    (first, any, last, last_map)
}

fn rayon_find_map_pipeline(data: Vec<u64>) -> (Option<u64>, Option<u64>, Option<u64>, Option<u64>) {
    let first_target = WORK_ITEMS as u64 - 2;
    let any_target = WORK_ITEMS as u64 - 1;
    let first = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .find_map_first(|value| {
            (value == first_target.wrapping_mul(3).wrapping_add(1)).then_some(value)
        });
    let any = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(5).wrapping_add(7))
        .find_map_any(|value| {
            (value == any_target.wrapping_mul(5).wrapping_add(7)).then_some(value)
        });
    let last = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .find_last(|value| value % 7 == 3);
    let last_map = rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(11).wrapping_add(17))
        .find_map_last(|value| (value % 7 == 3).then_some(value));
    (first, any, last, last_map)
}

fn moirai_position_pipeline(data: Vec<u64>) -> (Option<usize>, Option<usize>, Option<usize>) {
    let first_target = WORK_ITEMS as u64 - 2;
    let any_target = WORK_ITEMS as u64 - 1;
    let first = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .position_first(|value| value == first_target.wrapping_mul(3).wrapping_add(1));
    let any = MoiraiIntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(5).wrapping_add(7))
        .position_any(|value| value == any_target.wrapping_mul(5).wrapping_add(7));
    let last = MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value % 7)
        .position_last(|value| value == 3);
    (first, any, last)
}

fn rayon_position_pipeline(data: Vec<u64>) -> (Option<usize>, Option<usize>, Option<usize>) {
    let first_target = WORK_ITEMS as u64 - 2;
    let any_target = WORK_ITEMS as u64 - 1;
    let first = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .position_first(|value| value == first_target.wrapping_mul(3).wrapping_add(1));
    let any = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone())
        .map(|value| value.wrapping_mul(5).wrapping_add(7))
        .position_any(|value| value == any_target.wrapping_mul(5).wrapping_add(7));
    let last = rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value % 7)
        .position_last(|value| value == 3);
    (first, any, last)
}

fn moirai_positions_pipeline(data: Vec<u64>) -> Vec<usize> {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .positions(|value| value % 11 == 0)
        .collect::<Vec<_>>()
}

fn rayon_positions_pipeline(data: Vec<u64>) -> Vec<usize> {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .positions(|value| value % 11 == 0)
        .collect()
}

fn moirai_ref_copied_cloned_pipeline(data: &Vec<u64>) -> (Vec<u64>, Vec<String>) {
    let copied = MoiraiIntoParallelRefIterator::par_iter(data)
        .copied()
        .map(|value| value.wrapping_mul(5))
        .filter(|value| value % 3 != 0)
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>();
    let labels = data
        .iter()
        .map(|value| format!("label-{value}"))
        .collect::<Vec<_>>();
    let cloned = MoiraiIntoParallelRefIterator::par_iter(&labels)
        .cloned()
        .filter(|value| value.as_bytes().last().is_some_and(|byte| byte % 2 == 0))
        .take(WORK_ITEMS / 4)
        .collect::<Vec<_>>();
    (copied, cloned)
}

fn rayon_ref_copied_cloned_pipeline(data: &Vec<u64>) -> (Vec<u64>, Vec<String>) {
    let copied = rayon::prelude::IntoParallelRefIterator::par_iter(data)
        .copied()
        .map(|value| value.wrapping_mul(5))
        .filter(|value| value % 3 != 0)
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 2)
        .collect::<Vec<_>>();
    let labels = data
        .iter()
        .map(|value| format!("label-{value}"))
        .collect::<Vec<_>>();
    let cloned = rayon::prelude::IntoParallelRefIterator::par_iter(&labels)
        .cloned()
        .filter(|value| value.as_bytes().last().is_some_and(|byte| byte % 2 == 0))
        .collect::<Vec<_>>()
        .into_iter()
        .take(WORK_ITEMS / 4)
        .collect::<Vec<_>>();
    (copied, cloned)
}

fn moirai_non_clone_ref_map(data: &Vec<NonCloneBenchValue>) -> u64 {
    MoiraiIntoParallelRefIterator::par_iter(data)
        .map(|item| item.value.wrapping_mul(3).wrapping_add(1))
        .sum::<u64>()
}

fn rayon_non_clone_ref_map(data: &Vec<NonCloneBenchValue>) -> u64 {
    rayon::prelude::IntoParallelRefIterator::par_iter(data)
        .map(|item| item.value.wrapping_mul(3).wrapping_add(1))
        .sum::<u64>()
}

fn moirai_unzip_pipeline(data: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| (value.wrapping_mul(3), value.wrapping_mul(5)))
        .filter(|(left, _)| left % 2 == 0)
        .unzip()
}

fn rayon_unzip_pipeline(data: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| (value.wrapping_mul(3), value.wrapping_mul(5)))
        .filter(|(left, _)| left % 2 == 0)
        .unzip()
}

fn moirai_partition_map_pipeline(data: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    MoiraiIntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .partition_map(|value| {
            if value % 5 == 0 {
                MoiraiEither::Left(value.wrapping_mul(7))
            } else {
                MoiraiEither::Right(value.wrapping_mul(11))
            }
        })
}

fn rayon_partition_map_pipeline(data: Vec<u64>) -> (Vec<u64>, Vec<u64>) {
    rayon::prelude::IntoParallelIterator::into_par_iter(data)
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
        .partition_map(|value| {
            if value % 5 == 0 {
                RayonEither::Left(value.wrapping_mul(7))
            } else {
                RayonEither::Right(value.wrapping_mul(11))
            }
        })
}

fn iterator_adapter_comparison(c: &mut Criterion) {
    let data = source_data();
    let non_clone_data = non_clone_source_data();
    let moirai_owned = MoiraiIntoParallelIterator::into_par_iter(data.clone());
    let moirai_empty = MoiraiIntoParallelIterator::into_par_iter(Vec::<u64>::new());
    let moirai_range = MoiraiIntoParallelIterator::into_par_iter(0..WORK_ITEMS);
    let rayon_owned = rayon::prelude::IntoParallelIterator::into_par_iter(data.clone());
    let rayon_empty = rayon::prelude::IntoParallelIterator::into_par_iter(Vec::<u64>::new());
    let rayon_range = rayon::prelude::IntoParallelIterator::into_par_iter(0..WORK_ITEMS);

    let moirai_expected = moirai_indexed_boundary(&moirai_owned, &moirai_empty, &moirai_range);
    let rayon_expected = rayon_indexed_boundary(&rayon_owned, &rayon_empty, &rayon_range);
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_indexed_boundary");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_function(BenchmarkId::new("moirai", WORK_ITEMS), |b| {
        b.iter(|| {
            black_box(moirai_indexed_boundary(
                black_box(&moirai_owned),
                black_box(&moirai_empty),
                black_box(&moirai_range),
            ))
        })
    });
    group.bench_function(BenchmarkId::new("rayon", WORK_ITEMS), |b| {
        b.iter(|| {
            black_box(rayon_indexed_boundary(
                black_box(&rayon_owned),
                black_box(&rayon_empty),
                black_box(&rayon_range),
            ))
        })
    });
    group.finish();

    let mut moirai_output = Vec::with_capacity(WORK_ITEMS);
    let mut rayon_output = Vec::with_capacity(WORK_ITEMS);
    let moirai_expected = moirai_collect_into_vec_pipeline(data.clone(), &mut moirai_output);
    let rayon_expected = rayon_collect_into_vec_pipeline(data.clone(), &mut rayon_output);
    assert_eq!(moirai_output, rayon_output);
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_indexed_collect_into_vec");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter_batched(
            || (input.clone(), Vec::with_capacity(WORK_ITEMS)),
            |(source, mut output)| black_box(moirai_collect_into_vec_pipeline(source, &mut output)),
            BatchSize::SmallInput,
        )
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter_batched(
            || (input.clone(), Vec::with_capacity(WORK_ITEMS)),
            |(source, mut output)| black_box(rayon_collect_into_vec_pipeline(source, &mut output)),
            BatchSize::SmallInput,
        )
    });
    group.finish();

    let pair_data = pair_source_data();
    let mut moirai_left = Vec::with_capacity(WORK_ITEMS);
    let mut moirai_right = Vec::with_capacity(WORK_ITEMS);
    let mut rayon_left = Vec::with_capacity(WORK_ITEMS);
    let mut rayon_right = Vec::with_capacity(WORK_ITEMS);
    let moirai_expected =
        moirai_unzip_into_vecs_pipeline(pair_data.clone(), &mut moirai_left, &mut moirai_right);
    let rayon_expected =
        rayon_unzip_into_vecs_pipeline(pair_data.clone(), &mut rayon_left, &mut rayon_right);
    assert_eq!(moirai_left, rayon_left);
    assert_eq!(moirai_right, rayon_right);
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_indexed_unzip_into_vecs");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", WORK_ITEMS),
        &pair_data,
        |b, input| {
            b.iter_batched(
                || {
                    (
                        input.clone(),
                        Vec::with_capacity(WORK_ITEMS),
                        Vec::with_capacity(WORK_ITEMS),
                    )
                },
                |(source, mut left, mut right)| {
                    black_box(moirai_unzip_into_vecs_pipeline(
                        source, &mut left, &mut right,
                    ))
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new("rayon", WORK_ITEMS),
        &pair_data,
        |b, input| {
            b.iter_batched(
                || {
                    (
                        input.clone(),
                        Vec::with_capacity(WORK_ITEMS),
                        Vec::with_capacity(WORK_ITEMS),
                    )
                },
                |(source, mut left, mut right)| {
                    black_box(rayon_unzip_into_vecs_pipeline(
                        source, &mut left, &mut right,
                    ))
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();

    let moirai_expected = moirai_indexed_pipeline(data.clone());
    let rayon_expected = rayon_indexed_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_indexed_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_indexed_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_indexed_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_filter_flat_pipeline(data.clone());
    let rayon_expected = rayon_filter_flat_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_filter_flat_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_filter_flat_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_filter_flat_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let nested = nested_source_data();
    let moirai_expected = moirai_flatten_pipeline(nested.clone());
    let rayon_expected = rayon_flatten_pipeline(nested.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_flatten");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(
        BenchmarkId::new("moirai", WORK_ITEMS),
        &nested,
        |b, input| b.iter(|| black_box(moirai_flatten_pipeline(black_box(input.clone())))),
    );
    group.bench_with_input(
        BenchmarkId::new("rayon", WORK_ITEMS),
        &nested,
        |b, input| b.iter(|| black_box(rayon_flatten_pipeline(black_box(input.clone())))),
    );
    group.finish();

    let moirai_expected = moirai_take_skip_any_pipeline(data.clone());
    let rayon_expected = rayon_take_skip_any_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_take_skip_any");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_take_skip_any_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_take_skip_any_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_take_skip_any_while_pipeline(data.clone());
    let rayon_expected = rayon_take_skip_any_while_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_take_skip_any_while");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| {
            black_box(moirai_take_skip_any_while_pipeline(black_box(
                input.clone(),
            )))
        })
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_take_skip_any_while_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_map_state_pipeline(data.clone());
    let rayon_expected = rayon_map_state_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_map_state");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_map_state_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_map_state_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_update_pipeline(data.clone());
    let rayon_expected = rayon_update_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_update");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_update_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_update_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_while_some_pipeline(data.clone());
    let rayon_expected = rayon_while_some_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_while_some");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_while_some_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_while_some_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_try_for_each_pipeline(data.clone());
    let rayon_expected = rayon_try_for_each_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_try_for_each");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_try_for_each_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_try_for_each_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_for_each_state_pipeline(data.clone());
    let rayon_expected = rayon_for_each_state_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_for_each_state");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_for_each_state_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_for_each_state_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_try_for_each_state_pipeline(data.clone());
    let rayon_expected = rayon_try_for_each_state_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_try_for_each_state");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_try_for_each_state_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_try_for_each_state_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_try_reduce_pipeline(data.clone());
    let rayon_expected = rayon_try_reduce_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_try_reduce");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_try_reduce_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_try_reduce_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_try_reduce_with_pipeline(data.clone());
    let rayon_expected = rayon_try_reduce_with_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_try_reduce_with");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_try_reduce_with_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_try_reduce_with_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let left = source_data();
    let right = source_data()
        .into_iter()
        .map(|value| value.wrapping_add(WORK_ITEMS as u64))
        .collect::<Vec<_>>();
    let moirai_expected = moirai_chain_rev_pipeline(left.clone(), right.clone());
    let rayon_expected = rayon_chain_rev_pipeline(left.clone(), right.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_chain_rev_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_function(BenchmarkId::new("moirai", WORK_ITEMS), |b| {
        b.iter(|| {
            black_box(moirai_chain_rev_pipeline(
                black_box(left.clone()),
                black_box(right.clone()),
            ))
        })
    });
    group.bench_function(BenchmarkId::new("rayon", WORK_ITEMS), |b| {
        b.iter(|| {
            black_box(rayon_chain_rev_pipeline(
                black_box(left.clone()),
                black_box(right.clone()),
            ))
        })
    });
    group.finish();

    let moirai_expected = moirai_zip_eq_pipeline(left.clone(), right.clone());
    let rayon_expected = rayon_zip_eq_pipeline(left.clone(), right.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_zip_eq");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_function(BenchmarkId::new("moirai", WORK_ITEMS), |b| {
        b.iter(|| {
            black_box(moirai_zip_eq_pipeline(
                black_box(left.clone()),
                black_box(right.clone()),
            ))
        })
    });
    group.bench_function(BenchmarkId::new("rayon", WORK_ITEMS), |b| {
        b.iter(|| {
            black_box(rayon_zip_eq_pipeline(
                black_box(left.clone()),
                black_box(right.clone()),
            ))
        })
    });
    group.finish();

    let right_short = right
        .iter()
        .take(WORK_ITEMS / 2)
        .copied()
        .collect::<Vec<_>>();
    let moirai_expected =
        moirai_interleave_pipeline(left.clone(), right.clone(), right_short.clone());
    let rayon_expected =
        rayon_interleave_pipeline(left.clone(), right.clone(), right_short.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_indexed_interleave");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_function(BenchmarkId::new("moirai", WORK_ITEMS), |b| {
        b.iter(|| {
            black_box(moirai_interleave_pipeline(
                black_box(left.clone()),
                black_box(right.clone()),
                black_box(right_short.clone()),
            ))
        })
    });
    group.bench_function(BenchmarkId::new("rayon", WORK_ITEMS), |b| {
        b.iter(|| {
            black_box(rayon_interleave_pipeline(
                black_box(left.clone()),
                black_box(right.clone()),
                black_box(right_short.clone()),
            ))
        })
    });
    group.finish();

    let moirai_expected = moirai_step_by_pipeline(data.clone());
    let rayon_expected = rayon_step_by_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_indexed_step_by");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_step_by_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_step_by_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_intersperse_pipeline(data.clone());
    let rayon_expected = rayon_intersperse_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_intersperse");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_intersperse_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_intersperse_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_inspect_chunks_pipeline(data.clone());
    let rayon_expected = rayon_inspect_chunks_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_inspect_chunks_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_inspect_chunks_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_inspect_chunks_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_partition_pipeline(data.clone());
    let rayon_expected = rayon_partition_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_partition_pipeline");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_partition_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_partition_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_terminal_reducer_pipeline(data.clone());
    let rayon_expected = rayon_terminal_reducer_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_terminal_reducers");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_terminal_reducer_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_terminal_reducer_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_ordered_reducer_pipeline(data.clone());
    let rayon_expected = rayon_ordered_reducer_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_ordered_reducers");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_ordered_reducer_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_ordered_reducer_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_find_map_pipeline(data.clone());
    let rayon_expected = rayon_find_map_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_find_map");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_find_map_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_find_map_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_position_pipeline(data.clone());
    let rayon_expected = rayon_position_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_position");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_position_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_position_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_positions_pipeline(data.clone());
    let rayon_expected = rayon_positions_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_positions");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_positions_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_positions_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_ref_copied_cloned_pipeline(&data);
    let rayon_expected = rayon_ref_copied_cloned_pipeline(&data);
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_ref_copy_clone");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_ref_copied_cloned_pipeline(black_box(input))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_ref_copied_cloned_pipeline(black_box(input))))
    });
    group.finish();

    let moirai_expected = moirai_non_clone_ref_map(&non_clone_data);
    let rayon_expected = rayon_non_clone_ref_map(&non_clone_data);
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_non_clone_ref_map");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(
        BenchmarkId::new("moirai", WORK_ITEMS),
        &non_clone_data,
        |b, input| b.iter(|| black_box(moirai_non_clone_ref_map(black_box(input)))),
    );
    group.bench_with_input(
        BenchmarkId::new("rayon", WORK_ITEMS),
        &non_clone_data,
        |b, input| b.iter(|| black_box(rayon_non_clone_ref_map(black_box(input)))),
    );
    group.finish();

    let moirai_expected = moirai_unzip_pipeline(data.clone());
    let rayon_expected = rayon_unzip_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_unzip");
    group.sample_size(SAMPLE_SIZE);
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_unzip_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_unzip_pipeline(black_box(input.clone()))))
    });
    group.finish();

    let moirai_expected = moirai_partition_map_pipeline(data.clone());
    let rayon_expected = rayon_partition_map_pipeline(data.clone());
    assert_eq!(moirai_expected, rayon_expected);

    let mut group = c.benchmark_group("iterator_adapter_partition_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(moirai_partition_map_pipeline(black_box(input.clone()))))
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_partition_map_pipeline(black_box(input.clone()))))
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
    targets = iterator_adapter_comparison
}
criterion_main!(benches);
