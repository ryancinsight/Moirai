//! Indexed worker/caller partitioning and reduction kernels.

use moirai_core::error::{ExecutorError, ExecutorResult, TaskError};

pub(in crate::schedule::runtime) fn inline_map_reduce<T, Map, Reduce>(
    count: usize,
    identity: T,
    map: Map,
    reduce: Reduce,
) -> ExecutorResult<T>
where
    Map: Fn(usize) -> T,
    Reduce: Fn(T, T) -> T,
{
    use std::panic::{catch_unwind, AssertUnwindSafe};
    catch_unwind(AssertUnwindSafe(|| {
        let mut accumulator = identity;
        for index in 0..count {
            accumulator = reduce(accumulator, map(index));
        }
        accumulator
    }))
    .map_err(|_| ExecutorError::SpawnFailed(TaskError::Panicked))
}

pub(in crate::schedule::runtime) fn map_reduce_range<T, Map, Reduce>(
    start: usize,
    end: usize,
    identity: T,
    map: &Map,
    reduce: &Reduce,
) -> T
where
    Map: Fn(usize) -> T,
    Reduce: Fn(T, T) -> T,
{
    let mut accumulator = identity;
    for index in start..end {
        accumulator = reduce(accumulator, map(index));
    }
    accumulator
}

pub(in crate::schedule::runtime) fn indexed_chunk_count(
    count: usize,
    worker_count: usize,
) -> usize {
    count.min(worker_count.max(1).saturating_add(1))
}

pub(in crate::schedule::runtime) fn indexed_chunk_bounds(
    count: usize,
    chunk_count: usize,
    chunk_index: usize,
) -> (usize, usize) {
    let base = count / chunk_count;
    let remainder = count % chunk_count;
    let start = chunk_index * base + chunk_index.min(remainder);
    let len = base + usize::from(chunk_index < remainder);
    (start, start + len)
}

#[cfg(test)]
mod tests {
    use super::{indexed_chunk_bounds, indexed_chunk_count};

    #[test]
    fn assigns_small_domains_across_available_lanes() {
        assert_eq!(indexed_chunk_count(9, 8), 9);
        assert_eq!(indexed_chunk_count(2, 8), 2);
        assert_eq!(indexed_chunk_count(1, 8), 1);
        assert_eq!(indexed_chunk_count(0, 8), 0);
    }

    #[test]
    fn caps_large_domains_at_workers_plus_caller() {
        assert_eq!(indexed_chunk_count(1_000_000, 8), 9);
    }

    #[test]
    fn single_worker_uses_worker_plus_caller() {
        assert_eq!(indexed_chunk_count(1024, 1), 2);
        assert_eq!(indexed_chunk_count(2, 1), 2);
    }

    #[test]
    fn balances_remainder_across_every_chunk() {
        let bounds: Vec<_> = (0..9)
            .map(|chunk_index| indexed_chunk_bounds(10, 9, chunk_index))
            .collect();
        assert_eq!(
            bounds,
            vec![
                (0, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10)
            ]
        );
    }
}
