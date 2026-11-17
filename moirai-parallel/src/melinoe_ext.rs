//! Parallel partitioning drivers for branded Melinoe cell slices.
//!
//! Each driver dispatches through an [`ExecutionPolicy`] rather than assuming
//! parallel is always right. The policy is a zero-sized type parameter, so the
//! decision is made once at compile time for [`crate::Sequential`]/[`Parallel`],
//! and costs one inlined comparison for [`crate::Adaptive`].
//!
//! The unprefixed `par_partition_*` functions keep the historical
//! always-parallel behaviour ([`Parallel`]) so existing callers are unaffected;
//! the `*_with_policy` variants let a caller state its body weight, and
//! [`crate::Adaptive`] is the right choice for a caller that knows neither.
//!
//! A pool-backed call refreshes Moirai's Melinoe registration before dispatch.
//! Applications that call Melinoe's `partition_*` functions directly should
//! call [`moirai_executor::initialize`] during startup first.

use std::{iter, num::NonZeroUsize};

use super::policy::{ExecutionPolicy, Parallel};
use melinoe::cell::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_core::memory::MapOutput;
use moirai_executor::{SyncTask, global, initialize};

/// Split `cells` into disjoint shards of `chunk_size` and run `f` on each in parallel.
///
/// Disjoint partitions are processed concurrently on the Moirai thread pool using
/// its work-stealing scheduler, completely bypassing OS-thread spawning overhead.
///
/// This is [`par_partition_for_each_with_policy`] under [`Parallel`]: the pool is
/// entered for any non-empty region. Prefer the policy form when the per-element
/// body is cheap, since small regions lose to the dispatch cost — see
/// [`ADAPTIVE_PARALLEL_THRESHOLD`](super::policy::ADAPTIVE_PARALLEL_THRESHOLD).
pub fn par_partition_for_each<'brand, T, F>(
    cells: &mut [MelinoeCell<'brand, T>],
    chunk_size: usize,
    f: F,
) where
    T: Send,
    F: Fn(usize, WriterShard<'_, 'brand, T>) + Send + Sync,
{
    par_partition_for_each_with_policy::<Parallel, T, F>(cells, chunk_size, f);
}

/// Split `cells` into disjoint shards of `chunk_size` and run `f` on each,
/// parallelizing only when `P` permits it for this region size.
///
/// `P` is a zero-sized [`ExecutionPolicy`] marker. Select [`Parallel`] to always
/// use the pool, [`crate::Sequential`] to never use it, or [`crate::Adaptive`] to let the
/// element count decide. The shard boundaries are identical in every case;
/// callback execution order differs when work is distributed across workers.
pub fn par_partition_for_each_with_policy<'brand, P, T, F>(
    cells: &mut [MelinoeCell<'brand, T>],
    chunk_size: usize,
    f: F,
) where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(usize, WriterShard<'_, 'brand, T>) + Send + Sync,
{
    let n = cells.len();
    if n == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = n.div_ceil(chunk_size);

    // The policy decides on the *work size*, not the shard count: a caller
    // weighing "is this worth a pool dispatch" is really asking about total
    // elements, which is what the policy sees. Running the shards inline
    // preserves the parallel path's shard boundaries.
    if !P::parallelize(n) {
        // `chunks_mut` yields the same disjoint `chunk_size` windows, safely.
        for (c, chunk) in cells.chunks_mut(chunk_size).enumerate() {
            f(c * chunk_size, WriterShard::new(chunk));
        }
        return;
    }

    // Refresh the bridge before entering Melinoe. This keeps the wrapper safe
    // after a test or integration has cleared Melinoe's process-global slot.
    initialize();
    let partitions = WriterShard::new(cells).par_chunks(chunk_size);
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            // SAFETY: `for_each_indexed(num_chunks, _)` visits each chunk index
            // exactly once — the contract `get_unchecked_chunk` documents — and
            // distinct indices name disjoint element ranges, so no two tasks
            // alias.
            let shard = unsafe { partitions.get_unchecked_chunk(c) };
            f(c * chunk_size, shard);
        })
        .expect("moirai global executor: par_partition_for_each");
}

/// Split `cells` into disjoint shards of `chunk_size`, run `f` on each in parallel,
/// and collect the per-shard results into a `Vec<R>` in partition order.
///
/// This is [`par_partition_map_with_policy`] under [`Parallel`]; see
/// [`par_partition_for_each`] for why a caller with a cheap body should prefer
/// the policy form.
pub fn par_partition_map<'brand, T, R, F>(
    cells: &mut [MelinoeCell<'brand, T>],
    chunk_size: usize,
    f: F,
) -> Vec<R>
where
    T: Send,
    R: Send,
    F: Fn(usize, WriterShard<'_, 'brand, T>) -> R + Send + Sync,
{
    par_partition_map_with_policy::<Parallel, T, R, F>(cells, chunk_size, f)
}

/// Split `cells` into disjoint shards of `chunk_size`, run `f` on each, and
/// collect the per-shard results in partition order — parallelizing only when
/// `P` permits it for this region size.
///
/// Result slots are always in shard order regardless of whether the pool ran
/// them. Callback execution order may differ between policies.
pub fn par_partition_map_with_policy<'brand, P, T, R, F>(
    cells: &mut [MelinoeCell<'brand, T>],
    chunk_size: usize,
    f: F,
) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Send,
    R: Send,
    F: Fn(usize, WriterShard<'_, 'brand, T>) -> R + Send + Sync,
{
    let n = cells.len();
    if n == 0 || chunk_size == 0 {
        return Vec::new();
    }
    let num_chunks = n.div_ceil(chunk_size);

    // Sequential path: build results in the same shard-slot order as the pool.
    if !P::parallelize(n) {
        let mut results = Vec::with_capacity(num_chunks);
        for (c, chunk) in cells.chunks_mut(chunk_size).enumerate() {
            results.push(f(c * chunk_size, WriterShard::new(chunk)));
        }
        return results;
    }

    // Refresh the bridge before entering Melinoe. This keeps the wrapper safe
    // after a test or integration has cleared Melinoe's process-global slot.
    initialize();
    let mut output = MapOutput::new(num_chunks, NonZeroUsize::MIN);
    let chunks = output.chunks();
    let cells_partitions = WriterShard::new(cells).par_chunks(chunk_size);
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, |index| {
            // SAFETY: each indexed job owns a disjoint input shard.
            let shard = unsafe { cells_partitions.get_unchecked_chunk(index) };
            let value = iter::once_with(|| f(index * chunk_size, shard));
            // SAFETY: each input shard owns one distinct output slot, and
            // every indexed job joins before the parent takes the output.
            unsafe { chunks.write_chunk(index, value) };
        })
        .expect("invariant: partition map collection completed");
    output.into_vec()
}
