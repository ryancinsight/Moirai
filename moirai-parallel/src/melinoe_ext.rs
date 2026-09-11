//! Parallel partitioning drivers for branded Melinoe cell slices.
//!
//! Each driver dispatches through an [`ExecutionPolicy`] rather than assuming
//! parallel is always right. The policy is a zero-sized type parameter, so the
//! decision is made once at compile time for [`Sequential`]/[`Parallel`], and
//! costs one inlined comparison for [`Adaptive`].
//!
//! The unprefixed `par_partition_*` functions keep the historical
//! always-parallel behaviour ([`Parallel`]) so existing callers are unaffected;
//! the `*_with_policy` variants let a caller state its body weight, and
//! [`Adaptive`] is the right choice for a caller that knows neither.

use super::DisjointMutPtr;
use super::policy::{ExecutionPolicy, Parallel};
use melinoe::cell::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_executor::{SyncTask, global};

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
/// use the pool, [`Sequential`] to never use it, or [`Adaptive`] to let the
/// element count decide. The shard *tiling* is identical in every case, so the
/// closure observes the same `(start, shard)` sequence whether or not the work
/// is distributed across workers.
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
    // elements, which is what the policy sees. Running the shards inline here
    // preserves tiling exactly, so the observable result is policy-independent.
    if !P::parallelize(n) {
        let base = cells.as_mut_ptr();
        for c in 0..num_chunks {
            let start = c * chunk_size;
            if start >= n {
                break;
            }
            let end = (start + chunk_size).min(n);
            // SAFETY: shards for distinct `c` are pairwise disjoint within the
            // slice; sequential iteration visits each exactly once.
            let chunk_ref =
                unsafe { core::slice::from_raw_parts_mut(base.add(start), end - start) };
            f(start, WriterShard::new(chunk_ref));
        }
        return;
    }

    let base = DisjointMutPtr(cells.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            let start = c * chunk_size;
            if start >= n {
                return;
            }
            let end = (start + chunk_size).min(n);
            // SAFETY: chunks [start, end) for distinct c are pairwise disjoint
            // within the slice, and each is visited exactly once.
            let chunk_ref =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            let shard = WriterShard::new(chunk_ref);
            f(start, shard);
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
/// Results are always in shard order regardless of whether the pool ran them,
/// so the policy is not observable in the returned `Vec<R>`.
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

    // Sequential path: build the results in the same shard order the pool path
    // produces, so callers cannot tell the two apart from the output.
    if !P::parallelize(n) {
        let base = cells.as_mut_ptr();
        let mut results = Vec::with_capacity(num_chunks);
        for c in 0..num_chunks {
            let start = c * chunk_size;
            if start >= n {
                break;
            }
            let end = (start + chunk_size).min(n);
            // SAFETY: shards for distinct `c` are pairwise disjoint; visited once.
            let chunk_ref =
                unsafe { core::slice::from_raw_parts_mut(base.add(start), end - start) };
            results.push(f(start, WriterShard::new(chunk_ref)));
        }
        return results;
    }

    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(num_chunks);
    // SAFETY: capacity is `num_chunks`; every slot is written exactly once below.
    unsafe {
        out.set_len(num_chunks);
    }
    let cells_ptr = DisjointMutPtr(cells.as_mut_ptr());
    let out_ptr = DisjointMutPtr(out.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            let start = c * chunk_size;
            if start >= n {
                return;
            }
            let end = (start + chunk_size).min(n);
            // SAFETY: chunks [start, end) for distinct c are pairwise disjoint.
            let chunk_ref = unsafe {
                core::slice::from_raw_parts_mut(cells_ptr.base().add(start), end - start)
            };
            let shard = WriterShard::new(chunk_ref);
            let result = f(start, shard);
            // SAFETY: chunk index c is visited exactly once by the indexed
            // schedule, writing an uninitialized-but-reserved slot once.
            unsafe {
                out_ptr.get_mut(c).write(result);
            }
        })
        .expect("moirai global executor: par_partition_map");
    // SAFETY: every slot initialized above; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), num_chunks, out.capacity()) }
}
