//! Parallel partitioning drivers for branded Melinoe cell slices.

use super::DisjointMutPtr;
use melinoe::cell::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_executor::{global, SyncTask};

/// Split `cells` into disjoint shards of `chunk_size` and run `f` on each in parallel.
///
/// Disjoint partitions are processed concurrently on the Moirai thread pool using
/// its work-stealing scheduler, completely bypassing OS-thread spawning overhead.
pub fn par_partition_for_each<'brand, T, F>(
    cells: &mut [MelinoeCell<'brand, T>],
    chunk_size: usize,
    f: F,
) where
    T: Send,
    F: Fn(usize, WriterShard<'_, 'brand, T>) + Send + Sync,
{
    let n = cells.len();
    if n == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = n.div_ceil(chunk_size);
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
    let n = cells.len();
    if n == 0 || chunk_size == 0 {
        return Vec::new();
    }
    let num_chunks = n.div_ceil(chunk_size);
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
            unsafe {
                out_ptr.get_mut(c).write(result);
            }
        })
        .expect("moirai global executor: par_partition_map");
    // SAFETY: every slot initialized above; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), num_chunks, out.capacity()) }
}
