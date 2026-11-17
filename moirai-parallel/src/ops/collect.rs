use melinoe::{MelinoeCell, region::WriterShard};
use moirai_executor::{SyncTask, global};

use super::{enumerate_mut_with, worker_chunk_size};
use crate::policy::{ExecutionPolicy, Parallel};

/// Map each element of `data` with `f`, collecting into a `Vec<R>` in order,
/// scheduled by policy `P`.
pub fn map_collect_with<P, T, R, F>(data: &[T], f: F) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Send + Sync,
{
    let n = data.len();
    if !P::parallelize(n) {
        return data.iter().map(f).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: capacity is `n`; every slot is written exactly once below before
    // being read, and `MaybeUninit` makes `set_len` sound without initialization.
    unsafe {
        out.set_len(n);
    }
    enumerate_mut_with::<Parallel, _, _>(&mut out, |i, slot| {
        slot.write(f(&data[i]));
    });
    // SAFETY: every slot initialized above; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), n, out.capacity()) }
}

/// Parallel map over the index domain `0..len`, collecting into a `Vec<R>` in
/// order, scheduled by policy `P`.
///
/// `map(i)` produces the element at index `i`. Use this for index-aligned maps
/// over multiple slices that [`map_collect_with`] cannot express — e.g. an
/// elementwise product `map_collect_index_with::<Adaptive>(n, |i| a[i] * b[i])`.
pub fn map_collect_index_with<P, R, Map>(len: usize, map: Map) -> Vec<R>
where
    P: ExecutionPolicy,
    R: Send,
    Map: Fn(usize) -> R + Send + Sync,
{
    if !P::parallelize(len) {
        return (0..len).map(map).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(len);
    // SAFETY: capacity is `len`; every slot is written exactly once below.
    unsafe {
        out.set_len(len);
    }
    enumerate_mut_with::<Parallel, _, _>(&mut out, |i, slot| {
        slot.write(map(i));
    });
    // SAFETY: every slot initialized; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), len, out.capacity()) }
}

/// Map each element of `data` in place with `f(index, &mut element)`, collecting
/// each returned value into a `Vec<R>` in order, scheduled by policy `P`.
///
/// The synchronous equivalent of rayon's
/// `data.par_iter_mut().enumerate().map(f).collect()`: each element is mutated
/// and produces a result. Use for parallel solve-in-place-and-collect loops.
pub fn map_collect_mut_with<P, T, R, F>(data: &mut [T], f: F) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Send,
    R: Send,
    F: Fn(usize, &mut T) -> R + Send + Sync,
{
    let n = data.len();
    if !P::parallelize(n) {
        return data.iter_mut().enumerate().map(|(i, x)| f(i, x)).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: capacity is `n`; every slot is written exactly once below.
    unsafe {
        out.set_len(n);
    }
    let chunk_size = worker_chunk_size(n);
    let data_partitions =
        WriterShard::new(MelinoeCell::from_mut_slice(data)).par_chunks(chunk_size);
    let out_partitions =
        WriterShard::new(MelinoeCell::from_mut_slice(out.as_mut_slice())).par_chunks(chunk_size);
    let tasks = data_partitions.len();
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |c| {
            // Partition `c` covers the elements `[c * chunk_size, …)` in both
            // regions; the two regions have equal length, so their partition
            // counts agree.
            let start = c * chunk_size;
            // SAFETY: each index in `0..tasks` is visited exactly once, for both
            // the input element and the output slot at `start + offset`. The two
            // regions are distinct borrows, and distinct partitions are disjoint
            // ranges in each, so neither the element nor the slot at an absolute
            // index aliases another task's.
            let mut data_shard = unsafe { data_partitions.get_unchecked_chunk(c) };
            let mut out_shard = unsafe { out_partitions.get_unchecked_chunk(c) };
            for (offset, (element, slot)) in
                data_shard.iter_mut().zip(out_shard.iter_mut()).enumerate()
            {
                slot.write(f(start + offset, element));
            }
        })
        .expect("moirai global executor: map_collect_mut_with");
    // SAFETY: every slot initialized; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), n, out.capacity()) }
}
