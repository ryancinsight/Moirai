use std::num::NonZeroUsize;

use melinoe::{MelinoeCell, region::WriterShard};
use moirai_core::memory::MapOutput;
use moirai_executor::{SyncTask, global};

use super::worker_chunk_size;
use crate::policy::ExecutionPolicy;

/// Map each element of `data` with `f`, collecting into a `Vec<R>` in order,
/// scheduled by policy `P`.
pub fn map_collect_with<P, T, R, F>(data: &[T], f: F) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Send + Sync,
{
    map_collect_index_with::<P, _, _>(data.len(), |index| f(&data[index]))
}

/// Map over `0..len`, collecting results in index order under policy `P`.
///
/// Use indexed mapping to access the same logical position in multiple inputs.
pub fn map_collect_index_with<P, R, Map>(len: usize, map: Map) -> Vec<R>
where
    P: ExecutionPolicy,
    R: Send,
    Map: Fn(usize) -> R + Send + Sync,
{
    if !P::parallelize(len) || len == 0 {
        return (0..len).map(map).collect();
    }
    let chunk_size = NonZeroUsize::new(worker_chunk_size(len))
        .expect("invariant: each worker chunk holds at least one element");
    let mut output = MapOutput::new(len, chunk_size);
    let count = output.chunk_count();
    let chunks = output.chunks();
    global()
        .for_each_indexed::<SyncTask, _>(count, |index| {
            let range = chunks
                .range(index)
                .expect("invariant: the executor visits only configured chunks");
            // SAFETY: each scheduled index owns its output chunk until the
            // joined fan-out returns.
            unsafe { chunks.write_chunk(index, range.map(&map)) };
        })
        .expect("invariant: indexed map collection completed");
    output.into_vec()
}

/// Map each element in place with `f(index, &mut element)`, collecting the
/// returned values in order under policy `P`.
pub fn map_collect_mut_with<P, T, R, F>(data: &mut [T], f: F) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Send,
    R: Send,
    F: Fn(usize, &mut T) -> R + Send + Sync,
{
    let len = data.len();
    if !P::parallelize(len) || len == 0 {
        return data.iter_mut().enumerate().map(|(i, x)| f(i, x)).collect();
    }
    let chunk_size = NonZeroUsize::new(worker_chunk_size(len))
        .expect("invariant: each worker chunk holds at least one element");
    let partitions =
        WriterShard::new(MelinoeCell::from_mut_slice(data)).par_chunks(chunk_size.get());
    let mut output = MapOutput::new(len, chunk_size);
    let chunks = output.chunks();
    global()
        .for_each_indexed::<SyncTask, _>(partitions.len(), |index| {
            let start = index * chunk_size.get();
            // SAFETY: each scheduled index identifies a distinct input chunk.
            let mut shard = unsafe { partitions.get_unchecked_chunk(index) };
            let values = shard
                .iter_mut()
                .enumerate()
                .map(|(offset, item)| f(start + offset, item));
            // SAFETY: the output has the input's layout, and each index is
            // visited once in the joined fan-out.
            unsafe { chunks.write_chunk(index, values) };
        })
        .expect("invariant: mutable map collection completed");
    output.into_vec()
}

#[cfg(test)]
mod tests;
