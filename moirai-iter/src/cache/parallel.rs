//! Borrowed cache-map, traversal, and reduction execution.
//!
//! ZeroCopyParallelIter partitions the source into disjoint ranges. Joined
//! scheduler execution keeps the source, closure, and output allocation alive
//! until every worker terminates. Map workers publish a completion endpoint
//! only after initializing their whole range; their local writer drops a
//! partial prefix during unwinding, while the outer owner drops completed peer
//! ranges.

use std::mem;

use super::{prefetch_read_data, CACHE_CHUNK_SIZE, CACHE_LINE_SIZE};
use crate::{
    base::SendPtr,
    parallel::output::{output_chunk_range, ChunkWriter, MapOutput},
};

/// Default scheduler batch capacity used by the cache fan-out gate.
pub(super) const DEFAULT_RING_BUFFER_CAPACITY: usize = 1_024;

/// A zero-copy parallel iterator over a borrowed slice.
pub struct ZeroCopyParallelIter<'a, T> {
    pub(super) data: &'a [T],
    pub(super) chunk_size: usize,
}

impl<'a, T: Sync> ZeroCopyParallelIter<'a, T> {
    /// Create a zero-copy parallel iterator over data.
    ///
    /// Chunk planning uses process-available parallelism fixed at the first
    /// iterator construction. The count affects scheduling only.
    pub fn new(data: &'a [T]) -> Self {
        let chunk_size = zero_copy_chunk_size_for_lanes(
            data.len(),
            mem::size_of::<T>(),
            crate::base::process_parallelism(),
        );
        Self { data, chunk_size }
    }

    /// Apply func to every element, using joined fan-out for large slices.
    ///
    /// # Panics
    ///
    /// Panics when func panics or indexed fan-out partially executes and then
    /// fails.
    pub fn for_each<F>(&self, func: F)
    where
        F: Fn(&T) + Send + Sync,
    {
        if !should_execute_scoped_cache::<T>(self.data.len(), self.chunk_size) {
            self.data.iter().for_each(func);
            return;
        }

        let chunks: Vec<_> = self.data.chunks(self.chunk_size).collect();
        let num_chunks = chunks.len();
        let func_ptr = SendPtr(&func as *const F as *const () as *mut ());

        let visit_chunk = |index: usize| {
            // SAFETY: index is within the joined fan-out domain. The selected
            // chunk is shared-only, func is Sync, and both remain alive until
            // the fan-out joins.
            unsafe {
                let chunk = *chunks.get_unchecked(index);
                let chunk_slice = std::slice::from_raw_parts(chunk.as_ptr(), chunk.len());
                let func_ref = &*(func_ptr.as_ptr() as *const F);
                let cache_line_elements = (CACHE_LINE_SIZE / mem::size_of::<T>().max(1)).max(1);
                for (offset, item) in chunk_slice.iter().enumerate() {
                    if offset % cache_line_elements == 0
                        && offset + cache_line_elements < chunk_slice.len()
                    {
                        prefetch_read_data(
                            chunk_slice
                                .as_ptr()
                                .add(offset + cache_line_elements)
                                .cast(),
                            0,
                        );
                    }
                    func_ref(item);
                }
            }
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &visit_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(visit_chunk);
        }
    }

    /// Map every element into one ordered output vector.
    ///
    /// The parallel path allocates the final output once plus one compact
    /// completion endpoint per chunk. No borrowed chunk descriptor vector is
    /// materialized.
    ///
    /// # Panics
    ///
    /// Panics when func panics or indexed fan-out partially executes and then
    /// fails. Every initialized output is dropped exactly once during unwind.
    pub fn map<F, R>(&self, func: F) -> Vec<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Send,
    {
        if !should_execute_scoped_cache::<T>(self.data.len(), self.chunk_size) {
            return self.data.iter().map(&func).collect();
        }

        let mut output = MapOutput::new(self.data.len(), self.chunk_size);
        let num_chunks = output.chunk_count();
        let data_ptr = SendPtr(self.data.as_ptr().cast_mut().cast::<()>());
        let output_ptr = SendPtr(output.values_ptr().cast::<()>());
        let completed_ptr = SendPtr(output.completed_ptr().cast::<()>());
        let func_ptr = SendPtr(&func as *const F as *const () as *mut ());

        let map_chunk = |chunk_index: usize| {
            // SAFETY: chunk_index identifies one in-bounds input/output range
            // and one completion slot. Ranges are pairwise disjoint, and the
            // joined fan-out keeps every pointer target alive.
            unsafe {
                let chunk_range = output_chunk_range(self.data.len(), self.chunk_size, chunk_index);
                let chunk = std::slice::from_raw_parts(
                    data_ptr.as_ptr().cast::<T>().add(chunk_range.start),
                    chunk_range.len(),
                );
                let func_ref = &*(func_ptr.as_ptr() as *const F);
                let mut writer = ChunkWriter::new(output_ptr.as_ptr().cast(), chunk_range);
                for item in chunk {
                    writer.push(func_ref(item));
                }
                completed_ptr
                    .as_ptr()
                    .cast::<usize>()
                    .add(chunk_index)
                    .write(writer.finish());
            }
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &map_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(map_chunk);
        }

        output.into_vec()
    }

    /// Reduce all elements with associative func, or return None when empty.
    ///
    /// # Panics
    ///
    /// Panics when func panics or indexed fan-out partially executes and then
    /// fails.
    pub fn reduce<F>(&self, func: F) -> Option<T>
    where
        F: Fn(&T, &T) -> T + Send + Sync,
        T: Clone + Send,
    {
        if self.data.is_empty() {
            return None;
        }
        if self.data.len() == 1 {
            return Some(self.data[0].clone());
        }
        if !should_execute_scoped_cache::<T>(self.data.len(), self.chunk_size) {
            return self
                .data
                .iter()
                .cloned()
                .reduce(|left, right| func(&left, &right));
        }

        let chunks: Vec<_> = self.data.chunks(self.chunk_size).collect();
        let num_chunks = chunks.len();
        let mut results = (0..num_chunks).map(|_| None).collect::<Vec<_>>();
        let results_ptr = SendPtr(results.as_mut_ptr().cast::<()>());
        let func_ptr = SendPtr(&func as *const F as *const () as *mut ());

        let reduce_chunk = |index: usize| {
            // SAFETY: index selects one shared input chunk and one exclusive
            // result slot. The closure and both buffers outlive the joined
            // fan-out.
            unsafe {
                let chunk = *chunks.get_unchecked(index);
                let chunk_slice = std::slice::from_raw_parts(chunk.as_ptr(), chunk.len());
                let func_ref = &*(func_ptr.as_ptr() as *const F);
                let chunk_result = chunk_slice
                    .iter()
                    .cloned()
                    .reduce(|left, right| func_ref(&left, &right));
                results_ptr
                    .as_ptr()
                    .cast::<Option<T>>()
                    .add(index)
                    .write(chunk_result);
            }
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &reduce_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(reduce_chunk);
        }

        let mut current_results: Vec<T> = results.into_iter().flatten().collect();
        while current_results.len() > 1 {
            current_results = reduce_owned_pairs(current_results, &func);
        }
        current_results.into_iter().next()
    }
}

pub(super) fn reduce_owned_pairs<T, F>(items: Vec<T>, func: &F) -> Vec<T>
where
    F: Fn(&T, &T) -> T,
{
    let mut source = items.into_iter();
    let mut reduced = Vec::with_capacity(source.len().div_ceil(2));
    while let Some(left) = source.next() {
        match source.next() {
            Some(right) => reduced.push(func(&left, &right)),
            None => reduced.push(left),
        }
    }
    reduced
}

#[inline]
pub(super) fn should_execute_scoped_cache<T>(len: usize, chunk_size: usize) -> bool {
    let cache_chunk_items = (CACHE_CHUNK_SIZE / mem::size_of::<T>().max(1)).max(1);
    let scoped_item_floor = cache_chunk_items.saturating_mul(DEFAULT_RING_BUFFER_CAPACITY);
    len > chunk_size && len > scoped_item_floor
}

pub(super) const fn zero_copy_chunk_size_for_lanes(
    len: usize,
    element_size: usize,
    lane_count: usize,
) -> usize {
    let lanes = if lane_count == 0 { 1 } else { lane_count };
    let width = if element_size == 0 { 1 } else { element_size };
    let elements_per_cache_chunk = CACHE_CHUNK_SIZE / width;
    let fair_share = len / lanes;
    let chunk_size = if fair_share > elements_per_cache_chunk {
        fair_share
    } else {
        elements_per_cache_chunk
    };
    if chunk_size == 0 {
        1
    } else {
        chunk_size
    }
}
