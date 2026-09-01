//! Scoped chunked iterator execution.
//!
//! `ParallelIter` owns the input vector once and lends immutable chunks to
//! scheduler lanes. The invariant is `owner(Vec<T>) XOR borrowed chunks`:
//! no worker owns or refcounts the vector, and all borrows end before the
//! vector is dropped.
//!
//! # Fan-out safety
//!
//! Both operations hand workers raw pointers wrapped in `SendPtr` because the
//! borrow checker cannot see that the chunks partition the input and output.
//! What makes that sound:
//!
//! - **Disjoint writes.** Map worker `idx` writes only the output range matching
//!   its input chunk and its own completion slot. Reduce worker `idx` writes
//!   only `results[idx]`. No two workers touch the same output slot.
//! - **Shared reads only.** The closure and the chunk are reached as `&F` and
//!   `&[T]`; nothing is written through them. The `F: Sync` and `T: Sync` bounds
//!   are what let several workers hold those references at once. The
//!   `*const _ as *mut _` casts are an artifact of erasing the type through
//!   `SendPtr`, not a claim of unique access.
//! - **Lifetime.** Every pointer refers to a local of this frame — the vector,
//!   the closure, the results — and the fan-out joins every lane before it
//!   returns. No lane outlives what it points at.
//! - **Completion.** A map worker publishes its range only after initializing
//!   every slot. Its local guard drops an initialized prefix if the mapper
//!   panics; the outer guard drops peer ranges after the fan-out joins. The
//!   final conversion checks contiguous full coverage first. Reduce outputs
//!   remain `None` per chunk until complete. `sequential_fallback_permitted`
//!   panics on partial execution and re-runs the whole domain only when no lane
//!   ran.
//!
//! # Dispatch threshold
//!
//! Work is only chunked out when `len > chunk_size` and the chunk itself clears
//! `DEFAULT_RING_BUFFER_CAPACITY`; anything smaller runs sequentially, where
//! none of the above applies. `chunk_size` is derived from the parallelism of
//! the host, so whether a given input takes the parallel path is machine
//! dependent — tests that need to cover it construct the iterator with an
//! explicit chunk size rather than relying on the core count.

mod map_output;

use crate::base::SendPtr;
use map_output::{ChunkWriter, MapOutput};
/// Default ring buffer capacity (power of 2)
const DEFAULT_RING_BUFFER_CAPACITY: usize = 1024;

/// Parallel iterator with automatic scoped chunking.
pub struct ParallelIter<T> {
    data: Vec<T>,
    chunk_size: usize,
}

impl<T: Send + Sync> ParallelIter<T> {
    /// Create a new parallel iterator.
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Self {
            chunk_size: chunk_size(data.len()),
            data,
        }
    }

    /// Map borrowed input items in chunk order.
    ///
    /// The closure is borrowed by scoped workers, so callers can capture
    /// non-`'static` state without allocating an `Arc` for either the data or
    /// the operation.
    ///
    /// # Panics
    ///
    /// Panics when the mapping closure panics or indexed fan-out reports a
    /// failure after any chunk has executed.
    #[inline]
    pub fn map<F, U>(self, f: F) -> Vec<U>
    where
        F: Fn(&T) -> U + Send + Sync,
        U: Send,
    {
        let data = self.data;
        let chunk_size = self.chunk_size;

        if !should_execute_scoped(data.len(), chunk_size) {
            return data.iter().map(&f).collect();
        }

        let chunks: Vec<_> = data.chunks(chunk_size).collect();
        let num_chunks = chunks.len();

        let mut output = MapOutput::new(data.len(), num_chunks);
        let output_ptr = SendPtr(output.values_ptr().cast::<()>());
        let completed_ptr = SendPtr(output.completed_ptr().cast::<()>());
        let f_ptr_send = SendPtr(&f as *const F as *const () as *mut ());

        let map_chunk = |idx: usize| {
            // SAFETY: `idx < num_chunks` indexes `chunks` and completion slots
            // alike. `chunk_start..chunk_end` is the disjoint output range that
            // corresponds to this input chunk. The fan-out joins before the
            // output, chunks, or shared closure can be dropped.
            unsafe {
                let chunk = *chunks.get_unchecked(idx);
                let chunk_start = idx * chunk_size;
                let chunk_end = chunk_start + chunk.len();
                let chunk_slice = std::slice::from_raw_parts(chunk.as_ptr(), chunk.len());
                let f_ref = &*(f_ptr_send.as_ptr() as *const F);
                let mut writer =
                    ChunkWriter::new(output_ptr.as_ptr().cast(), chunk_start..chunk_end);
                for item in chunk_slice {
                    writer.push(f_ref(item));
                }
                let completed = writer.finish();
                completed_ptr
                    .as_ptr()
                    .cast::<Option<std::ops::Range<usize>>>()
                    .add(idx)
                    .write(Some(completed));
            }
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &map_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(map_chunk);
        }

        output.into_vec()
    }

    /// Reduce borrowed input items through per-chunk partial reductions.
    ///
    /// The identity value is cloned once per active chunk, then partial values
    /// are folded in source-chunk order. Correctness requires `identity` to be
    /// a neutral element for `f`, matching the standard parallel-reduction
    /// contract.
    #[inline]
    pub fn reduce<F>(self, identity: T, f: F) -> T
    where
        F: Fn(T, &T) -> T + Send + Sync,
        T: Clone,
    {
        let data = self.data;
        let chunk_size = self.chunk_size;

        if !should_execute_scoped(data.len(), chunk_size) {
            return data.iter().fold(identity, &f);
        }

        let chunks: Vec<_> = data.chunks(chunk_size).collect();
        let num_chunks = chunks.len();

        let mut results: Vec<Option<T>> = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            results.push(None);
        }

        // Pre-allocate cloned identities to avoid capturing T by value in the static closure
        let mut chunk_identities = vec![identity.clone(); num_chunks];
        let identities_ptr = SendPtr(chunk_identities.as_mut_ptr() as *mut ());

        let results_ptr = SendPtr(results.as_mut_ptr() as *mut ());
        let f_ptr_send = SendPtr(&f as *const F as *const () as *mut ());

        // SAFETY: as `map`, plus `chunk_identities`, which holds one entry per
        // chunk and is only ever read here — lane `idx` clones its own and never
        // writes through the pointer.
        let reduce_chunk = |idx: usize| unsafe {
            let chunk = *chunks.get_unchecked(idx);
            let chunk_ptr = chunk.as_ptr();
            let chunk_len = chunk.len();
            let chunk_slice = std::slice::from_raw_parts(chunk_ptr, chunk_len);
            let f_ref = &*(f_ptr_send.as_ptr() as *const F);
            let chunk_identity = (*(identities_ptr.as_ptr() as *const T).add(idx)).clone();
            let chunk_result = chunk_slice.iter().fold(chunk_identity, f_ref);
            *(results_ptr.as_ptr() as *mut Option<T>).add(idx) = Some(chunk_result);
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &reduce_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(reduce_chunk);
        }

        results
            .into_iter()
            .flatten()
            .fold(identity, |accumulator, value| f(accumulator, &value))
    }
}

#[inline]
fn chunk_size(len: usize) -> usize {
    let worker_count = themis::CpuTopology::detect()
        .map(|topology| topology.logical_processors())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .max(1);

    len.div_ceil(worker_count).max(1)
}

#[inline]
fn should_execute_scoped(len: usize, chunk_size: usize) -> bool {
    len > chunk_size && chunk_size > DEFAULT_RING_BUFFER_CAPACITY
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    /// Build an iterator that is guaranteed to take the chunked path.
    ///
    /// `new` derives the chunk size from the host's parallelism, so on any
    /// machine an input has to be large enough that `len / cores` still clears
    /// `DEFAULT_RING_BUFFER_CAPACITY` before the fan-out runs at all. Setting the
    /// size directly makes the coverage independent of the core count.
    fn chunked<T: Send + Sync>(data: Vec<T>, chunk_size: usize) -> ParallelIter<T> {
        assert!(
            should_execute_scoped(data.len(), chunk_size),
            "the fixture must reach the parallel path, not the sequential fallback"
        );
        ParallelIter { data, chunk_size }
    }

    const CHUNK: usize = DEFAULT_RING_BUFFER_CAPACITY + 1;
    const LEN: usize = CHUNK * 4 + 7; // several chunks plus a short remainder

    #[test]
    fn parallel_map_preserves_boundary_shapes_and_order() {
        for len in [0, 1, CHUNK, CHUNK * 4, LEN] {
            let data: Vec<u64> = (0..len as u64).collect();
            let expected: Vec<u64> = data.iter().map(|value| value * 3).collect();
            let iter = ParallelIter {
                data,
                chunk_size: CHUNK,
            };

            let mapped = iter.map(|value| value * 3);

            assert_eq!(
                mapped, expected,
                "map must preserve every value for logical length {len}"
            );
        }
    }

    #[test]
    fn parallel_reduce_matches_the_sequential_fold() {
        let data: Vec<u64> = (0..LEN as u64).collect();
        let expected: u64 = data.iter().sum();

        let reduced = chunked(data, CHUNK).reduce(0, |sum, value| sum + value);

        assert_eq!(
            reduced, expected,
            "every chunk's partial fold must reach the final result"
        );
    }

    #[test]
    fn parallel_map_moves_non_clone_outputs_and_drops_them_once() {
        struct TrackedOutput {
            value: u64,
            drops: Arc<AtomicUsize>,
        }

        impl Drop for TrackedOutput {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let output_drops = Arc::clone(&drops);
        let mapped = chunked((0..LEN as u64).collect(), CHUNK).map(move |value| TrackedOutput {
            value: *value * 3,
            drops: Arc::clone(&output_drops),
        });

        assert_eq!(mapped.len(), LEN);
        assert!(
            mapped
                .iter()
                .enumerate()
                .all(|(index, output)| output.value == index as u64 * 3),
            "every non-Clone output must move into its ordered final slot"
        );
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(mapped);
        assert_eq!(drops.load(Ordering::Relaxed), LEN);
    }

    #[test]
    fn parallel_map_drops_every_initialized_output_when_mapper_panics() {
        struct TrackedOutput(Arc<AtomicUsize>);

        impl Drop for TrackedOutput {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let created = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let output_created = Arc::clone(&created);
        let output_dropped = Arc::clone(&dropped);

        let result = catch_unwind(AssertUnwindSafe(|| {
            chunked((0..LEN as u64).collect(), CHUNK).map(move |value| {
                assert_ne!(*value, CHUNK as u64 + 3, "mapper panic sentinel");
                output_created.fetch_add(1, Ordering::Relaxed);
                TrackedOutput(Arc::clone(&output_dropped))
            })
        }));

        let payload = match result {
            Err(payload) => payload,
            Ok(mapped) => panic!(
                "invariant: mapper sentinel must panic, but returned {} outputs",
                mapped.len()
            ),
        };
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .expect("invariant: the scheduler propagates a string panic payload");
        assert!(
            message.contains("indexed fan-out failed after partial execution"),
            "unexpected propagated panic: {message}"
        );
        assert_eq!(
            dropped.load(Ordering::Relaxed),
            created.load(Ordering::Relaxed),
            "every output initialized before the panic must be dropped exactly once"
        );
    }

    #[test]
    fn parallel_map_preserves_zero_sized_outputs() {
        let mapped = chunked(vec![1_u8; LEN], CHUNK).map(|_| ());

        assert_eq!(mapped, vec![(); LEN]);
    }
}
