//! Scoped chunked iterator execution.
//!
//! `ParallelIter` owns the input vector once and lends immutable chunks to
//! scheduler lanes. The invariant is `owner(Vec<T>) XOR borrowed chunks`:
//! no worker owns or refcounts the vector, and all borrows end before the
//! vector is dropped.
//!
//! # Fan-out safety
//!
//! Mapping borrows the source and closure until the scheduler joins every
//! job. Each job owns one output range. The chunk guard drops a partial
//! prefix on panic, and the result owner drops completed peers.
//!
//! Reduction uses SendPtr to give each job one distinct result slot. All
//! pointers remain valid until the joined fan-out returns. A sequential
//! replay is possible only after the scheduler reports that no job ran.
//!
//! # Dispatch threshold
//!
//! Work is only chunked out when `len > chunk_size` and the chunk itself clears
//! `DEFAULT_RING_BUFFER_CAPACITY`; anything smaller runs sequentially, where
//! none of the above applies. `chunk_size` is derived from the parallelism of
//! the host, so whether a given input takes the parallel path is machine
//! dependent — tests that need to cover it construct the iterator with an
//! explicit chunk size rather than relying on the core count.

use std::num::NonZeroUsize;

use moirai_core::memory::MapOutput;

use crate::base::SendPtr;

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

        let chunk_size = NonZeroUsize::new(chunk_size)
            .expect("invariant: selected iterator chunks are nonempty");
        let mut output = MapOutput::new(data.len(), chunk_size);
        let num_chunks = output.chunk_count();
        let chunks = output.chunks();
        let map_chunk = |index| {
            let range = chunks
                .range(index)
                .expect("invariant: the executor visits only configured chunks");
            let values = data[range].iter().map(&f);
            // SAFETY: joined indexed fan-out gives each index one writer.
            unsafe { chunks.write_chunk(index, values) };
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
    let worker_count = crate::base::process_parallelism();

    chunk_size_for_lanes(len, worker_count)
}

const fn chunk_size_for_lanes(len: usize, lane_count: usize) -> usize {
    let lanes = if lane_count == 0 { 1 } else { lane_count };
    let size = len.div_ceil(lanes);
    if size == 0 { 1 } else { size }
}

#[inline]
fn should_execute_scoped(len: usize, chunk_size: usize) -> bool {
    len > chunk_size && chunk_size > DEFAULT_RING_BUFFER_CAPACITY
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
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
    fn chunk_size_normalizes_lanes_and_covers_the_domain() {
        assert_eq!(chunk_size_for_lanes(0, 0), 1);
        assert_eq!(chunk_size_for_lanes(1, 0), 1);
        assert_eq!(chunk_size_for_lanes(10, 3), 4);
        assert_eq!(chunk_size_for_lanes(10, 32), 1);
    }

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
        let message = crate::test_support::panic_message(payload.as_ref());
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
