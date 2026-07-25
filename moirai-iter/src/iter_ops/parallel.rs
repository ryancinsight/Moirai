//! Scoped chunked iterator execution.
//!
//! `ParallelIter` owns the input vector once and lends immutable chunks to
//! scoped worker threads. The invariant is `owner(Vec<T>) XOR borrowed chunks`:
//! no worker owns or refcounts the vector, and all borrows end before the
//! vector is dropped.
//!
//! # Fan-out safety
//!
//! Both operations hand workers three raw pointers wrapped in `SendPtr` — the
//! chunk, the caller's closure, and the output slot — because the borrow checker
//! cannot see that the chunks partition the input. What makes that sound:
//!
//! - **Disjoint writes.** Worker `idx` writes only `results[idx]`, and the
//!   chunks partition the input, so no two workers touch the same output slot or
//!   read overlapping data.
//! - **Shared reads only.** The closure and the chunk are reached as `&F` and
//!   `&[T]`; nothing is written through them. The `F: Sync` and `T: Sync` bounds
//!   are what let several workers hold those references at once. The
//!   `*const _ as *mut _` casts are an artifact of erasing the type through
//!   `SendPtr`, not a claim of unique access.
//! - **Lifetime.** Every pointer refers to a local of this frame — the vector,
//!   the closure, the results — and both dispatch paths block before returning:
//!   the executor call joins internally, and the pool path joins in
//!   `PoolJoinGuard::wait`. No worker outlives what it points at.
//! - **Completion.** `results` starts as `None` per chunk and the tail
//!   `flatten()` silently drops any that stayed `None`, so a chunk that never
//!   ran would quietly shorten a `map` or omit a term from a `reduce`. Neither
//!   join path allows that: `pool_fallback_permitted` panics on a partially
//!   executed fan-out, and `wait` panics unless every task reported completion.
//!
//! Because the outputs are `Option<_>` rather than `MaybeUninit`, a skipped
//! chunk here would be a wrong answer rather than a read of uninitialized
//! memory — which is why the completion guarantee is the load-bearing one.
//!
//! # Dispatch threshold
//!
//! Work is only chunked out when `len > chunk_size` and the chunk itself clears
//! `DEFAULT_RING_BUFFER_CAPACITY`; anything smaller runs sequentially, where
//! none of the above applies. `chunk_size` is derived from the parallelism of
//! the host, so whether a given input takes the parallel path is machine
//! dependent — tests that need to cover it construct the iterator with an
//! explicit chunk size rather than relying on the core count.

use crate::base::{get_shared_thread_pool, PoolJoinGuard, SendPtr};
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

        // Optimized: we use get_shared_thread_pool() instead of std::thread::scope to avoid thread creation overhead.
        let chunks: Vec<_> = data.chunks(chunk_size).collect();
        let num_chunks = chunks.len();

        let mut results: Vec<Option<Vec<U>>> = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            results.push(None);
        }

        let results_ptr = SendPtr(results.as_mut_ptr() as *mut ());
        let f_ptr_send = SendPtr(&f as *const F as *const () as *mut ());

        let run_on_global = moirai_executor::global()
            // SAFETY: `idx < num_chunks` indexes `chunks` and `results` alike,
            // so the chunk is a live borrow of `data` and the write lands on
            // this worker's own slot. `f` outlives the call because the fan-out
            // joins before returning, and is only read, which `F: Sync` allows.
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, |idx| unsafe {
                let chunk = *chunks.get_unchecked(idx);
                let chunk_ptr = chunk.as_ptr();
                let chunk_len = chunk.len();
                let chunk_slice = std::slice::from_raw_parts(chunk_ptr, chunk_len);
                let f_ref = &*(f_ptr_send.as_ptr() as *const F);
                let chunk_result = chunk_slice.iter().map(f_ref).collect::<Vec<_>>();
                *(results_ptr.as_ptr() as *mut Option<Vec<U>>).add(idx) = Some(chunk_result);
            });

        if crate::base::pool_fallback_permitted(&run_on_global) {
            let pool = get_shared_thread_pool();
            let (tx, rx) = std::sync::mpsc::channel();
            let guard = PoolJoinGuard::new(rx, num_chunks);
            for (idx, chunk) in chunks.into_iter().enumerate() {
                let tx = tx.clone();
                let chunk_ptr = SendPtr(chunk.as_ptr() as *mut ());
                let chunk_len = chunk.len();

                pool.execute(move || {
                    // SAFETY: as the executor path — `idx` is this chunk's own
                    // slot, the chunk pointer is a live borrow of `data`, and
                    // `guard.wait()` below keeps this frame alive until every
                    // worker has finished with all three pointers.
                    unsafe {
                        let chunk_slice =
                            std::slice::from_raw_parts(chunk_ptr.as_ptr() as *const T, chunk_len);
                        let f_ref = &*(f_ptr_send.as_ptr() as *const F);
                        let chunk_result = chunk_slice.iter().map(f_ref).collect::<Vec<_>>();
                        *(results_ptr.as_ptr() as *mut Option<Vec<U>>).add(idx) =
                            Some(chunk_result);
                    }
                    let _ = tx.send(());
                });
            }

            drop(tx);
            guard.wait();
        }

        results.into_iter().flatten().flatten().collect()
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

        let run_on_global = moirai_executor::global()
            // SAFETY: as `map`, plus `chunk_identities`, which holds one entry
            // per chunk and is only ever read here — worker `idx` clones its own
            // and never writes through the pointer.
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, |idx| unsafe {
                let chunk = *chunks.get_unchecked(idx);
                let chunk_ptr = chunk.as_ptr();
                let chunk_len = chunk.len();
                let chunk_slice = std::slice::from_raw_parts(chunk_ptr, chunk_len);
                let f_ref = &*(f_ptr_send.as_ptr() as *const F);
                let chunk_identity = (*(identities_ptr.as_ptr() as *const T).add(idx)).clone();
                let chunk_result = chunk_slice.iter().fold(chunk_identity, f_ref);
                *(results_ptr.as_ptr() as *mut Option<T>).add(idx) = Some(chunk_result);
            });

        if crate::base::pool_fallback_permitted(&run_on_global) {
            let pool = get_shared_thread_pool();
            let (tx, rx) = std::sync::mpsc::channel();
            let guard = PoolJoinGuard::new(rx, num_chunks);
            for (idx, chunk) in chunks.into_iter().enumerate() {
                let tx = tx.clone();
                let chunk_ptr = SendPtr(chunk.as_ptr() as *mut ());
                let chunk_len = chunk.len();

                pool.execute(move || {
                    // SAFETY: as the executor path; `guard.wait()` below keeps
                    // this frame alive until every worker is done with the
                    // chunk, closure, identity, and result pointers.
                    unsafe {
                        let chunk_slice =
                            std::slice::from_raw_parts(chunk_ptr.as_ptr() as *const T, chunk_len);
                        let f_ref = &*(f_ptr_send.as_ptr() as *const F);
                        let chunk_identity =
                            (*(identities_ptr.as_ptr() as *const T).add(idx)).clone();
                        let chunk_result = chunk_slice.iter().fold(chunk_identity, f_ref);
                        *(results_ptr.as_ptr() as *mut Option<T>).add(idx) = Some(chunk_result);
                    }
                    let _ = tx.send(());
                });
            }

            drop(tx);
            guard.wait();
        }

        results
            .into_iter()
            .flatten()
            .fold(identity, |accumulator, value| f(accumulator, &value))
    }
}

#[inline]
fn chunk_size(len: usize) -> usize {
    let worker_count = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);

    len.div_ceil(worker_count).max(1)
}

#[inline]
fn should_execute_scoped(len: usize, chunk_size: usize) -> bool {
    len > chunk_size && chunk_size > DEFAULT_RING_BUFFER_CAPACITY
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn parallel_map_matches_the_sequential_result() {
        let data: Vec<u64> = (0..LEN as u64).collect();
        let expected: Vec<u64> = data.iter().map(|value| value * 3).collect();

        let mapped = chunked(data, CHUNK).map(|value| value * 3);

        assert_eq!(
            mapped, expected,
            "the chunked map must preserve every element and its order"
        );
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
    fn parallel_map_covers_a_ragged_final_chunk() {
        // The last chunk is shorter than the rest; a length or offset derived
        // from `chunk_size` rather than the chunk itself would drop or duplicate
        // its elements.
        let data: Vec<u64> = (0..LEN as u64).collect();

        let mapped = chunked(data, CHUNK).map(|value| *value);

        assert_eq!(mapped.len(), LEN);
        assert_eq!(mapped.last(), Some(&(LEN as u64 - 1)));
    }
}
