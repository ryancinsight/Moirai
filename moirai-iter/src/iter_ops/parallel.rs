//! Scoped chunked iterator execution.
//!
//! `ParallelIter` owns the input vector once and lends immutable chunks to
//! scoped worker threads. The invariant is `owner(Vec<T>) XOR borrowed chunks`:
//! no worker owns or refcounts the vector, and all borrows end before the
//! vector is dropped.

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
