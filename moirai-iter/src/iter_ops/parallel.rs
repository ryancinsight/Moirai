//! Scoped chunked iterator execution.
//!
//! `ParallelIter` owns the input vector once and lends immutable chunks to
//! scoped worker threads. The invariant is `owner(Vec<T>) XOR borrowed chunks`:
//! no worker owns or refcounts the vector, and all borrows end before the
//! vector is dropped.

use moirai_core::constants::DEFAULT_RING_BUFFER_CAPACITY;

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

        std::thread::scope(|scope| {
            let handles = data
                .chunks(chunk_size)
                .map(|chunk| {
                    let f = &f;
                    scope.spawn(move || chunk.iter().map(f).collect::<Vec<_>>())
                })
                .collect::<Vec<_>>();

            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("parallel map worker panicked"))
                .collect()
        })
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

        let partials = std::thread::scope(|scope| {
            let handles = data
                .chunks(chunk_size)
                .map(|chunk| {
                    let f = &f;
                    let chunk_identity = identity.clone();
                    scope.spawn(move || chunk.iter().fold(chunk_identity, f))
                })
                .collect::<Vec<_>>();

            handles
                .into_iter()
                .map(|handle| handle.join().expect("parallel reduce worker panicked"))
                .collect::<Vec<_>>()
        });

        partials
            .into_iter()
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
