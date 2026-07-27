//! Parallel execution context.

use super::base::ExecutionBase;
use super::hybrid::owned_chunks;
use crate::base::SendPtr;
use std::fmt::Debug;

/// Parallel execution context for CPU-bound work
///
/// Work runs on the process-wide scheduler rather than a context-owned pool,
/// so several contexts share one worker set instead of over-subscribing the
/// machine with a thread pool each.
#[derive(Clone)]
pub struct ParallelContext {
    chunk_size: usize,
}

impl Default for ParallelContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for ParallelContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelContext")
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl ParallelContext {
    /// Create a new parallel context with the default chunk size
    pub fn new() -> Self {
        Self { chunk_size: 1000 }
    }

    /// Create with specific chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}

impl ParallelContext {
    /// Execute an iterator operation with parallel processing
    pub fn execute_iter<T, F, R>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let chunk_size = self.chunk_size.max(1);

        if items.len() <= chunk_size {
            return Ok(items.into_iter().map(func).collect());
        }

        let item_count = items.len();
        let chunks = owned_chunks(items, chunk_size);
        let num_chunks = chunks.len();

        // One owned input slot and one owned output slot per chunk. The chunk
        // is taken out and the result written back through the same index, so
        // ordering falls out of the index domain rather than a post-hoc sort of
        // whatever arrived — the previous channel collect ended as soon as the
        // senders dropped, so a panicking chunk silently returned a short `Vec`.
        let mut chunks: Vec<Option<Vec<T>>> = chunks.into_iter().map(Some).collect();
        let mut chunk_results: Vec<Option<Vec<R>>> = (0..num_chunks).map(|_| None).collect();
        let chunks_ptr = SendPtr(chunks.as_mut_ptr());
        let results_ptr = SendPtr(chunk_results.as_mut_ptr());

        // SAFETY: the fan-out visits each index in `0..num_chunks` exactly once,
        // so no two lanes touch the same input or output slot, and both vectors
        // outlive the joined call.
        let map_chunk = |idx: usize| unsafe {
            let chunk = (*chunks_ptr.as_ptr().add(idx))
                .take()
                .expect("invariant: each chunk is claimed by exactly one index");
            let mapped: Vec<R> = chunk.into_iter().map(&func).collect();
            *results_ptr.as_ptr().add(idx) = Some(mapped);
        };

        let run_on_global = moirai_executor::global()
            .for_each_indexed::<moirai_executor::schedule::SyncTask, _>(num_chunks, &map_chunk);

        if crate::base::sequential_fallback_permitted(&run_on_global) {
            (0..num_chunks).for_each(map_chunk);
        }

        let mut results = Vec::with_capacity(item_count);
        for chunk in chunk_results {
            results.extend(chunk.expect("invariant: every chunk index produced a result"));
        }

        Ok(results)
    }

    /// Execute a closure with the context
    pub fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // Execute immediately in parallel context
        Ok(func())
    }
}

impl ExecutionBase for ParallelContext {
    fn context_type(&self) -> &'static str {
        "Parallel"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: usize = 8;
    const ITEMS: usize = CHUNK * 5;

    #[test]
    fn execute_iter_returns_every_item_in_input_order() {
        let context = ParallelContext::with_chunk_size(CHUNK);
        let items: Vec<usize> = (0..ITEMS).collect();

        let doubled = context
            .execute_iter(items.clone(), |item| item * 2)
            .expect("chunked execution must succeed");

        assert_eq!(
            doubled,
            items.iter().map(|item| item * 2).collect::<Vec<_>>(),
            "results must follow input order, not completion order"
        );
    }

    #[test]
    fn execute_iter_propagates_a_chunk_panic_instead_of_truncating() {
        // The previous channel-collect ended when the senders dropped, so a
        // panicking chunk returned a short `Vec` and the caller could not tell.
        // A missing chunk must surface, not shrink the result.
        let context = ParallelContext::with_chunk_size(CHUNK);
        let items: Vec<usize> = (0..ITEMS).collect();

        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(move || {
            context.execute_iter(items, |item| {
                assert_ne!(item, ITEMS - 1, "chunk panic");
                item
            })
        });
        std::panic::set_hook(previous_hook);

        assert!(
            outcome.is_err(),
            "a panicking chunk must reach the caller rather than shorten the result"
        );
    }
}
