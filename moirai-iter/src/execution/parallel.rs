//! Parallel execution context.

use std::sync::Arc;
use std::fmt::Debug;
use crate::base::ThreadPool;
use super::base::ExecutionBase;
use super::hybrid::owned_chunks;

/// Parallel execution context for CPU-bound work
#[derive(Clone)]
pub struct ParallelContext {
    thread_pool: Arc<ThreadPool>,
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
    /// Create a new parallel context with default thread pool
    pub fn new() -> Self {
        let thread_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            thread_pool: Arc::new(ThreadPool::new(thread_count)),
            chunk_size: 1000,
        }
    }

    /// Create with specific chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        let thread_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            thread_pool: Arc::new(ThreadPool::new(thread_count)),
            chunk_size,
        }
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
        let mut results = Vec::with_capacity(item_count);
        let (tx, rx) = std::sync::mpsc::channel();
        let func = Arc::new(func);

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            let tx = tx.clone();
            let func = Arc::clone(&func);

            self.thread_pool.execute(move || {
                let chunk_results: Vec<R> = chunk.into_iter().map(|item| func(item)).collect();
                tx.send((chunk_idx, chunk_results)).unwrap();
            });
        }
        drop(tx); // Close the sender

        // Collect results in order
        let mut ordered_results: Vec<(usize, Vec<R>)> = Vec::new();
        for (chunk_idx, chunk_results) in rx {
            ordered_results.push((chunk_idx, chunk_results));
        }

        // Sort by chunk index to maintain order
        ordered_results.sort_by_key(|(idx, _)| *idx);

        for (_, chunk_results) in ordered_results {
            results.extend(chunk_results);
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
