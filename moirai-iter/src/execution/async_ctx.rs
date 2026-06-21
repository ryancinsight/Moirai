//! Async execution context.

use super::base::ExecutionBase;
use super::hybrid::owned_chunks;

/// Async execution context for I/O-bound work
#[derive(Clone)]
pub struct AsyncContext {
    pub(super) batch_size: usize,
    pub(super) max_concurrent: usize,
}

impl Default for AsyncContext {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncContext {
    /// Create a new async context
    pub fn new() -> Self {
        Self {
            batch_size: 100,
            max_concurrent: 1000,
        }
    }

    /// Create with specific batch size
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self {
            batch_size,
            max_concurrent: 1000,
        }
    }

    /// Set maximum concurrent operations
    pub fn with_max_concurrent(mut self, max_concurrent: usize) -> Self {
        self.max_concurrent = max_concurrent;
        self
    }
}

impl AsyncContext {
    /// Execute an iterator operation with async processing
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
        let mut results = Vec::with_capacity(items.len());

        for batch in owned_chunks(items, self.batch_size) {
            for item in batch {
                let result = func(item);
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Execute a closure with the context
    pub fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // In async context, execute immediately for now
        // Real implementation would use async runtime
        Ok(func())
    }
}

impl ExecutionBase for AsyncContext {
    fn context_type(&self) -> &'static str {
        "Async"
    }
}
