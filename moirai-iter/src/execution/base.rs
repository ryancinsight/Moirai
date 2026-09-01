//! Base trait and enum for execution contexts.

use futures::StreamExt;

use crate::stream::{retained_buffered, retained_unordered};

use super::async_ctx::AsyncContext;
use super::hybrid::HybridContext;
use super::parallel::ParallelContext;

/// Base trait for all execution contexts
pub trait ExecutionBase: Send + Sync {
    /// Get context type name for debugging
    fn context_type(&self) -> &'static str;

    /// Check if the context is ready for execution
    fn is_ready(&self) -> bool {
        true
    }
}

/// Concrete execution context enum that wraps different strategy implementations
/// This approach ensures type safety while avoiding dyn-compatibility issues
#[derive(Clone)]
pub enum ExecutionContext {
    /// Parallel execution for CPU-bound work
    Parallel(ParallelContext),
    /// Async execution for I/O-bound work  
    Async(AsyncContext),
    /// Hybrid execution that adapts between strategies
    Hybrid(HybridContext),
}

impl ExecutionContext {
    /// Execute a function once with the appropriate context
    pub fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match self {
            ExecutionContext::Parallel(ctx) => ctx.execute(func),
            ExecutionContext::Async(ctx) => ctx.execute(func),
            ExecutionContext::Hybrid(ctx) => ctx.execute(func),
        }
    }

    /// Execute an iterator operation with proper type erasure
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
        match self {
            ExecutionContext::Parallel(ctx) => ctx.execute_iter(items, func),
            ExecutionContext::Async(ctx) => ctx.execute_iter(items, func),
            ExecutionContext::Hybrid(ctx) => ctx.execute_iter(items, func),
        }
    }

    /// Execute async iterator operations
    pub async fn execute_async_iter<T, F, Fut, R>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let concurrency = self.async_concurrency_limit();
        let results = retained_buffered(futures::stream::iter(items).map(func), concurrency)
            .collect::<Vec<_>>()
            .await;
        Ok(results)
    }

    /// Execute async filter operations
    pub async fn execute_async_filter<T, F, Fut>(
        &self,
        items: Vec<T>,
        predicate: F,
    ) -> Result<Vec<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(&T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = bool> + Send + 'static,
    {
        let concurrency = self.async_concurrency_limit();
        let predicate = &predicate;
        let futures = futures::stream::iter(items).map(|item| async move {
            let keep = predicate(&item).await;
            (keep, item)
        });
        let results = retained_buffered(futures, concurrency)
            .filter_map(|(keep, item)| async move { keep.then_some(item) })
            .collect::<Vec<_>>()
            .await;
        Ok(results)
    }

    /// Execute async for_each operations
    pub async fn execute_async_for_each<T, F, Fut>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let concurrency = self.async_concurrency_limit();
        retained_unordered(futures::stream::iter(items).map(func), concurrency)
            .for_each(|()| async {})
            .await;
        Ok(())
    }

    /// Execute parallel reduce operations
    pub async fn execute_reduce<T, F>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Result<Option<T>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        Ok(items.into_iter().reduce(func))
    }

    /// Get context type name
    pub fn context_type(&self) -> &'static str {
        match self {
            ExecutionContext::Parallel(ctx) => ctx.context_type(),
            ExecutionContext::Async(ctx) => ctx.context_type(),
            ExecutionContext::Hybrid(ctx) => ctx.context_type(),
        }
    }

    fn async_concurrency_limit(&self) -> usize {
        match self {
            ExecutionContext::Parallel(_) => crate::base::process_parallelism(),
            ExecutionContext::Async(ctx) => ctx.max_concurrent,
            ExecutionContext::Hybrid(ctx) => ctx.async_context.max_concurrent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrency_limits_preserve_context_configuration() {
        let parallel = ExecutionContext::Parallel(ParallelContext::new());
        assert_eq!(
            parallel.async_concurrency_limit(),
            crate::base::process_parallelism()
        );

        let asynchronous = ExecutionContext::Async(AsyncContext::new().with_max_concurrent(7));
        assert_eq!(asynchronous.async_concurrency_limit(), 7);

        let mut hybrid = HybridContext::new();
        hybrid.async_context = AsyncContext::new().with_max_concurrent(11);
        let hybrid = ExecutionContext::Hybrid(hybrid);
        assert_eq!(hybrid.async_concurrency_limit(), 11);
    }
}
