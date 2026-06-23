//! Base trait and enum for execution contexts.

use futures::StreamExt;
use std::sync::Arc;

use super::async_ctx::AsyncContext;
use super::hybrid::HybridContext;
use super::parallel::ParallelContext;
use crate::distributed::DistributedContext;
use crate::multi_system::MultiSystemContext;

const DEFAULT_ASYNC_CONCURRENCY: usize = 1024;

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
    /// Distributed execution across multiple machines
    Distributed(DistributedContext),
    /// Multi-system execution across heterogeneous compute
    MultiSystem(MultiSystemContext),
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
            ExecutionContext::Distributed(_ctx) => {
                // For now, execute locally - real implementation would distribute
                Ok(func())
            }
            ExecutionContext::MultiSystem(_ctx) => {
                // For now, execute locally - real implementation would coordinate systems
                Ok(func())
            }
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
            ExecutionContext::Distributed(_ctx) => {
                // For now, execute sequentially - real implementation would distribute
                Ok(items.into_iter().map(func).collect())
            }
            ExecutionContext::MultiSystem(_ctx) => {
                // For now, execute sequentially - real implementation would coordinate
                Ok(items.into_iter().map(func).collect())
            }
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
        let func = Arc::new(func);
        let results = futures::stream::iter(items)
            .map(|item| {
                let func = Arc::clone(&func);
                async move { func(item).await }
            })
            .buffered(concurrency)
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
        let predicate = Arc::new(predicate);
        let results = futures::stream::iter(items)
            .map(|item| {
                let predicate = Arc::clone(&predicate);
                async move {
                    let keep = predicate(&item).await;
                    (keep, item)
                }
            })
            .buffered(concurrency)
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
        let func = Arc::new(func);
        futures::stream::iter(items)
            .map(|item| {
                let func = Arc::clone(&func);
                async move { func(item).await }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
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
            ExecutionContext::Distributed(_) => "Distributed",
            ExecutionContext::MultiSystem(_) => "MultiSystem",
        }
    }

    fn async_concurrency_limit(&self) -> usize {
        match self {
            ExecutionContext::Async(ctx) => ctx.max_concurrent,
            ExecutionContext::Hybrid(ctx) => ctx.async_context.max_concurrent,
            _ => std::thread::available_parallelism()
                .map(|available| available.get())
                .unwrap_or(DEFAULT_ASYNC_CONCURRENCY)
                .max(1),
        }
    }
}
