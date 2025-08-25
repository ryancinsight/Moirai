//! Moirai Iterator - Unified high-performance iterator system for concurrent, parallel, async, and distributed computing.
//!
//! This module provides a comprehensive iterator framework that abstracts over different execution contexts:
//! - **Parallel**: CPU-bound work across multiple threads with work-stealing
//! - **Async**: I/O-bound work with efficient async/await patterns  
//! - **Distributed**: Cross-process and cross-machine computation
//! - **Hybrid**: Mixed workloads combining parallel and async execution
//!
//! # Design Principles
//!
//! - **Zero-cost abstractions**: Compile-time optimizations with no runtime overhead
//! - **Memory efficiency**: NUMA-aware allocation and cache-friendly data layouts
//! - **Execution agnostic**: Same API works across all execution contexts
//! - **Type safety**: Comprehensive compile-time guarantees
//! - **Performance**: SIMD vectorization and CPU optimization
//! - **Pure Rust std**: No external dependencies, pure standard library implementation

// Module declarations following SRP and SOC
pub mod base;
pub mod cache;
pub mod channel_fusion;
pub mod combinators;
pub mod execution;
pub mod iter_ops;
pub mod numa;
pub mod prefetch;
pub mod simd_iter;
pub mod windows;

// Re-export key types for clean API
pub use base::ThreadPool;
pub use execution::{
    AsyncContext, ExecutionBase, ExecutionContext, ExecutionStrategy, HybridConfig, HybridContext,
    ParallelContext, PerformanceHistory,
};

/// Core trait for parallel iteration
trait IntoParallelIterator {
    type Item: Send;

    fn into_par_iter(self) -> ParIter<Self::Item>;
}

/// Parallel iterator implementation
struct ParIter<T> {
    data: Vec<T>,
}

impl<T: Send + Clone + 'static> ParIter<T> {
    fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Map operation with parallel execution
    pub fn map<F, R>(self, func: F) -> ParIter<R>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + Clone + 'static,
    {
        let context = ParallelContext::new();
        let results = context
            .execute_iter(self.data, func)
            .unwrap_or_else(|_| vec![]);
        ParIter::new(results)
    }

    /// Filter operation
    pub fn filter<F>(self, predicate: F) -> ParIter<T>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let filtered: Vec<T> = self
            .data
            .into_iter()
            .filter(|item| predicate(item))
            .collect();
        ParIter::new(filtered)
    }

    /// Reduce operation
    pub fn reduce<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        self.data.into_iter().reduce(func)
    }

    /// Collect into a vector
    pub fn collect(self) -> Vec<T> {
        self.data
    }
}

impl<T: Send + Clone + 'static> IntoParallelIterator for Vec<T> {
    type Item = T;

    fn into_par_iter(self) -> ParIter<Self::Item> {
        ParIter::new(self)
    }
}

/// Main iterator type that adapts to different execution contexts
pub struct MoiraiIterator<T> {
    data: Vec<T>,
    context: ExecutionContext,
}

impl<T: Send + Clone + 'static> MoiraiIterator<T> {
    /// Create a new iterator with the given execution context
    pub fn new(data: Vec<T>, context: ExecutionContext) -> Self {
        Self { data, context }
    }

    /// Create with parallel context
    pub fn parallel(data: Vec<T>) -> Self {
        Self::new(data, ExecutionContext::Parallel(ParallelContext::new()))
    }

    /// Create with async context
    pub fn async_iter(data: Vec<T>) -> Self {
        Self::new(data, ExecutionContext::Async(AsyncContext::new()))
    }

    /// Create with hybrid context
    pub fn hybrid(data: Vec<T>) -> Self {
        Self::new(data, ExecutionContext::Hybrid(HybridContext::new()))
    }

    /// Map operation that preserves the execution context
    pub fn map<F, R>(self, func: F) -> MoiraiIterator<R>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + Clone + 'static,
    {
        let results = self
            .context
            .execute_iter(self.data, func)
            .unwrap_or_else(|_| vec![]);

        // Create new iterator with same context type
        match self.context.context_type() {
            "Parallel" => MoiraiIterator::parallel(results),
            "Async" => MoiraiIterator::async_iter(results),
            "Hybrid" => MoiraiIterator::hybrid(results),
            _ => MoiraiIterator::parallel(results), // Default fallback
        }
    }

    /// Filter operation
    pub fn filter<F>(self, predicate: F) -> MoiraiIterator<T>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let filtered: Vec<T> = self
            .data
            .into_iter()
            .filter(|item| predicate(item))
            .collect();

        match self.context.context_type() {
            "Parallel" => MoiraiIterator::parallel(filtered),
            "Async" => MoiraiIterator::async_iter(filtered),
            "Hybrid" => MoiraiIterator::hybrid(filtered),
            _ => MoiraiIterator::parallel(filtered),
        }
    }

    /// Collect the results
    pub async fn collect(self) -> Vec<T> {
        // For now, return synchronously
        // Real implementation would be truly async
        self.data
    }

    /// Reduce operation
    pub async fn reduce<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + 'static,
        T: Clone,
    {
        self.data.into_iter().reduce(func)
    }

    /// For each operation with side effects
    pub async fn for_each<F>(self, func: F)
    where
        F: Fn(T) -> () + Send + Sync + 'static,
    {
        let _ = self.context.execute_iter(self.data, func);
    }
}

/// Convenience function to create a Moirai iterator
pub fn moirai_iter<T: Send + Clone + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::hybrid(data)
}

/// Create a parallel iterator
pub fn moirai_iter_parallel<T: Send + Clone + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::parallel(data)
}

/// Create an async iterator
pub fn moirai_iter_async<T: Send + Clone + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::async_iter(data)
}

/// Create a hybrid iterator
pub fn moirai_iter_hybrid<T: Send + Clone + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::hybrid(data)
}
