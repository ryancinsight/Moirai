//! Moirai Iterator - Unified high-performance iterator system for concurrent, parallel, async, and distributed computing.
//!
//! This module provides a comprehensive iterator framework that abstracts over different execution contexts:
//! - **Parallel**: CPU-bound work across multiple threads with work-stealing
//! - **Async**: I/O-bound work with efficient async/await patterns  
//! - **Distributed**: Cross-process and cross-machine computation
//! - **Multi-System**: Coordinated processing across multiple machines and GPUs
//! - **Hybrid**: Mixed workloads combining parallel and async execution
//!
//! # Design Principles
//!
//! - **Zero-cost abstractions**: Compile-time optimizations with no runtime overhead
//! - **Memory efficiency**: NUMA-aware allocation and cache-friendly data layouts
//! - **Execution agnostic**: Same API works across all execution contexts
//! - **Type safety**: Comprehensive compile-time guarantees
//! - **Performance**: SIMD vectorization and CPU optimization
//! - **Async compatibility**: Native async/await support throughout
//! - **Multi-system scaling**: Seamless scaling across machines and compute units

// Module declarations following SRP and SOC
pub mod advanced_patterns;
pub mod async_iter;
pub mod base;
pub mod cache;
pub mod channel_fusion;
pub mod combinators;
pub mod distributed;
pub mod execution;
pub mod iter_ops;
pub mod multi_system;
pub mod numa;
pub mod parallel;
pub mod prefetch;
pub mod simd_iter;
pub mod windows;

// Re-export key types for clean API
pub use async_iter::{AsyncIterator, AsyncParallelIterator, IntoAsyncIterator};
pub use base::ThreadPool;
pub use distributed::{DistributedContext, DistributedIterator, NodeConfig};
pub use execution::{
    AsyncContext, ExecutionBase, ExecutionContext, ExecutionStrategy, HybridConfig, HybridContext,
    ParallelContext, PerformanceHistory,
};
pub use multi_system::{MultiSystemContext, MultiSystemIterator, SystemConfig};
pub use parallel::{
    IntoParallelIterator, IntoParallelRefIterator, ParallelExtend, ParallelIterator,
    RangeParIter, VecParIter, VecRefParIter,
};

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

    /// Create with distributed context for multi-machine processing
    pub fn distributed(data: Vec<T>) -> Self {
        Self::new(data, ExecutionContext::Distributed(distributed::DistributedContext::new()))
    }

    /// Create with multi-system context for coordinated compute
    pub fn multi_system(data: Vec<T>) -> Self {
        Self::new(data, ExecutionContext::MultiSystem(multi_system::MultiSystemContext::new()))
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
            "Distributed" => MoiraiIterator::distributed(results),
            "MultiSystem" => MoiraiIterator::multi_system(results),
            _ => MoiraiIterator::parallel(results), // Default fallback
        }
    }

    /// Async map operation for I/O-bound transformations
    pub async fn map_async<F, Fut, R>(self, func: F) -> MoiraiIterator<R>
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = R> + Send + 'static,
        R: Send + Clone + 'static,
    {
        let results = self
            .context
            .execute_async_iter(self.data, func)
            .await
            .unwrap_or_else(|_| vec![]);

        match self.context.context_type() {
            "Parallel" => MoiraiIterator::parallel(results),
            "Async" => MoiraiIterator::async_iter(results),
            "Hybrid" => MoiraiIterator::hybrid(results),
            "Distributed" => MoiraiIterator::distributed(results),
            "MultiSystem" => MoiraiIterator::multi_system(results),
            _ => MoiraiIterator::async_iter(results),
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
            "Distributed" => MoiraiIterator::distributed(filtered),
            "MultiSystem" => MoiraiIterator::multi_system(filtered),
            _ => MoiraiIterator::parallel(filtered),
        }
    }

    /// Async filter operation
    pub async fn filter_async<F, Fut>(self, predicate: F) -> MoiraiIterator<T>
    where
        F: Fn(&T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = bool> + Send + 'static,
        T: Clone,
    {
        let results = self
            .context
            .execute_async_filter(self.data, predicate)
            .await
            .unwrap_or_else(|_| vec![]);

        match self.context.context_type() {
            "Parallel" => MoiraiIterator::parallel(results),
            "Async" => MoiraiIterator::async_iter(results),
            "Hybrid" => MoiraiIterator::hybrid(results),
            "Distributed" => MoiraiIterator::distributed(results),
            "MultiSystem" => MoiraiIterator::multi_system(results),
            _ => MoiraiIterator::async_iter(results),
        }
    }

    /// Collect the results
    pub async fn collect(self) -> Vec<T> {
        // For now, return synchronously
        // Real implementation would be truly async
        self.data
    }

    /// Async collect that waits for all tasks to complete
    pub async fn collect_async(self) -> Vec<T> {
        // This would actually wait for async operations to complete
        // For now, delegate to sync collect
        self.collect().await
    }

    /// Reduce operation
    pub async fn reduce<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + 'static,
        T: Clone,
    {
        self.data.into_iter().reduce(func)
    }

    /// Parallel reduce with work-stealing
    pub async fn reduce_parallel<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + 'static,
        T: Clone,
    {
        // Delegate to execution context for parallel reduction
        self.context
            .execute_reduce(self.data, func)
            .await
            .unwrap_or(None)
    }

    /// For each operation with side effects
    pub async fn for_each<F>(self, func: F)
    where
        F: Fn(T) + Send + Sync + 'static,
    {
        let _ = self.context.execute_iter(self.data, func);
    }

    /// Async for each operation
    pub async fn for_each_async<F, Fut>(self, func: F)
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let _ = self.context.execute_async_for_each(self.data, func).await;
    }

    /// Partition data across multiple systems/nodes
    pub async fn partition_across_systems<F>(self, partition_func: F) -> Vec<MoiraiIterator<T>>
    where
        F: Fn(&T) -> usize + Send + Sync + 'static,
        T: Clone,
    {
        match &self.context {
            ExecutionContext::MultiSystem(ctx) => {
                ctx.partition_data(self.data, partition_func).await
            }
            ExecutionContext::Distributed(ctx) => {
                ctx.partition_data(self.data, partition_func).await
            }
            _ => {
                // Fallback: single partition
                vec![self]
            }
        }
    }

    /// Convert to async stream for streaming processing
    pub fn into_async_stream(self) -> impl futures::Stream<Item = T> + Send + 'static
    where
        T: 'static,
    {
        futures::stream::iter(self.data)
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

/// Create a distributed iterator for multi-machine processing
pub fn moirai_iter_distributed<T: Send + Clone + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::distributed(data)
}

/// Create a multi-system iterator for coordinated compute across multiple machines and GPUs
pub fn moirai_iter_multi_system<T: Send + Clone + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::multi_system(data)
}

/// Parallel range iterator - Rayon compatibility
pub fn par_range(start: usize, end: usize) -> impl ParallelIterator<Item = usize> {
    parallel::RangeParIter::new(start, end)
}

/// Async range iterator - Tokio compatibility  
pub fn async_range(start: usize, end: usize) -> impl AsyncIterator<Item = usize> {
    async_iter::AsyncRangeIter::new(start, end)
}
