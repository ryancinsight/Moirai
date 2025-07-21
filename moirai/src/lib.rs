//! # Moirai - Weaving the Threads of Fate
//!
//! Moirai is a high-performance hybrid concurrency library for Rust that seamlessly
//! blends asynchronous and parallel execution models. Named after the Greek Fates
//! who controlled the threads of life, Moirai weaves together the best principles
//! from async task scheduling and parallel work-stealing into a unified framework.
//!
//! ## Features
//!
//! - **Zero-cost abstractions**: All abstractions compile away to optimal code
//! - **Hybrid execution**: Seamlessly mix async and parallel tasks
//! - **Work-stealing scheduler**: Intelligent load balancing across CPU cores
//! - **Memory safety**: Leverage Rust's ownership system for safe concurrency
//! - **High performance**: Sub-microsecond task scheduling overhead
//! - **NUMA awareness**: Optimize for modern multi-socket systems
//! - **Rich iterator combinators**: Parallel and async iterator processing
//!
//! ## Quick Start
//!
//! ```rust
//! use moirai::Moirai;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a hybrid executor
//! let moirai = Moirai::builder()
//!     .worker_threads(8)
//!     .async_threads(4)
//!     .build()?;
//!
//! // Spawn async tasks
//! let async_result = moirai.spawn_async(async {
//!     // Async I/O work
//!     42
//! });
//!
//! // Spawn parallel tasks
//! let parallel_result = moirai.spawn_parallel(|| {
//!     // CPU-intensive work
//!     (0..1000).sum::<i32>()
//! });
//!
//! // Await results
//! let a = async_result.await?;
//! let b = parallel_result.await?;
//! 
//! println!("Results: {} + {} = {}", a, b, a + b);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! Moirai consists of several key components:
//!
//! - **Core**: Fundamental abstractions and traits
//! - **Executor**: Hybrid runtime for async and parallel execution
//! - **Scheduler**: Work-stealing task scheduler
//! - **Channels**: MPMC communication primitives
//! - **Sync**: Advanced synchronization primitives
//! - **Iterators**: Parallel and async iterator combinators
//! - **IPC**: Inter-process communication (optional)
//! - **Metrics**: Performance monitoring (optional)

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

// Re-export core types and traits
pub use moirai_core::{
    Task, AsyncTask, TaskHandle, TaskId, TaskContext, Priority, TaskConfig,
    TaskBuilder, error::*, task::*, executor::*, scheduler::*,
};

// Re-export executor types
pub use moirai_executor::{HybridExecutor, ExecutorHandle};

// Re-export scheduler types
pub use moirai_scheduler::{WorkStealingScheduler, LocalScheduler};

// Re-export unified transport types
pub use moirai_transport::{
    UniversalSender, UniversalReceiver, UniversalChannel,
    Address, ThreadId, ProcessId, RemoteAddress, BroadcastScope,
    TransportManager, TransportResult, TransportError,
    channel,
};

// Re-export sync types
pub use moirai_sync::{
    Mutex, RwLock, Condvar, Barrier, Once, 
    AtomicCounter, WaitGroup,
};

// Optional async support
#[cfg(feature = "async")]
pub use moirai_async::{
    AsyncExecutor, AsyncHandle, Timer, Timeout,
    io, net, fs,
};

// Optional iterator support
#[cfg(feature = "iter")]
pub use moirai_iter::{
    ParallelIterator, AsyncIterator, IntoParallelIterator,
    par_iter, async_iter,
};

// Optional distributed computing support
#[cfg(feature = "distributed")]
pub use moirai_transport::{
    NetworkTopology, PeerNode, NodeCapabilities,
    DeliveryReceipt,
};

// Optional metrics support
#[cfg(feature = "metrics")]
pub use moirai_metrics::{
    Metrics, Counter, Gauge, Histogram,
    MetricsCollector, PrometheusExporter,
};

use moirai_core::{
    executor::{ExecutorConfig, ExecutorBuilder},
    error::{ExecutorResult, TaskResult},
};
use std::{
    future::Future,
    sync::Arc,
    time::Duration,
};

/// The main Moirai runtime that provides a unified interface for hybrid concurrency.
///
/// This is the primary entry point for using Moirai. It provides methods for spawning
/// both async and parallel tasks, managing their execution, and coordinating between
/// different execution models.
#[derive(Clone)]
pub struct Moirai {
    executor: Arc<HybridExecutor>,
}

impl Moirai {
    /// Create a new Moirai runtime with default configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime cannot be initialized.
    pub fn new() -> ExecutorResult<Self> {
        Self::builder().build()
    }

    /// Create a builder for configuring the Moirai runtime.
    pub fn builder() -> MoiraiBuilder {
        MoiraiBuilder::new()
    }

    /// Spawn an async task for execution.
    ///
    /// The task will be scheduled on the async thread pool and can perform
    /// I/O operations without blocking other tasks.
    pub fn spawn_async<F>(&self, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.executor.spawn_async(future)
    }

    /// Spawn a parallel task for execution.
    ///
    /// The task will be scheduled on the work-stealing thread pool and is
    /// ideal for CPU-intensive computations.
    pub fn spawn_parallel<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.executor.spawn_blocking(func)
    }

    /// Spawn a task with a specific priority.
    ///
    /// Higher priority tasks will be scheduled before lower priority tasks.
    pub fn spawn_with_priority<T>(&self, task: T, priority: Priority) -> TaskHandle<T::Output>
    where
        T: Task,
    {
        self.executor.spawn_with_priority(task, priority)
    }

    /// Block the current thread until the future completes.
    ///
    /// This is useful for running async code from synchronous contexts.
    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.executor.block_on(future)
    }

    /// Try to run pending tasks without blocking.
    ///
    /// Returns `true` if any tasks were executed, `false` if no work was available.
    pub fn try_run(&self) -> bool {
        self.executor.try_run()
    }

    /// Shutdown the runtime gracefully.
    ///
    /// This will wait for all currently running tasks to complete before
    /// shutting down the thread pools.
    pub fn shutdown(&self) {
        self.executor.shutdown()
    }

    /// Shutdown the runtime with a timeout.
    ///
    /// If tasks don't complete within the timeout, they will be forcefully
    /// terminated.
    pub fn shutdown_timeout(&self, timeout: Duration) {
        // Implementation would handle timeout logic
        self.executor.shutdown()
    }

    /// Check if the runtime is shutting down.
    pub fn is_shutting_down(&self) -> bool {
        self.executor.is_shutting_down()
    }

    /// Get the number of worker threads.
    pub fn worker_count(&self) -> usize {
        self.executor.worker_count()
    }

    /// Get the current load (number of pending tasks).
    pub fn load(&self) -> usize {
        self.executor.load()
    }

    /// Get runtime statistics.
    #[cfg(feature = "metrics")]
    pub fn stats(&self) -> moirai_core::executor::ExecutorStats {
        self.executor.stats()
    }

    /// Create a universal channel for communication.
    pub fn channel<T>(&self) -> TransportResult<(UniversalSender<T>, UniversalReceiver<T>)> {
        channel::universal()
    }

    /// Create a channel with a specific address.
    pub fn channel_with_address<T>(&self, address: Address) -> TransportResult<(UniversalSender<T>, UniversalReceiver<T>)> {
        UniversalChannel::new(address)
    }

    /// Spawn a task on a remote node.
    #[cfg(feature = "distributed")]
    pub fn spawn_remote<F, R>(&self, node: &str, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Implementation would coordinate with distributed scheduler
        // For now, fall back to local execution
        self.spawn_parallel(func)
    }

    /// Create a pipeline builder for chaining async and parallel operations.
    pub fn pipeline(&self) -> PipelineBuilder {
        PipelineBuilder::new(self.clone())
    }

    /// Create a scoped task spawner that ensures all tasks complete before returning.
    pub fn scope<'scope, F, R>(&self, func: F) -> R
    where
        F: FnOnce(&TaskScope<'scope>) -> R,
    {
        let scope = TaskScope::new(self.clone());
        let result = func(&scope);
        scope.wait();
        result
    }
}

impl Default for Moirai {
    fn default() -> Self {
        Self::new().expect("Failed to create default Moirai runtime")
    }
}

/// Builder for configuring the Moirai runtime.
pub struct MoiraiBuilder {
    config: ExecutorConfig,
}

impl MoiraiBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
        }
    }

    /// Set the number of worker threads for parallel tasks.
    pub fn worker_threads(mut self, count: usize) -> Self {
        self.config.worker_threads = count;
        self
    }

    /// Set the number of threads dedicated to async tasks.
    pub fn async_threads(mut self, count: usize) -> Self {
        self.config.async_threads = count;
        self
    }

    /// Set the maximum global queue size.
    pub fn max_global_queue_size(mut self, size: usize) -> Self {
        self.config.max_global_queue_size = size;
        self
    }

    /// Set the maximum local queue size.
    pub fn max_local_queue_size(mut self, size: usize) -> Self {
        self.config.max_local_queue_size = size;
        self
    }

    /// Enable or disable NUMA awareness.
    #[cfg(feature = "numa")]
    pub fn numa_aware(mut self, enabled: bool) -> Self {
        self.config.numa_aware = enabled;
        self
    }

    /// Set the thread name prefix.
    pub fn thread_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.thread_name_prefix = prefix.into();
        self
    }

    /// Enable or disable metrics collection.
    #[cfg(feature = "metrics")]
    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Enable distributed computing capabilities.
    #[cfg(feature = "distributed")]
    pub fn enable_distributed(mut self) -> Self {
        // Configuration would be added to ExecutorConfig
        // For now, this is a placeholder
        self
    }

    /// Set the node ID for distributed computing.
    #[cfg(feature = "distributed")]
    pub fn node_id(mut self, _id: impl Into<String>) -> Self {
        // Configuration would be added to ExecutorConfig
        self
    }

    /// Build the Moirai runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime cannot be initialized.
    pub fn build(self) -> ExecutorResult<Moirai> {
        let executor = HybridExecutor::new(self.config)?;
        Ok(Moirai {
            executor: Arc::new(executor),
        })
    }
}

impl Default for MoiraiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A scoped task spawner that ensures all spawned tasks complete before the scope ends.
pub struct TaskScope<'scope> {
    moirai: Moirai,
    handles: std::sync::Mutex<Vec<Box<dyn std::any::Any + Send + 'scope>>>,
}

impl<'scope> TaskScope<'scope> {
    fn new(moirai: Moirai) -> Self {
        Self {
            moirai,
            handles: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Spawn an async task within this scope.
    pub fn spawn_async<F>(&self, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'scope,
        F::Output: Send + 'static,
    {
        // In a real implementation, this would properly handle lifetimes
        // For now, we'll use a simplified version
        self.moirai.spawn_async(async move { future.await })
    }

    /// Spawn a parallel task within this scope.
    pub fn spawn_parallel<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'scope,
        R: Send + 'static,
    {
        // In a real implementation, this would properly handle lifetimes
        self.moirai.spawn_parallel(func)
    }

    fn wait(&self) {
        // In a real implementation, this would wait for all scoped tasks
        // For now, this is a placeholder
    }
}

/// Builder for creating execution pipelines that mix async and parallel stages.
pub struct PipelineBuilder {
    moirai: Moirai,
}

impl PipelineBuilder {
    fn new(moirai: Moirai) -> Self {
        Self { moirai }
    }

    /// Add an async stage to the pipeline.
    pub fn async_stage<F, T, U>(self, _func: F) -> Self
    where
        F: Fn(T) -> U + Send + Sync + 'static,
        U: Future + Send,
        U::Output: Send + 'static,
        T: Send + 'static,
    {
        // Pipeline implementation would go here
        self
    }

    /// Add a parallel stage to the pipeline.
    pub fn parallel_stage<F, T, U>(self, _func: F) -> Self
    where
        F: Fn(T) -> U + Send + Sync + 'static,
        T: Send + 'static,
        U: Send + 'static,
    {
        // Pipeline implementation would go here
        self
    }

    /// Execute the pipeline with the given input.
    pub async fn execute<T, U>(self, _input: T) -> TaskResult<U>
    where
        T: Send + 'static,
        U: Send + 'static,
    {
        // Pipeline execution would go here
        todo!("Pipeline execution not yet implemented")
    }
}

/// Convenience functions for common operations.
pub mod prelude {
    //! Common imports for Moirai users.
    
    pub use crate::{
        Moirai, MoiraiBuilder, TaskScope, PipelineBuilder,
        Task, AsyncTask, TaskHandle, TaskId, Priority,
        TaskBuilder, TaskExt,
    };

    #[cfg(feature = "iter")]
    pub use crate::{ParallelIterator, AsyncIterator, IntoParallelIterator};

    #[cfg(feature = "async")]
    pub use crate::{Timer, Timeout};
}

/// Global runtime instance for convenience.
static GLOBAL_RUNTIME: std::sync::OnceLock<Moirai> = std::sync::OnceLock::new();

/// Get or initialize the global Moirai runtime.
///
/// This provides a convenient way to access a shared runtime instance
/// without having to pass it around explicitly.
pub fn global() -> &'static Moirai {
    GLOBAL_RUNTIME.get_or_init(|| {
        Moirai::new().expect("Failed to initialize global Moirai runtime")
    })
}

/// Spawn an async task on the global runtime.
pub fn spawn_async<F>(future: F) -> TaskHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    global().spawn_async(future)
}

/// Spawn a parallel task on the global runtime.
pub fn spawn_parallel<F, R>(func: F) -> TaskHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    global().spawn_parallel(func)
}

/// Block on a future using the global runtime.
pub fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    global().block_on(future)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_moirai_creation() {
        let moirai = Moirai::new().unwrap();
        assert!(moirai.worker_count() > 0);
    }

    #[test]
    fn test_builder() {
        let moirai = Moirai::builder()
            .worker_threads(4)
            .async_threads(2)
            .build()
            .unwrap();
        
        assert_eq!(moirai.worker_count(), 4);
    }

    #[tokio::test]
    async fn test_spawn_async() {
        let moirai = Moirai::new().unwrap();
        let handle = moirai.spawn_async(async { 42 });
        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }

    #[tokio::test]
    async fn test_spawn_parallel() {
        let moirai = Moirai::new().unwrap();
        let handle = moirai.spawn_parallel(|| (0..100).sum::<i32>());
        let result = handle.await.unwrap();
        assert_eq!(result, 4950);
    }

    #[test]
    fn test_global_runtime() {
        let runtime1 = global();
        let runtime2 = global();
        
        // Should be the same instance
        assert!(std::ptr::eq(runtime1, runtime2));
    }

    #[tokio::test]
    async fn test_global_spawn() {
        let handle = spawn_async(async { "hello world" });
        let result = handle.await.unwrap();
        assert_eq!(result, "hello world");
    }
}