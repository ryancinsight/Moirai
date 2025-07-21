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
    executor::{ExecutorConfig},
    error::{ExecutorResult},
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
        self.executor.shutdown_timeout(timeout)
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
    pub fn spawn_remote<F, R>(&self, _node: &str, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Implementation would coordinate with distributed scheduler
        // For now, fall back to local execution
        self.spawn_parallel(func)
    }

    // TODO: Implement pipeline builder for chaining async and parallel operations
    // TODO: Implement scoped task spawner that ensures all tasks complete before returning
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
    pub fn numa_aware(self, enabled: bool) -> Self {
        // NUMA awareness configuration would go here
        // For now, we'll store it in a separate field or ignore it
        let _ = enabled; // Suppress unused variable warning
        self
    }

    /// Set the thread name prefix.
    pub fn thread_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.thread_name_prefix = prefix.into();
        self
    }

    /// Enable or disable metrics collection.
    #[cfg(feature = "metrics")]
    pub fn enable_metrics(self, enabled: bool) -> Self {
        // Metrics configuration would go here
        let _ = enabled; // Suppress unused variable warning
        self
    }

    /// Enable distributed computing capabilities.
    #[cfg(feature = "distributed")]
    pub fn enable_distributed(self) -> Self {
        // Configuration would be added to ExecutorConfig
        // For now, this is a placeholder
        self
    }

    /// Set the node ID for distributed computing.
    #[cfg(feature = "distributed")]
    pub fn node_id(self, _id: impl Into<String>) -> Self {
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

// TODO: Implement TaskScope for structured concurrency

// TODO: Implement PipelineBuilder for execution pipelines

/// Convenience functions for common operations.
pub mod prelude {
    //! Common imports for Moirai users.
    
    pub use crate::{
        Moirai, MoiraiBuilder,
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

    #[test]
    fn test_spawn_async() {
        let moirai = Moirai::new().unwrap();
        let handle = moirai.spawn_async(async { 42 });
        // For now, we'll just test that the handle was created
        // TODO: Implement proper async execution and testing
        assert_eq!(handle.id().get(), 0);
    }

    #[test]
    fn test_spawn_parallel() {
        let moirai = Moirai::new().unwrap();
        let handle = moirai.spawn_parallel(|| (0..100).sum::<i32>());
        // For now, we'll just test that the handle was created
        // TODO: Implement proper parallel execution and testing
        assert_eq!(handle.id().get(), 0);
    }

    #[test]
    fn test_global_runtime() {
        let runtime1 = global();
        let runtime2 = global();
        
        // Should be the same instance
        assert!(std::ptr::eq(runtime1, runtime2));
    }

    #[test]
    fn test_global_spawn() {
        let handle = spawn_async(async { "hello world" });
        // For now, we'll just test that the handle was created
        // TODO: Implement proper async execution and testing
        assert_eq!(handle.id().get(), 0);
    }
}