use crate::{MoiraiBuilder, MoiraiScope};
#[cfg(feature = "metrics")]
use moirai_core::executor::Executor;
use moirai_core::{
    error::*,
    executor::{ExecutorControl, TaskSpawner},
    Priority, Task, TaskBuilder, TaskHandle,
};
use moirai_executor::{BlockingTask, HybridExecutor, SyncTask};
use std::{future::Future, sync::Arc, time::Duration};

/// The main Moirai runtime that provides a unified interface for hybrid concurrency.
///
/// This is the primary entry point for using Moirai. It provides methods for spawning
/// both async and parallel tasks, managing their execution, and coordinating between
/// different execution models.
///
/// # Examples
///
/// ```
/// use moirai::Moirai;
/// use std::sync::atomic::{AtomicU32, Ordering};
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a new runtime
/// let runtime = Moirai::new()?;
///
/// // Spawn a parallel task
/// let counter = Arc::new(AtomicU32::new(0));
/// let counter_clone = counter.clone();
/// let handle = runtime.spawn_fn(move || {
///     for _ in 0..1000 {
///         counter_clone.fetch_add(1, Ordering::Relaxed);
///     }
///     counter_clone.load(Ordering::Relaxed)
/// });
///
/// // Spawn an async task
/// let async_handle = runtime.spawn_async(async {
///     // Simulate some async work
///     std::thread::sleep(std::time::Duration::from_millis(10));
///     "async task completed"
/// });
///
/// // The tasks will execute concurrently
/// println!("Tasks spawned, runtime is working...");
///
/// // Shutdown gracefully
/// runtime.shutdown();
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Moirai {
    pub(crate) executor: Arc<HybridExecutor>,
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
    #[must_use]
    pub fn builder() -> MoiraiBuilder {
        MoiraiBuilder::new()
    }

    /// Spawn a task for parallel execution.
    ///
    /// This is a convenience method for spawning CPU-bound tasks.
    ///
    /// # Panics
    ///
    /// Panics if the executor fails to spawn the task, which should not happen
    /// under normal circumstances unless the runtime is shutting down.
    pub fn spawn<T>(&self, task: T) -> TaskHandle<T::Output>
    where
        T: Task,
    {
        self.executor.spawn(task).expect("Failed to spawn task")
    }

    /// Spawn a parallel task using a closure.
    ///
    /// The task will be executed on the work-stealing thread pool.
    ///
    /// # Panics
    ///
    /// Panics if the executor fails to spawn the blocking task, which should not happen
    /// under normal circumstances unless the runtime is shutting down.
    pub fn spawn_fn<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.executor
            .spawn_blocking(func)
            .expect("Failed to spawn blocking task")
    }

    /// Spawn an async task for execution.
    ///
    /// The task will be executed on the async thread pool.
    ///
    /// # Panics
    ///
    /// Panics if the executor fails to spawn the async task, which should not happen
    /// under normal circumstances unless the runtime is shutting down.
    pub fn spawn_async<F>(&self, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.executor
            .spawn_async(future)
            .expect("Failed to spawn async task")
    }

    /// Spawn a blocking task that may block the current thread.
    ///
    /// Use this for I/O-bound or blocking operations.
    ///
    /// # Panics
    ///
    /// Panics if the executor fails to spawn the blocking task, which should not happen
    /// under normal circumstances unless the runtime is shutting down.
    pub fn spawn_blocking<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.executor
            .spawn_blocking(func)
            .expect("Failed to spawn blocking task")
    }

    /// Spawn a fire-and-forget closure whose result is discarded.
    ///
    /// This is the cheapest dispatch path: it returns no handle and skips the
    /// per-task result-slot allocation that [`spawn_fn`](Self::spawn_fn)
    /// performs, making it the right choice for background work whose output is
    /// not needed (event handlers, logging, cache warming). The task is still
    /// drained on [`shutdown`](Self::shutdown).
    ///
    /// # Panics
    ///
    /// Panics if the executor fails to spawn the task, which should not happen
    /// unless the runtime is shutting down.
    pub fn spawn_detached<F>(&self, func: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.executor
            .spawn_detached(func)
            .expect("Failed to spawn detached task");
    }

    /// Run a completion-only scoped fan-out on the unified scheduler.
    ///
    /// Use this when tasks only need to publish side effects through borrowed
    /// synchronization primitives and the caller must wait for all tasks before
    /// continuing. Scoped jobs may be coalesced and start after the scope body
    /// has finished registering work.
    ///
    /// # Errors
    ///
    /// Returns an executor error if the runtime is shutting down or if a scoped
    /// task panics.
    pub fn scope<'scope, F>(&'scope self, body: F) -> ExecutorResult<()>
    where
        F: FnOnce(&MoiraiScope<'scope>) -> ExecutorResult<()>,
    {
        self.executor.scope::<BlockingTask, _>(body)
    }

    /// Run indexed work in worker-sized chunks on the unified scheduler.
    ///
    /// Use this for CPU-bound data-parallel fan-out where the caller needs
    /// completion, not one task handle per item. Work executes through the
    /// compute-worker pool; potentially blocking work belongs on [`Self::scope`].
    /// The closure may borrow data that lives for the call because all chunks
    /// complete before this method returns.
    ///
    /// # Errors
    ///
    /// Returns an executor error if the runtime is shutting down or if any
    /// chunk panics.
    pub fn for_each_indexed<'scope, F>(&'scope self, count: usize, task: F) -> ExecutorResult<()>
    where
        F: Fn(usize) + Send + Sync + 'scope,
    {
        self.executor.for_each_indexed::<SyncTask, _>(count, task)
    }

    /// Run indexed map/reduce in worker-sized chunks on the unified scheduler.
    ///
    /// `identity` must be the neutral element for `reduce`. Use this for
    /// CPU-bound indexed data-parallel reductions where per-item task handles
    /// are not required. Work executes through the compute-worker pool.
    ///
    /// # Errors
    ///
    /// Returns an executor error if the runtime is shutting down or if any
    /// chunk panics.
    pub fn map_reduce_indexed<'scope, T, Map, Reduce>(
        &'scope self,
        count: usize,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> ExecutorResult<T>
    where
        T: Send + Clone + 'scope,
        Map: Fn(usize) -> T + Send + Sync + 'scope,
        Reduce: Fn(T, T) -> T + Send + Sync + 'scope,
    {
        self.executor
            .map_reduce_indexed::<SyncTask, _, _, _>(count, identity, map, reduce)
    }

    /// Spawn a task with a specific priority.
    ///
    /// Higher priority tasks will be executed before lower priority tasks.
    ///
    /// # Panics
    ///
    /// Panics if the executor fails to spawn the task with priority, which should not happen
    /// under normal circumstances unless the runtime is shutting down.
    pub fn spawn_with_priority<T>(&self, task: T, priority: Priority) -> TaskHandle<T::Output>
    where
        T: Task,
    {
        self.executor
            .spawn_with_priority(task, priority, None)
            .expect("Failed to spawn task with priority")
    }

    /// Spawn a closure with priority as a task (convenience method).
    pub fn spawn_fn_with_priority<F, R>(&self, f: F, priority: Priority) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Let the executor handle ID assignment and priority
        let task = TaskBuilder::new().build(f);
        self.spawn_with_priority(task, priority)
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
    #[must_use]
    pub fn try_run(&self) -> bool {
        self.executor.try_run()
    }

    /// Returns true when queued or active runtime work exists.
    #[must_use]
    pub fn has_work(&self) -> bool {
        self.executor.has_work()
    }

    /// Wait until queued and active runtime work completes without shutting down workers.
    ///
    /// Use this as a non-destructive process-fusion barrier when producers have
    /// finished submitting a batch and the runtime should process all available
    /// work before the caller continues. New tasks submitted after this method
    /// observes quiescence belong to a later batch.
    ///
    /// # Errors
    ///
    /// Returns an executor error if the scheduler join operation fails.
    pub fn join(&self) -> ExecutorResult<()> {
        self.executor.join()
    }

    /// Shutdown the runtime gracefully.
    ///
    /// This will wait for all currently running tasks to complete before
    /// shutting down the thread pools.
    pub fn shutdown(&self) {
        self.executor.shutdown();
    }

    /// Shutdown the runtime with a timeout.
    ///
    /// If tasks don't complete within the timeout, they will be forcefully
    /// terminated.
    pub fn shutdown_timeout(&self, timeout: Duration) {
        self.executor.shutdown_timeout(timeout);
    }

    /// Check if the runtime is shutting down.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.executor.is_shutting_down()
    }

    /// Get the number of worker threads.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.executor.worker_count()
    }

    /// Get the current load (number of pending tasks).
    #[must_use]
    pub fn load(&self) -> usize {
        self.executor.load()
    }

    /// Get runtime statistics.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn stats(&self) -> moirai_core::executor::ExecutorStats {
        self.executor.stats()
    }

    /// Create a channel for communication, bounded at
    /// [`DEFAULT_CHANNEL_CAPACITY`].
    ///
    /// Bounded is the default because an unbounded queue converts a slow
    /// consumer into unbounded memory growth: a full channel blocks its
    /// producer (or returns [`ChannelError::Full`] from `try_send`) instead of
    /// allocating. Use [`Self::bounded_channel`] when the right capacity is
    /// known; the unbounded queue remains available as
    /// `moirai_core::channel::unbounded`, whose documentation states the cost.
    ///
    /// [`DEFAULT_CHANNEL_CAPACITY`]: moirai_core::channel::DEFAULT_CHANNEL_CAPACITY
    /// [`ChannelError::Full`]: moirai_core::channel::ChannelError::Full
    #[must_use]
    pub fn channel<T: Send + 'static>(
        &self,
    ) -> (
        moirai_core::channel::MpmcSender<T>,
        moirai_core::channel::MpmcReceiver<T>,
    ) {
        moirai_core::channel::mpmc(moirai_core::channel::DEFAULT_CHANNEL_CAPACITY)
    }

    /// Create a bounded channel with an explicit capacity.
    #[must_use]
    pub fn bounded_channel<T: Send + 'static>(
        &self,
        capacity: usize,
    ) -> (
        moirai_core::channel::MpmcSender<T>,
        moirai_core::channel::MpmcReceiver<T>,
    ) {
        moirai_core::channel::mpmc(capacity)
    }

    /// Create a GPU context for GPU-accelerated computing
    ///
    /// This initializes a GPU context using wgpu-rs for cross-platform GPU support.
    /// The context can be used to create compute pipelines and execute GPU tasks.
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable GPU device is found or if GPU initialization fails.
    #[cfg(feature = "gpu")]
    pub async fn create_gpu_context(&self) -> Result<moirai_gpu::GpuContext, moirai_gpu::GpuError> {
        moirai_gpu::GpuContext::new().await
    }

    /// Create a GPU context with specific device preferences
    ///
    /// This allows fine-grained control over GPU device selection.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU device meeting the preferences is found.
    #[cfg(feature = "gpu")]
    pub async fn create_gpu_context_with_preferences(
        &self,
        preferences: moirai_gpu::DevicePreferences,
    ) -> Result<moirai_gpu::GpuContext, moirai_gpu::GpuError> {
        moirai_gpu::GpuContext::with_preferences(preferences).await
    }

    /// Spawn a GPU task for execution
    ///
    /// This spawns a GPU-accelerated task that will be executed on the GPU.
    /// The task must implement the `GpuTask` trait.
    ///
    /// # Panics
    ///
    /// Panics if the GPU context is not available or if the task fails to spawn.
    #[cfg(feature = "gpu")]
    pub fn spawn_gpu<T>(
        &self,
        gpu_context: &moirai_gpu::GpuContext,
        task: T,
    ) -> moirai_gpu::GpuTaskFuture<T::Output>
    where
        T: moirai_gpu::GpuTask + Send + 'static,
        T::Output: Send + 'static,
    {
        gpu_context.spawn_gpu_task(task)
    }
}

impl Default for Moirai {
    fn default() -> Self {
        Self::new().expect("Failed to create default Moirai runtime")
    }
}
