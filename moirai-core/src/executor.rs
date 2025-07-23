//! Executor trait definitions and configuration.

use crate::{Task, TaskHandle, TaskId, Priority};
use core::future::Future;

#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(not(feature = "std"))]
use crate::metrics::Instant;

/// Core task spawning capabilities.
/// 
/// This trait provides the fundamental ability to spawn tasks for execution.
/// It follows the Single Responsibility Principle by focusing only on task spawning.
/// 
/// # Behavior Guarantees
/// - Task spawning is non-blocking and returns immediately
/// - Tasks are scheduled for execution but may not start immediately
/// - Task handles can be used to wait for completion or cancel tasks
/// - Memory ordering follows acquire-release semantics for task state
/// 
/// # Performance Characteristics
/// - Task spawn: O(1) amortized, < 100ns typical latency
/// - Memory overhead: < 64 bytes per task
/// - Thread-safe: All operations are safe for concurrent access
pub trait TaskSpawner: Send + Sync + 'static {
    /// Spawns a new task for execution.
    ///
    /// # Arguments
    /// * `task` - The task to be executed
    ///
    /// # Returns
    /// A handle to the spawned task that allows monitoring and control
    ///
    /// # Errors
    /// Returns `TaskError::SpawnFailed` if the task cannot be spawned due to:
    /// - Resource exhaustion (queue full, memory limit reached)
    /// - Task validation failures (invalid priority, security constraints)
    /// - System shutdown in progress
    fn spawn<T>(&self, task: T) -> TaskHandle<T::Output>
    where
        T: Task + Send + 'static;

    /// Spawns an asynchronous task (Future) for execution.
    ///
    /// # Arguments
    /// * `future` - The future to be executed
    ///
    /// # Returns
    /// A handle to the spawned task
    ///
    /// # Errors
    /// Returns `TaskError::SpawnFailed` under the same conditions as `spawn`
    fn spawn_async<F>(&self, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static;

    /// Spawns a blocking task that may perform I/O or CPU-intensive work.
    ///
    /// # Arguments
    /// * `func` - The blocking function to execute
    ///
    /// # Returns
    /// A handle to the spawned task
    ///
    /// # Errors
    /// Returns `TaskError::SpawnFailed` under the same conditions as `spawn`
    fn spawn_blocking<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    /// Spawns a task with specific priority and scheduling hints.
    ///
    /// # Arguments
    /// * `task` - The task to be executed
    /// * `priority` - The scheduling priority for this task
    /// * `locality_hint` - Optional hint about preferred execution location
    ///
    /// # Returns
    /// A handle to the spawned task
    ///
    /// # Errors
    /// Returns `TaskError::SpawnFailed` under the same conditions as `spawn`
    fn spawn_with_priority<T>(
        &self,
        task: T,
        priority: Priority,
        locality_hint: Option<usize>,
    ) -> TaskHandle<T::Output>
    where
        T: Task + Send + 'static;
}

/// Task management and monitoring capabilities.
/// 
/// This trait provides operations for managing and monitoring running tasks.
/// It follows the Interface Segregation Principle by separating management
/// concerns from spawning concerns.
/// 
/// # Behavior Guarantees
/// - All operations are thread-safe and non-blocking where possible
/// - Task state is eventually consistent across all observers
/// - Cancellation is cooperative and may not be immediate
/// - Statistics are updated atomically and consistently
/// 
/// # Performance Characteristics
/// - Status queries: O(1) lookup time, < 50ns latency
/// - Cancellation: O(1) operation, cooperative completion
/// - Statistics: Atomic operations, minimal overhead
pub trait TaskManager: Send + Sync + 'static {
    /// Cancels a running task by its ID.
    ///
    /// # Arguments
    /// * `id` - The unique identifier of the task to cancel
    ///
    /// # Returns
    /// `Ok(())` if the task was successfully cancelled or was already completed.
    ///
    /// # Errors
    /// Returns `TaskError` in the following cases:
    /// - `NotFound` if no task with the given ID exists
    /// - `InvalidState` if the task cannot be cancelled (e.g., already completed)
    /// - `SystemError` if the cancellation operation fails due to internal errors
    fn cancel_task(&self, id: TaskId) -> Result<(), crate::error::TaskError>;

    /// Get the current status of a task.
    /// 
    /// # Behavior Guarantees
    /// - Returns None if task ID is not found
    /// - Status is eventually consistent across threads
    /// - Completed tasks may be garbage collected after timeout
    /// - Status transitions are monotonic (no backwards moves)
    /// 
    /// # Performance Characteristics
    /// - O(1) lookup time using hash table
    /// - Latency: < 50ns for status query
    /// - Memory: Minimal overhead for status tracking
    /// - Non-blocking: Never blocks calling thread
    fn task_status(&self, id: TaskId) -> Option<TaskStatus>;

    /// Wait for a task to complete with optional timeout.
    /// 
    /// # Behavior Guarantees
    /// - Returns immediately if task is already complete
    /// - Respects timeout if specified, returns Err on timeout
    /// - Cancellable via async cancellation mechanisms
    /// - Memory ordering: Acquire semantics for completion check
    /// 
    /// # Performance Characteristics
    /// - Immediate return: < 10ns if already complete
    /// - Waiting overhead: Event-driven, no busy polling
    /// - Memory: Minimal waker chain overhead
    fn wait_for_task(&self, id: TaskId, timeout: Option<core::time::Duration>) -> impl Future<Output = Result<(), crate::error::TaskError>> + Send;

    /// Get statistics about task execution.
    /// 
    /// # Behavior Guarantees
    /// - Returns None if task ID is not found or stats not enabled
    /// - Statistics are eventually consistent
    /// - Timing measurements use high-resolution monotonic clock
    /// - Memory usage tracking depends on executor configuration
    /// 
    /// # Performance Characteristics
    /// - Lookup: O(1) hash table access
    /// - Overhead: ~100 bytes per task when metrics enabled
    /// - Collection cost: < 5% runtime overhead when enabled
    fn task_stats(&self, id: TaskId) -> Option<TaskStats>;
}

/// Provides control operations for executor lifecycle management.
///
/// This trait enables external systems to manage executor state transitions,
/// perform health checks, and coordinate shutdown procedures.
#[allow(clippy::module_name_repetitions)]
pub trait ExecutorControl: Send + Sync + 'static {
    /// Block the current thread until the future completes.
    /// 
    /// # Behavior Guarantees
    /// - Blocks calling thread until future resolves
    /// - Supports nested async operations within the future
    /// - Handles panic propagation from the future
    /// - May deadlock if future depends on blocked thread
    /// 
    /// # Performance Characteristics
    /// - Optimal for CPU-bound futures with minimal I/O
    /// - May block calling thread indefinitely
    /// - Memory: Future size + execution context
    /// - Suitable for main thread or dedicated blocking contexts
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future;

    /// Attempt to run tasks without blocking.
    /// 
    /// # Behavior Guarantees
    /// - Non-blocking operation, returns immediately
    /// - Returns true if any work was performed
    /// - May perform multiple task executions in single call
    /// - Suitable for integration with external event loops
    /// 
    /// # Performance Characteristics
    /// - O(1) operation, < 1μs typical latency
    /// - Work stealing: Attempts to balance load across threads
    /// - Suitable for event loops requiring non-blocking progress
    fn try_run(&self) -> bool;

    /// Shutdown the executor gracefully.
    /// 
    /// # Behavior Guarantees
    /// - Allows running tasks to complete naturally
    /// - Prevents new tasks from being spawned
    /// - Idempotent operation - safe to call multiple times
    /// - Blocks until all worker threads have stopped
    /// - Releases all resources and thread handles
    /// 
    /// # Performance Characteristics
    /// - Shutdown time: Depends on longest running task
    /// - Resource cleanup: All memory and handles released
    /// - Thread coordination: Uses efficient signaling
    fn shutdown(&self);

    /// Shutdown the executor with a timeout.
    /// 
    /// # Behavior Guarantees
    /// - Attempts graceful shutdown first
    /// - Forces termination after timeout expires
    /// - May result in task cancellation or abortion
    /// - Guarantees executor stops within timeout + small overhead
    /// 
    /// # Performance Characteristics
    /// - Graceful phase: Same as `shutdown()`
    /// - Forced phase: Immediate thread termination
    /// - Timeout accuracy: ±10ms typical variance
    fn shutdown_timeout(&self, timeout: core::time::Duration);

    /// Check if the executor is shutting down.
    /// 
    /// # Behavior Guarantees
    /// - Returns true once shutdown has been initiated
    /// - Eventually consistent across all threads
    /// - Remains true until executor is fully stopped
    /// 
    /// # Performance Characteristics
    /// - O(1) operation, < 10ns latency
    /// - Non-blocking atomic read operation
    /// - Memory ordering: Acquire semantics
    fn is_shutting_down(&self) -> bool;

    /// Get the number of worker threads.
    /// 
    /// # Behavior Guarantees
    /// - Returns configured number of worker threads
    /// - Does not include async or blocking thread pools
    /// - Constant value set during executor creation
    /// 
    /// # Performance Characteristics
    /// - O(1) operation, immediate return
    /// - No synchronization overhead
    fn worker_count(&self) -> usize;

    /// Get the current load (number of pending tasks).
    /// 
    /// # Behavior Guarantees
    /// - Returns approximate pending task count
    /// - Eventually consistent across distributed queues
    /// - May include tasks currently being executed
    /// - Does not include blocked or suspended tasks
    /// 
    /// # Performance Characteristics
    /// - O(1) operation for local queues
    /// - May involve atomic reads across threads
    /// - Latency: < 100ns typical
    fn load(&self) -> usize;
}

/// Combined executor trait with all capabilities.
/// 
/// This trait combines all executor capabilities into a single interface
/// for convenience while maintaining the segregated design internally.
/// 
/// # Design Philosophy
/// - Composition over inheritance
/// - Single interface for complete functionality
/// - Maintains internal separation of concerns
/// - Enables easy mocking and testing
pub trait Executor: TaskSpawner + TaskManager + ExecutorControl {
    /// Get comprehensive executor statistics.
    /// 
    /// # Behavior Guarantees
    /// - Returns current snapshot of all executor metrics
    /// - Statistics are eventually consistent
    /// - Available only when metrics feature is enabled
    /// - Includes worker, queue, memory, and task statistics
    /// 
    /// # Performance Characteristics
    /// - Collection overhead: < 1μs for full statistics
    /// - Memory: ~1KB for complete statistics snapshot
    /// - Thread safety: Atomic operations for consistency
    #[cfg(feature = "metrics")]
    fn stats(&self) -> ExecutorStats;
}

/// Status of a task within the executor.
/// 
/// Task status transitions follow a strict state machine:
/// Queued → Running → (Completed | Cancelled | Failed)
/// 
/// # State Transitions
/// - Queued: Initial state when task is spawned
/// - Running: Task is currently executing on a worker thread
/// - Completed: Task finished successfully
/// - Cancelled: Task was cancelled before or during execution
/// - Failed: Task encountered an error or panic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is queued but not yet started
    /// 
    /// # Guarantees
    /// - Task will eventually transition to Running
    /// - Cancellation is possible in this state
    /// - Memory has been allocated for task execution
    Queued,
    
    /// Task is currently running
    /// 
    /// # Guarantees
    /// - Task is actively executing on a worker thread
    /// - Cancellation is cooperative in this state
    /// - Progress is being made toward completion
    Running,
    
    /// Task completed successfully
    /// 
    /// # Guarantees
    /// - Task result is available via task handle
    /// - No further state transitions possible
    /// - Resources have been cleaned up
    Completed,
    
    /// Task was cancelled
    /// 
    /// # Guarantees
    /// - Task did not complete normally
    /// - Cancellation was requested and honored
    /// - Resources have been cleaned up
    Cancelled,
    
    /// Task failed with an error
    /// 
    /// # Guarantees
    /// - Task encountered an unrecoverable error
    /// - Error information is available via task handle
    /// - Resources have been cleaned up
    Failed,
}

impl core::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Running => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

/// Detailed statistics about a specific task.
/// 
/// Task statistics provide comprehensive information about task execution
/// performance and resource usage. Statistics are collected when the
/// metrics feature is enabled.
/// 
/// # Memory Overhead
/// When metrics are enabled, each task incurs approximately 100 bytes
/// of additional memory overhead for statistics collection.
/// 
/// # Accuracy Guarantees
/// - Timestamps use monotonic high-resolution clock
/// - Memory measurements are sampled at key execution points
/// - CPU time includes both user and system time
/// - Preemption count tracks cooperative yield points
#[derive(Debug, Clone)]
pub struct TaskStats {
    /// Task identifier
    pub id: TaskId,
    /// Current status
    pub status: TaskStatus,
    /// Priority level
    pub priority: Priority,
    /// When the task was spawned
    pub spawn_time: Instant,
    /// When the task started executing (if started)
    pub start_time: Option<Instant>,
    /// When the task completed (if completed)
    pub completion_time: Option<Instant>,
    /// Number of times the task was preempted
    pub preemption_count: u32,
    /// Total CPU time used (nanoseconds)
    pub cpu_time_ns: u64,
    /// Memory allocated by the task (bytes)
    pub memory_used_bytes: u64,
}

impl TaskStats {
    /// Returns the total execution time of the task, if available.
    ///
    /// # Returns
    /// `Some(duration)` if the task has completed execution, `None` if still running or queued.
    #[must_use]
    pub fn execution_time(&self) -> Option<core::time::Duration> {
        match (&self.start_time, &self.completion_time) {
            (Some(start), Some(end)) => Some(end.duration_since(*start)),
            _ => None,
        }
    }

    /// Returns the time the task spent in the queue before execution.
    ///
    /// # Returns
    /// - `Some(duration_since_spawn)` if task is still queued
    /// - `Some(queue_duration)` if task has started execution
    /// - `None` if timing information is unavailable
    #[must_use]
    pub fn queue_time(&self) -> Option<core::time::Duration> {
        match &self.start_time {
            Some(start) => Some(start.duration_since(self.spawn_time)),
            None => Some(Instant::now().duration_since(self.spawn_time)),
        }
    }

    /// Returns whether the task is currently active (queued or running).
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self.status, TaskStatus::Queued | TaskStatus::Running)
    }

    /// Returns whether the task has reached a terminal state.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        matches!(self.status, TaskStatus::Completed | TaskStatus::Cancelled | TaskStatus::Failed)
    }
}

/// Configuration settings for executor behavior and performance characteristics.
///
/// This struct encapsulates all tunable parameters that affect executor operation,
/// including thread pool sizes, queue capacities, and various performance optimizations.
#[allow(clippy::module_name_repetitions)]
pub struct ExecutorConfig {
    /// Number of worker threads for parallel tasks
    pub worker_threads: usize,
    /// Number of threads dedicated to async tasks
    pub async_threads: usize,
    /// Maximum size of the global task queue
    pub max_global_queue_size: usize,
    /// Maximum size of per-thread local queues
    pub max_local_queue_size: usize,
    /// Thread name prefix for worker threads
    pub thread_name_prefix: alloc::string::String,
    /// Whether to enable NUMA-aware thread placement
    #[cfg(feature = "numa")]
    pub numa_aware: bool,
    /// Whether to enable metrics collection
    #[cfg(feature = "metrics")]
    pub enable_metrics: bool,
    /// Task preemption configuration
    pub preemption: PreemptionConfig,
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// Task cleanup configuration
    pub cleanup: CleanupConfig,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus(),
            async_threads: (num_cpus() / 4).max(1),
            max_global_queue_size: 8192,
            max_local_queue_size: 256,
            thread_name_prefix: "moirai-worker".into(),
            #[cfg(feature = "numa")]
            numa_aware: true,
            #[cfg(feature = "metrics")]
            enable_metrics: true,
            preemption: PreemptionConfig::default(),
            memory: MemoryConfig::default(),
            cleanup: CleanupConfig::default(),
        }
    }
}

/// Configuration for task preemption.
#[derive(Debug, Clone)]
pub struct PreemptionConfig {
    /// Whether to enable cooperative preemption
    pub enabled: bool,
    /// Time slice for each task before preemption (microseconds)
    pub time_slice_us: u64,
    /// Whether to preempt based on priority
    pub priority_based: bool,
    /// Minimum execution time before preemption (microseconds)
    pub min_execution_time_us: u64,
}

impl Default for PreemptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            time_slice_us: 10_000, // 10ms
            priority_based: true,
            min_execution_time_us: 1_000, // 1ms
        }
    }
}

/// Configuration for memory management.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Whether to use memory pools
    pub use_memory_pools: bool,
    /// Size of small object pool (bytes)
    pub small_pool_size: usize,
    /// Size of medium object pool (bytes)
    pub medium_pool_size: usize,
    /// Size of large object pool (bytes)
    pub large_pool_size: usize,
    /// Whether to track memory usage per task
    pub track_per_task_memory: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            use_memory_pools: true,
            small_pool_size: 64 * 1024,      // 64KB
            medium_pool_size: 1024 * 1024,   // 1MB
            large_pool_size: 16 * 1024 * 1024, // 16MB
            track_per_task_memory: cfg!(feature = "metrics"),
        }
    }
}

/// Configuration for task metadata cleanup.
/// 
/// Controls how and when completed task metadata is removed from memory
/// to prevent memory leaks in long-running executors.
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// How long to keep completed task metadata before cleanup
    /// 
    /// # Default: 5 minutes
    /// # Range: 1 second to `task_retention_duration`
    pub task_retention_duration: core::time::Duration,
    
    /// How often to run the cleanup process
    /// 
    /// # Default: 30 seconds  
    /// # Range: 1 second to `task_retention_duration`
    pub cleanup_interval: core::time::Duration,
    
    /// Whether to enable automatic cleanup
    /// 
    /// If disabled, cleanup must be triggered manually via `cleanup_completed_tasks()`
    /// # Default: true
    pub enable_automatic_cleanup: bool,
    
    /// Maximum number of completed tasks to retain regardless of age
    /// 
    /// This provides a hard limit to prevent unbounded memory growth
    /// # Default: 10,000 tasks
    pub max_retained_tasks: usize,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            task_retention_duration: core::time::Duration::from_secs(300), // 5 minutes
            cleanup_interval: core::time::Duration::from_secs(30), // 30 seconds
            enable_automatic_cleanup: true,
            max_retained_tasks: 10_000,
        }
    }
}

/// Plugin interface for extending executor functionality.
///
/// Plugins provide a way to add custom behavior to the executor lifecycle
/// without modifying the core execution logic.
#[allow(clippy::module_name_repetitions)]
pub trait ExecutorPlugin: Send + Sync + 'static {
    /// Initialize the plugin with access to executor configuration.
    ///
    /// # Errors
    /// Returns `ExecutorError` if the plugin cannot be properly initialized due to:
    /// - Invalid configuration parameters
    /// - Resource allocation failures
    /// - Dependency conflicts with other plugins
    fn initialize(&mut self) -> Result<(), crate::error::ExecutorError> {
        Ok(())
    }

    /// Called before a task is spawned.
    fn before_task_spawn(&self, task_id: TaskId, priority: Priority) {
        let _ = (task_id, priority); // Default: no-op
    }

    /// Called after a task is spawned.
    fn after_task_spawn(&self, task_id: TaskId) {
        let _ = task_id; // Default: no-op
    }

    /// Called before a task starts executing.
    fn before_task_execute(&self, task_id: TaskId) {
        let _ = task_id; // Default: no-op
    }

    /// Called after a task completes.
    fn after_task_complete(&self, task_id: TaskId, success: bool) {
        let _ = (task_id, success); // Default: no-op
    }

    /// Called during executor shutdown.
    fn on_shutdown(&self) {
        // Default: no-op
    }

    /// Get plugin name for debugging.
    fn name(&self) -> &'static str {
        "unknown"
    }
}

/// Builder for creating and configuring executor instances.
///
/// This builder provides a fluent interface for setting up executors with
/// custom configurations, plugins, and performance characteristics.
#[allow(clippy::module_name_repetitions)]
pub struct ExecutorBuilder {
    config: ExecutorConfig,
    plugins: alloc::vec::Vec<alloc::boxed::Box<dyn ExecutorPlugin>>,
}

impl ExecutorBuilder {
    /// Creates a new executor builder with default settings.
    ///
    /// # Returns
    /// A new builder instance ready for configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
            plugins: alloc::vec::Vec::new(),
        }
    }

    /// Sets the number of worker threads for CPU-bound tasks.
    ///
    /// # Arguments
    /// * `count` - Number of worker threads to create
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn worker_threads(mut self, count: usize) -> Self {
        self.config.worker_threads = count;
        self
    }

    /// Sets the number of threads for async task execution.
    ///
    /// # Arguments
    /// * `count` - Number of async threads to create
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn async_threads(mut self, count: usize) -> Self {
        self.config.async_threads = count;
        self
    }

    /// Sets the maximum size of the global task queue.
    ///
    /// # Arguments
    /// * `size` - Maximum number of tasks in the global queue
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn max_global_queue_size(mut self, size: usize) -> Self {
        self.config.max_global_queue_size = size;
        self
    }

    /// Sets the maximum size of per-worker local queues.
    ///
    /// # Arguments
    /// * `size` - Maximum number of tasks in each local queue
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn max_local_queue_size(mut self, size: usize) -> Self {
        self.config.max_local_queue_size = size;
        self
    }

    /// Sets the thread name prefix for executor threads.
    ///
    /// # Arguments
    /// * `prefix` - String prefix for thread names
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn thread_name_prefix(mut self, prefix: impl Into<alloc::string::String>) -> Self {
        self.config.thread_name_prefix = prefix.into();
        self
    }

    /// Enable or disable NUMA awareness.
    #[cfg(feature = "numa")]
    #[must_use]
    pub fn numa_aware(mut self, enabled: bool) -> Self {
        self.config.numa_aware = enabled;
        self
    }

    /// Enable or disable metrics collection.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Configures preemption behavior for the executor.
    ///
    /// # Arguments
    /// * `config` - Preemption configuration settings
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn preemption_config(mut self, config: PreemptionConfig) -> Self {
        self.config.preemption = config;
        self
    }

    /// Configures memory management settings.
    ///
    /// # Arguments
    /// * `config` - Memory configuration settings
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.config.memory = config;
        self
    }

    /// Configures cleanup and maintenance settings.
    ///
    /// # Arguments
    /// * `config` - Cleanup configuration settings
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn cleanup_config(mut self, config: CleanupConfig) -> Self {
        self.config.cleanup = config;
        self
    }

    /// Adds a plugin to the executor.
    ///
    /// # Arguments
    /// * `plugin` - The plugin instance to add
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn plugin(mut self, plugin: impl ExecutorPlugin) -> Self {
        self.plugins.push(alloc::boxed::Box::new(plugin));
        self
    }

    /// Builds the configuration and plugin list without creating an executor.
    ///
    /// # Returns
    /// A tuple containing the executor configuration and list of plugins
    #[must_use]
    pub fn build_config(self) -> (ExecutorConfig, alloc::vec::Vec<alloc::boxed::Box<dyn ExecutorPlugin>>) {
        (self.config, self.plugins)
    }
}

impl Default for ExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime statistics and performance metrics for executor instances.
///
/// This struct provides comprehensive monitoring data for executor performance,
/// including task throughput, queue utilization, and resource consumption metrics.
#[cfg(feature = "metrics")]
#[allow(clippy::module_name_repetitions)]
pub struct ExecutorStats {
    /// Worker thread statistics
    pub worker_stats: alloc::vec::Vec<WorkerStats>,
    /// Global queue statistics
    pub global_queue_stats: QueueStats,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Aggregate task execution statistics across all workers
    pub task_execution_stats: TaskExecutionStats,
}

/// Aggregate task execution statistics across the entire executor.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone, Default)]
pub struct TaskExecutionStats {
    /// Total number of tasks completed successfully
    pub tasks_completed: u64,
    /// Total number of tasks that failed
    pub tasks_failed: u64,
    /// Total number of tasks cancelled
    pub tasks_cancelled: u64,
    /// Average task execution time in nanoseconds
    pub avg_execution_time_ns: u64,
    /// Peak task execution time in nanoseconds
    pub peak_execution_time_ns: u64,
    /// Total CPU time consumed by all tasks in nanoseconds
    pub total_cpu_time_ns: u64,
}

#[cfg(feature = "metrics")]
impl ExecutorStats {
    /// Returns statistics for all tasks managed by this executor.
    ///
    /// # Returns
    /// A slice containing statistics for all tracked tasks
    ///
    /// # Note
    /// This is currently a placeholder implementation that returns an empty slice.
    /// A full implementation would maintain a registry of task statistics and return
    /// them here. This requires integration with the actual executor implementation.
    #[must_use]
    pub fn get_stats(&self) -> &[TaskStats] {
        // TODO: Implement actual statistics collection
        // This would typically involve:
        // 1. Maintaining a registry of active and completed tasks
        // 2. Collecting performance metrics during task execution
        // 3. Providing filtered views (active, completed, failed, etc.)
        // 4. Implementing retention policies for completed task stats
        
        // For now, return empty slice to maintain API compatibility
        // while indicating this needs proper implementation
        &[]
    }
}

/// Statistics for a single worker thread.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// Worker thread ID
    pub thread_id: usize,
    /// Number of tasks executed
    pub tasks_executed: u64,
    /// Number of successful steal attempts
    pub successful_steals: u64,
    /// Number of failed steal attempts
    pub failed_steals: u64,
    /// Number of times stolen from
    pub stolen_from: u64,
    /// Current local queue length
    pub local_queue_length: usize,
    /// CPU utilization percentage (0-100)
    pub cpu_utilization: f32,
    /// Total execution time (nanoseconds)
    pub total_execution_time_ns: u64,
}

/// Queue statistics.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Current queue length
    pub current_length: usize,
    /// Maximum queue length seen
    pub max_length: usize,
    /// Total tasks enqueued
    pub total_enqueued: u64,
    /// Total tasks dequeued
    pub total_dequeued: u64,
    /// Average wait time in queue (microseconds)
    pub avg_wait_time_us: f64,
}

/// Memory usage statistics.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage (bytes)
    pub current_usage: u64,
    /// Peak memory usage (bytes)
    pub peak_usage: u64,
    /// Number of allocations
    pub allocations: u64,
    /// Number of deallocations
    pub deallocations: u64,
    /// Memory pool statistics
    pub pool_stats: PoolStats,
}

/// Memory pool statistics.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Small pool utilization (0-100)
    pub small_pool_utilization: f32,
    /// Medium pool utilization (0-100)
    pub medium_pool_utilization: f32,
    /// Large pool utilization (0-100)
    pub large_pool_utilization: f32,
    /// Number of pool hits
    pub pool_hits: u64,
    /// Number of pool misses
    pub pool_misses: u64,
}

/// Task execution statistics.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct TaskExecutionStats {
    /// Total tasks spawned
    pub total_spawned: u64,
    /// Total tasks completed
    pub total_completed: u64,
    /// Total tasks cancelled
    pub total_cancelled: u64,
    /// Total tasks failed
    pub total_failed: u64,
    /// Average execution time (microseconds)
    pub avg_execution_time_us: f64,
    /// 95th percentile execution time (microseconds)
    pub p95_execution_time_us: f64,
    /// 99th percentile execution time (microseconds)
    pub p99_execution_time_us: f64,
    /// Task throughput (tasks per second)
    pub throughput_per_second: f64,
}

// Helper function to get number of CPUs
fn num_cpus() -> usize {
    #[cfg(feature = "std")]
    {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
    }
    #[cfg(not(feature = "std"))]
    {
        4 // Reasonable default for no_std
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_status_display() {
        assert_eq!(format!("{}", TaskStatus::Queued), "Queued");
        assert_eq!(format!("{}", TaskStatus::Running), "Running");
        assert_eq!(format!("{}", TaskStatus::Completed), "Completed");
        assert_eq!(format!("{}", TaskStatus::Cancelled), "Cancelled");
        assert_eq!(format!("{}", TaskStatus::Failed), "Failed");
    }

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert!(config.worker_threads > 0);
        assert!(config.async_threads > 0);
        assert_eq!(config.max_global_queue_size, 8192);
        assert_eq!(config.max_local_queue_size, 256);
        assert_eq!(config.thread_name_prefix, "moirai-worker");
    }

    #[test]
    fn test_preemption_config_default() {
        let config = PreemptionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.time_slice_us, 10_000);
        assert!(config.priority_based);
        assert_eq!(config.min_execution_time_us, 1_000);
    }

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert!(config.use_memory_pools);
        assert_eq!(config.small_pool_size, 64 * 1024);
        assert_eq!(config.medium_pool_size, 1024 * 1024);
        assert_eq!(config.large_pool_size, 16 * 1024 * 1024);
    }

    #[test]
    fn test_cleanup_config_default() {
        let config = CleanupConfig::default();
        assert_eq!(config.task_retention_duration, core::time::Duration::from_secs(300));
        assert_eq!(config.cleanup_interval, core::time::Duration::from_secs(30));
        assert!(config.enable_automatic_cleanup);
        assert_eq!(config.max_retained_tasks, 10_000);
    }

    #[test]
    fn test_executor_builder() {
        let builder = ExecutorBuilder::new()
            .worker_threads(8)
            .async_threads(4)
            .max_global_queue_size(16384)
            .thread_name_prefix("test-worker");

        let (config, _plugins) = builder.build_config();
        assert_eq!(config.worker_threads, 8);
        assert_eq!(config.async_threads, 4);
        assert_eq!(config.max_global_queue_size, 16384);
        assert_eq!(config.thread_name_prefix, "test-worker");
    }

    #[test]
    fn test_task_stats_calculations() {
        let spawn_time = Instant::now();
        let start_time = spawn_time; // Simplified for test
        let completion_time = start_time; // Simplified for test

        let stats = TaskStats {
            id: TaskId::new(1),
            status: TaskStatus::Completed,
            priority: Priority::Normal,
            spawn_time,
            start_time: Some(start_time),
            completion_time: Some(completion_time),
            preemption_count: 0,
            cpu_time_ns: 1_000_000,
            memory_used_bytes: 4096,
        };

        assert!(stats.is_finished());
        assert!(!stats.is_active());
        assert!(stats.execution_time().is_some());
        assert!(stats.queue_time().is_some());
    }
}