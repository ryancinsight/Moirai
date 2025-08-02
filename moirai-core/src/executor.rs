//! Executor trait and implementations.
//!
//! This module provides the core executor abstraction for the Moirai runtime.
//! It defines traits for task spawning, management, and lifecycle control.

use crate::{Task, TaskId, Priority, TaskHandle, TaskContext};
use crate::error::ExecutorResult;
use crate::platform::*;
use core::cell::UnsafeCell;

// Thread-local task context for improved locality (inspired by Tokio)
crate::thread_local_static! {
    static CURRENT_TASK: UnsafeCell<Option<TaskId>> = UnsafeCell::new(None)
}

crate::thread_local_static! {
    static EXECUTOR_CONTEXT: UnsafeCell<Option<TaskContext>> = UnsafeCell::new(None)
}

crate::thread_local_static! {
    static LOCAL_QUEUE: UnsafeCell<Vec<Box<dyn Send>>> = UnsafeCell::new(Vec::with_capacity(32))
}

/// Get the current task ID if running within a task context
pub fn current_task_id() -> Option<TaskId> {
    CURRENT_TASK.with(|cell| unsafe { (*cell.get()).clone() })
}

/// Set the current task ID for this thread
#[allow(dead_code)]
pub(crate) fn set_current_task(id: Option<TaskId>) {
    CURRENT_TASK.with(|cell| unsafe {
        *cell.get() = id;
    });
}

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
    fn spawn<T>(&self, task: T) -> ExecutorResult<TaskHandle<T::Output>>
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
    fn spawn_async<F>(&self, future: F) -> ExecutorResult<TaskHandle<F::Output>>
    where
        F: core::future::Future + Send + 'static,
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
    fn spawn_blocking<F, R>(&self, func: F) -> ExecutorResult<TaskHandle<R>>
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
    ) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static;
        
    /// Spawn a task on the current thread's local queue for better locality
    /// (inspired by Tokio's spawn_local)
    fn spawn_local<T>(&self, task: T) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + 'static,
    {
        // Default implementation falls back to regular spawn
        // Executors can override for better locality
        self.spawn(task)
    }
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
    fn cancel_task(&self, id: TaskId) -> ExecutorResult<()>;

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

    /// Wait for a task to complete.
    /// 
    /// Returns a future that resolves when the task completes or the timeout expires.
    /// This enables async/await patterns for task coordination.
    /// 
    /// # Arguments
    /// - `id`: The task ID to wait for
    /// - `timeout`: Optional timeout duration
    /// 
    /// # Returns
    /// A future that resolves to:
    /// - `Ok(())` when the task completes successfully
    /// - `Err(TaskError::Timeout)` if the timeout expires
    /// - `Err(TaskError::NotFound)` if the task doesn't exist
    /// 
    /// # Performance
    /// - Immediate return: < 10ns if already complete
    /// - Waiting overhead: Event-driven, no busy polling
    /// - Memory: Minimal waker chain overhead
    fn wait_for_task(&self, id: TaskId, timeout: Option<core::time::Duration>) -> impl core::future::Future<Output = ExecutorResult<()>> + Send;

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
        F: core::future::Future;

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

/// Executor statistics (placeholder for when metrics feature is disabled)
#[cfg(not(feature = "metrics"))]
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats;

/// Executor statistics with full metrics
#[cfg(feature = "metrics")]
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    /// Number of tasks executed
    pub tasks_executed: u64,
    /// Number of tasks in queue
    pub tasks_queued: usize,
    /// Average task execution time
    pub avg_execution_time_ns: u64,
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
    pub thread_name_prefix: String,
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
    fn initialize(&mut self) -> ExecutorResult<()> {
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

/// Builder for creating executors with custom configuration.
pub struct ExecutorBuilder {
    config: ExecutorConfig,
    #[allow(dead_code)]
    plugins: Vec<Box<dyn ExecutorPlugin>>,
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
            plugins: Vec::new(),
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
    pub fn thread_name_prefix(mut self, prefix: impl Into<String>) -> Self {
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