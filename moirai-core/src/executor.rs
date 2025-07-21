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
    /// Spawn a task for execution.
    /// 
    /// # Behavior Guarantees
    /// - Returns immediately without blocking
    /// - Task is queued for execution on available worker
    /// - Respects task priority for scheduling order
    /// - Task context is preserved throughout execution
    /// 
    /// # Performance Characteristics
    /// - Latency: < 100ns for local tasks
    /// - Memory: ~64 bytes per task overhead
    /// - Scalability: Linear up to 128 cores
    fn spawn<T>(&self, task: T) -> TaskHandle<T::Output>
    where
        T: Task;

    /// Spawn an async task for execution.
    /// 
    /// # Behavior Guarantees
    /// - Future is polled on async executor thread pool
    /// - Supports cooperative cancellation via task handles
    /// - Preserves async context and waker chains
    /// - Automatically handles Future trait requirements
    /// 
    /// # Performance Characteristics
    /// - Spawn latency: < 200ns for async tasks
    /// - Polling overhead: < 50ns per poll
    /// - Memory: Future size + 64 bytes overhead
    fn spawn_async<F>(&self, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static;

    /// Spawn a blocking task that may block the current thread.
    /// 
    /// # Behavior Guarantees
    /// - Executes on dedicated blocking thread pool
    /// - Does not block async executor threads
    /// - Supports cancellation via task handles
    /// - Thread stack size configurable via ExecutorConfig
    /// 
    /// # Performance Characteristics
    /// - Spawn latency: < 500ns (thread pool coordination)
    /// - Thread overhead: ~8MB stack per blocking thread
    /// - Scalability: Configurable blocking thread pool size
    fn spawn_blocking<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    /// Spawn a task with a specific priority.
    /// 
    /// # Behavior Guarantees
    /// - Higher priority tasks are scheduled before lower priority
    /// - Priority inversion protection via priority inheritance
    /// - Maintains fairness within same priority level
    /// - Critical priority tasks can preempt lower priority tasks
    /// 
    /// # Performance Characteristics
    /// - Priority scheduling overhead: < 10ns additional cost
    /// - Queue operations: O(1) for most priorities, O(log n) for priority queue
    fn spawn_with_priority<T>(&self, task: T, priority: Priority) -> TaskHandle<T::Output>
    where
        T: Task;
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
    /// Cancel a running task.
    /// 
    /// # Behavior Guarantees
    /// - Cancellation is cooperative and may not be immediate
    /// - Returns Ok(()) if cancellation was requested successfully
    /// - Task may still complete normally if already finishing
    /// - Cancellation is idempotent - safe to call multiple times
    /// - Memory ordering: Release semantics for cancellation flag
    /// 
    /// # Performance Characteristics
    /// - Latency: < 50ns for cancellation request
    /// - Memory: Single atomic flag per task
    /// - Thread safety: Lock-free cancellation signaling
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

/// Executor lifecycle and control operations.
/// 
/// This trait provides high-level executor control operations.
/// It follows the Command pattern for executor lifecycle management.
/// 
/// # Behavior Guarantees
/// - Shutdown operations are idempotent and thread-safe
/// - Graceful shutdown allows running tasks to complete
/// - Forced shutdown may terminate tasks abruptly
/// - Load balancing is automatic and transparent
/// 
/// # Performance Characteristics
/// - Control operations: < 1μs latency
/// - Shutdown time: Depends on running task completion
/// - Load reporting: O(1) atomic operations
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
    /// - Graceful phase: Same as shutdown()
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
    /// Calculate the total execution time.
    /// 
    /// # Returns
    /// - Some(duration) if task has started and completed
    /// - None if task hasn't started or is still running
    /// 
    /// # Accuracy
    /// - Resolution: Nanosecond precision
    /// - Clock: Monotonic system clock
    pub fn execution_time(&self) -> Option<core::time::Duration> {
        match (self.start_time, self.completion_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    /// Calculate the time spent waiting in queue.
    /// 
    /// # Returns
    /// - Some(duration) if task has started
    /// - Some(duration_since_spawn) if task is still queued
    /// 
    /// # Accuracy
    /// - Includes scheduler overhead and load balancing time
    /// - May include time spent in priority queues
    pub fn queue_time(&self) -> Option<core::time::Duration> {
        match self.start_time {
            Some(start) => Some(start.duration_since(self.spawn_time)),
            None => Some(Instant::now().duration_since(self.spawn_time)),
        }
    }

    /// Check if the task is currently active.
    /// 
    /// # Definition
    /// Active tasks are either queued or currently running.
    /// Completed, cancelled, or failed tasks are not active.
    pub fn is_active(&self) -> bool {
        matches!(self.status, TaskStatus::Queued | TaskStatus::Running)
    }

    /// Check if the task has completed (successfully or not).
    /// 
    /// # Definition
    /// Finished tasks have reached a terminal state and will
    /// not transition to any other state.
    pub fn is_finished(&self) -> bool {
        matches!(self.status, TaskStatus::Completed | TaskStatus::Cancelled | TaskStatus::Failed)
    }
}

/// Configuration for executor behavior.
#[derive(Debug, Clone)]
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
    /// # Range: 1 second to 1 hour
    pub task_retention_duration: core::time::Duration,
    
    /// How often to run the cleanup process
    /// 
    /// # Default: 30 seconds  
    /// # Range: 1 second to task_retention_duration
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

/// Plugin trait for extending executor functionality.
pub trait ExecutorPlugin: Send + Sync + 'static {
    /// Initialize the plugin.
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

/// Builder for creating executor configurations.
pub struct ExecutorBuilder {
    config: ExecutorConfig,
    plugins: alloc::vec::Vec<alloc::boxed::Box<dyn ExecutorPlugin>>,
}

impl ExecutorBuilder {
    /// Create a new executor builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
            plugins: alloc::vec::Vec::new(),
        }
    }

    /// Set the number of worker threads.
    pub fn worker_threads(mut self, count: usize) -> Self {
        self.config.worker_threads = count;
        self
    }

    /// Set the number of async threads.
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

    /// Set the thread name prefix.
    pub fn thread_name_prefix(mut self, prefix: impl Into<alloc::string::String>) -> Self {
        self.config.thread_name_prefix = prefix.into();
        self
    }

    /// Enable or disable NUMA awareness.
    #[cfg(feature = "numa")]
    pub fn numa_aware(mut self, enabled: bool) -> Self {
        self.config.numa_aware = enabled;
        self
    }

    /// Enable or disable metrics collection.
    #[cfg(feature = "metrics")]
    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Set preemption configuration.
    pub fn preemption_config(mut self, config: PreemptionConfig) -> Self {
        self.config.preemption = config;
        self
    }

    /// Set memory configuration.
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.config.memory = config;
        self
    }

    /// Set cleanup configuration.
    pub fn cleanup_config(mut self, config: CleanupConfig) -> Self {
        self.config.cleanup = config;
        self
    }

    /// Add a plugin to the executor.
    pub fn plugin(mut self, plugin: impl ExecutorPlugin) -> Self {
        self.plugins.push(alloc::boxed::Box::new(plugin));
        self
    }

    /// Build the executor configuration.
    pub fn build_config(self) -> (ExecutorConfig, alloc::vec::Vec<alloc::boxed::Box<dyn ExecutorPlugin>>) {
        (self.config, self.plugins)
    }
}

impl Default for ExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Overall executor statistics.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    /// Worker thread statistics
    pub worker_stats: alloc::vec::Vec<WorkerStats>,
    /// Global queue statistics
    pub global_queue_stats: QueueStats,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// Task execution statistics
    pub task_stats: TaskExecutionStats,
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
            .map(|n| n.get())
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