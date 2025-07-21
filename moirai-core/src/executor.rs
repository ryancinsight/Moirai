//! Executor trait definitions and configuration.

use crate::{Task, AsyncTask, TaskHandle, TaskId, Priority};
use core::future::Future;

/// Core task spawning capabilities.
pub trait TaskSpawner: Send + Sync + 'static {
    /// Spawn a task for execution.
    fn spawn<T>(&self, task: T) -> TaskHandle<T::Output>
    where
        T: Task;

    /// Spawn an async task for execution.
    fn spawn_async<F>(&self, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static;

    /// Spawn a blocking task that may block the current thread.
    fn spawn_blocking<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;

    /// Spawn a task with a specific priority.
    fn spawn_with_priority<T>(&self, task: T, priority: Priority) -> TaskHandle<T::Output>
    where
        T: Task;
}

/// Task management and monitoring capabilities.
pub trait TaskManager: Send + Sync + 'static {
    /// Cancel a running task.
    /// 
    /// # Behavior Guarantees
    /// - Cancellation is cooperative and may not be immediate
    /// - Returns Ok(()) if cancellation was requested successfully
    /// - Task may still complete normally if already finishing
    fn cancel_task(&self, id: TaskId) -> Result<(), crate::error::TaskError>;

    /// Get the current status of a task.
    /// 
    /// # Performance Characteristics
    /// - O(1) lookup time
    /// - Non-blocking operation
    fn task_status(&self, id: TaskId) -> Option<TaskStatus>;

    /// Wait for a task to complete with optional timeout.
    /// 
    /// # Behavior Guarantees
    /// - Returns immediately if task is already complete
    /// - Respects timeout if specified
    async fn wait_for_task(&self, id: TaskId, timeout: Option<core::time::Duration>) -> Result<(), crate::error::TaskError>;

    /// Get statistics about task execution.
    fn task_stats(&self, id: TaskId) -> Option<TaskStats>;
}

/// Executor lifecycle and control operations.
pub trait ExecutorControl: Send + Sync + 'static {
    /// Block the current thread until the future completes.
    /// 
    /// # Performance Characteristics
    /// - Optimal for CPU-bound futures
    /// - May block the calling thread
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future;

    /// Attempt to run tasks without blocking.
    /// 
    /// # Behavior Guarantees
    /// - Non-blocking operation
    /// - Returns true if any work was performed
    /// 
    /// # Performance Characteristics
    /// - O(1) operation
    /// - Suitable for event loops
    fn try_run(&self) -> bool;

    /// Shutdown the executor gracefully.
    /// 
    /// # Behavior Guarantees
    /// - Allows running tasks to complete
    /// - Prevents new tasks from being spawned
    /// - Idempotent operation
    fn shutdown(&self);

    /// Shutdown the executor with a timeout.
    /// 
    /// # Behavior Guarantees
    /// - Attempts graceful shutdown first
    /// - Forces termination after timeout
    /// - May result in task cancellation
    fn shutdown_timeout(&self, timeout: core::time::Duration);

    /// Check if the executor is shutting down.
    /// 
    /// # Performance Characteristics
    /// - O(1) operation
    /// - Lock-free read
    fn is_shutting_down(&self) -> bool;
}

/// Executor information and metrics.
pub trait ExecutorInfo: Send + Sync + 'static {
    /// Get the number of worker threads.
    /// 
    /// # Performance Characteristics
    /// - O(1) operation
    /// - Configuration-time constant
    fn worker_count(&self) -> usize;

    /// Get the current load (number of pending tasks).
    /// 
    /// # Performance Characteristics
    /// - O(1) amortized
    /// - May involve atomic operations
    fn load(&self) -> usize;

    /// Get detailed executor statistics.
    #[cfg(feature = "metrics")]
    fn stats(&self) -> ExecutorStats;

    /// Get the executor configuration.
    fn config(&self) -> &ExecutorConfig;
}

/// Plugin system for extending executor functionality.
pub trait ExecutorPlugin: Send + Sync + 'static {
    /// Configure the executor before startup.
    /// 
    /// # Behavior Guarantees
    /// - Called once during executor initialization
    /// - Configuration changes take effect immediately
    fn configure(&self, config: &mut ExecutorConfig);

    /// Called when a task is spawned.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting spawn performance
    /// - Called on the spawning thread
    fn on_task_spawn(&self, task_id: TaskId, priority: Priority) {
        let _ = (task_id, priority); // Default: no-op
    }

    /// Called when a task completes.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting completion performance
    /// - Called on the executing thread
    fn on_task_complete(&self, task_id: TaskId, result: &Result<(), crate::error::TaskError>) {
        let _ = (task_id, result); // Default: no-op
    }

    /// Called when a task is cancelled.
    fn on_task_cancel(&self, task_id: TaskId) {
        let _ = task_id; // Default: no-op
    }

    /// Called during executor shutdown.
    fn on_shutdown(&self) {
        // Default: no-op
    }
}

/// The main executor trait that combines all capabilities.
/// 
/// # Design Philosophy
/// 
/// This trait serves as a convenience trait that combines all executor
/// capabilities. Implementations can choose to implement the individual
/// traits directly for more granular control.
/// 
/// # Performance Characteristics
/// 
/// - Task spawn: O(1) amortized
/// - Task cancellation: O(1) lookup + cooperative cancellation
/// - Shutdown: O(n) where n is the number of running tasks
pub trait Executor: TaskSpawner + TaskManager + ExecutorControl + ExecutorInfo {
    /// The configuration type for this executor.
    type Config: ExecutorConfigTrait;
    
    /// The handle type returned when starting this executor.
    type Handle: ExecutorHandle;

    /// Add a plugin to this executor.
    /// 
    /// # Behavior Guarantees
    /// - Plugins are applied in the order they are added
    /// - Plugin configuration happens during executor creation
    fn with_plugin<P: ExecutorPlugin>(self, plugin: P) -> Self;
}

/// Trait for executor configurations.
pub trait ExecutorConfigTrait: Send + Sync + Clone {
    /// Validate the configuration.
    fn validate(&self) -> Result<(), crate::error::ExecutorError>;

    /// Get the number of worker threads.
    fn worker_threads(&self) -> usize;

    /// Get the number of async threads.
    fn async_threads(&self) -> usize;
}

/// Trait for executor handles.
pub trait ExecutorHandle: Send + Sync + 'static {
    /// Wait for the executor to shut down.
    /// 
    /// # Behavior Guarantees
    /// - Blocks until all tasks complete and executor shuts down
    /// - Returns immediately if executor is already shut down
    fn join(self);

    /// Wait for shutdown with a timeout.
    /// 
    /// # Behavior Guarantees
    /// - Returns Ok(()) if shutdown completes within timeout
    /// - Returns Err if timeout expires
    fn join_timeout(self, timeout: core::time::Duration) -> Result<(), crate::error::ExecutorError>;
}

/// Status of a task in the executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is queued and waiting to be executed
    Pending,
    /// Task is currently executing
    Running,
    /// Task completed successfully
    Completed,
    /// Task was cancelled
    Cancelled,
    /// Task failed with an error
    Failed,
}

/// Statistics about a specific task.
#[derive(Debug, Clone)]
pub struct TaskStats {
    /// When the task was spawned
    pub spawn_time: core::time::Instant,
    /// When the task started executing (if it has started)
    pub start_time: Option<core::time::Instant>,
    /// When the task completed (if it has completed)
    pub completion_time: Option<core::time::Instant>,
    /// Current status
    pub status: TaskStatus,
    /// Priority level
    pub priority: Priority,
    /// Estimated computational cost
    pub estimated_cost: u32,
    /// Actual CPU time used (if available)
    pub cpu_time: Option<core::time::Duration>,
}

impl TaskStats {
    /// Calculate the time spent waiting in queue.
    pub fn queue_time(&self) -> Option<core::time::Duration> {
        self.start_time.map(|start| start.duration_since(self.spawn_time))
    }

    /// Calculate the execution time.
    pub fn execution_time(&self) -> Option<core::time::Duration> {
        match (self.start_time, self.completion_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    /// Calculate the total time from spawn to completion.
    pub fn total_time(&self) -> Option<core::time::Duration> {
        self.completion_time.map(|end| end.duration_since(self.spawn_time))
    }
}

/// Configuration for the executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Number of worker threads for parallel tasks
    pub worker_threads: usize,
    /// Number of threads dedicated to async tasks
    pub async_threads: usize,
    /// Maximum number of tasks in the global queue
    pub max_global_queue_size: usize,
    /// Maximum number of tasks in per-thread queues
    pub max_local_queue_size: usize,
    /// Work stealing strategy configuration
    pub work_stealing: WorkStealingConfig,
    /// Thread naming prefix
    pub thread_name_prefix: alloc::string::String,
    /// Whether to enable NUMA awareness
    pub numa_aware: bool,
    /// Stack size for worker threads (bytes)
    pub stack_size: Option<usize>,
    /// Whether to enable metrics collection
    pub enable_metrics: bool,
    /// Plugins to load
    pub plugins: alloc::vec::Vec<alloc::boxed::Box<dyn ExecutorPlugin>>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus(),
            async_threads: (num_cpus() / 4).max(1),
            max_global_queue_size: 1024,
            max_local_queue_size: 256,
            work_stealing: WorkStealingConfig::default(),
            thread_name_prefix: alloc::string::String::from("moirai-worker"),
            numa_aware: false,
            stack_size: None,
            enable_metrics: cfg!(feature = "metrics"),
            plugins: alloc::vec::Vec::new(),
        }
    }
}

impl ExecutorConfigTrait for ExecutorConfig {
    fn validate(&self) -> Result<(), crate::error::ExecutorError> {
        if self.worker_threads == 0 {
            return Err(crate::error::ExecutorError::InvalidConfiguration);
        }
        if self.async_threads == 0 {
            return Err(crate::error::ExecutorError::InvalidConfiguration);
        }
        if self.max_global_queue_size == 0 || self.max_local_queue_size == 0 {
            return Err(crate::error::ExecutorError::InvalidConfiguration);
        }
        Ok(())
    }

    fn worker_threads(&self) -> usize {
        self.worker_threads
    }

    fn async_threads(&self) -> usize {
        self.async_threads
    }
}

/// Configuration for work stealing behavior.
#[derive(Debug, Clone)]
pub struct WorkStealingConfig {
    /// Maximum number of tasks to steal at once
    pub max_steal_batch: usize,
    /// Number of steal attempts before giving up
    pub max_steal_attempts: usize,
    /// Whether to use random victim selection
    pub random_victim_selection: bool,
    /// Minimum tasks in queue before allowing steals
    pub steal_threshold: usize,
}

impl Default for WorkStealingConfig {
    fn default() -> Self {
        Self {
            max_steal_batch: 16,
            max_steal_attempts: 3,
            random_victim_selection: true,
            steal_threshold: 2,
        }
    }
}

/// Builder for creating executor configurations.
pub struct ExecutorBuilder {
    config: ExecutorConfig,
}

impl ExecutorBuilder {
    /// Create a new executor builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
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

    /// Configure work stealing behavior.
    pub fn work_stealing(mut self, config: WorkStealingConfig) -> Self {
        self.config.work_stealing = config;
        self
    }

    /// Set the thread name prefix.
    pub fn thread_name_prefix(mut self, prefix: impl Into<alloc::string::String>) -> Self {
        self.config.thread_name_prefix = prefix.into();
        self
    }

    /// Enable or disable NUMA awareness.
    pub fn numa_aware(mut self, enabled: bool) -> Self {
        self.config.numa_aware = enabled;
        self
    }

    /// Set the stack size for worker threads.
    pub fn stack_size(mut self, size: usize) -> Self {
        self.config.stack_size = Some(size);
        self
    }

    /// Enable or disable metrics collection.
    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Add a plugin to the executor.
    pub fn with_plugin<P: ExecutorPlugin>(mut self, plugin: P) -> Self {
        self.config.plugins.push(alloc::boxed::Box::new(plugin));
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ExecutorConfig {
        self.config
    }
}

impl Default for ExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the number of logical CPU cores.
fn num_cpus() -> usize {
    // In a real implementation, this would detect the actual CPU count
    // For now, we'll use a reasonable default
    4
}

/// Statistics about executor performance.
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    /// Total number of tasks executed
    pub tasks_executed: u64,
    /// Total number of tasks spawned
    pub tasks_spawned: u64,
    /// Number of successful work steals
    pub successful_steals: u64,
    /// Number of failed work steal attempts
    pub failed_steals: u64,
    /// Average task execution time (microseconds)
    pub avg_task_execution_time: f64,
    /// Current number of active threads
    pub active_threads: usize,
    /// Current number of idle threads
    pub idle_threads: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Total CPU time used (microseconds)
    pub total_cpu_time: u64,
}

impl ExecutorStats {
    /// Calculate the steal success rate.
    /// 
    /// # Performance Characteristics
    /// - O(1) calculation
    /// 
    /// # Returns
    /// - Value between 0.0 and 1.0
    /// - 0.0 if no steal attempts have been made
    pub fn steal_success_rate(&self) -> f64 {
        let total_attempts = self.successful_steals + self.failed_steals;
        if total_attempts == 0 {
            0.0
        } else {
            self.successful_steals as f64 / total_attempts as f64
        }
    }

    /// Calculate the task completion rate.
    /// 
    /// # Performance Characteristics
    /// - O(1) calculation
    /// 
    /// # Returns
    /// - Value between 0.0 and 1.0
    /// - 0.0 if no tasks have been spawned
    pub fn task_completion_rate(&self) -> f64 {
        if self.tasks_spawned == 0 {
            0.0
        } else {
            self.tasks_executed as f64 / self.tasks_spawned as f64
        }
    }

    /// Calculate thread utilization.
    /// 
    /// # Performance Characteristics
    /// - O(1) calculation
    /// 
    /// # Returns
    /// - Value between 0.0 and 1.0
    /// - 0.0 if no threads are configured
    pub fn thread_utilization(&self) -> f64 {
        let total_threads = self.active_threads + self.idle_threads;
        if total_threads == 0 {
            0.0
        } else {
            self.active_threads as f64 / total_threads as f64
        }
    }

    /// Calculate tasks per second throughput.
    /// 
    /// # Parameters
    /// - `elapsed_seconds`: Time period to calculate throughput over
    /// 
    /// # Returns
    /// - Tasks per second
    /// - 0.0 if elapsed_seconds <= 0
    pub fn throughput(&self, elapsed_seconds: f64) -> f64 {
        if elapsed_seconds <= 0.0 {
            0.0
        } else {
            self.tasks_executed as f64 / elapsed_seconds
        }
    }
}

/// A trait for objects that can provide executor statistics.
pub trait ExecutorMetrics {
    /// Get current executor statistics.
    /// 
    /// # Performance Characteristics
    /// - O(1) operation for cached metrics
    /// - May involve atomic reads
    fn stats(&self) -> ExecutorStats;

    /// Reset statistics counters.
    /// 
    /// # Behavior Guarantees
    /// - Atomically resets all counters to zero
    /// - Does not affect running tasks
    fn reset_stats(&self);

    /// Get real-time metrics for monitoring.
    /// 
    /// # Performance Characteristics
    /// - O(1) operation
    /// - Suitable for high-frequency monitoring
    fn realtime_metrics(&self) -> RealtimeMetrics;
}

/// Real-time metrics for monitoring.
#[derive(Debug, Clone)]
pub struct RealtimeMetrics {
    /// Current queue lengths per worker
    pub queue_lengths: alloc::vec::Vec<usize>,
    /// Current CPU usage per thread
    pub cpu_usage: alloc::vec::Vec<f64>,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Number of tasks spawned in the last second
    pub recent_spawn_rate: f64,
    /// Number of tasks completed in the last second
    pub recent_completion_rate: f64,
}

// We need to add these imports for the alloc types
extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_config_validation() {
        let mut config = ExecutorConfig::default();
        assert!(config.validate().is_ok());

        config.worker_threads = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_executor_builder() {
        let config = ExecutorBuilder::new()
            .worker_threads(8)
            .async_threads(2)
            .max_global_queue_size(2048)
            .numa_aware(true)
            .build();

        assert_eq!(config.worker_threads, 8);
        assert_eq!(config.async_threads, 2);
        assert_eq!(config.max_global_queue_size, 2048);
        assert!(config.numa_aware);
    }

    #[test]
    fn test_executor_stats() {
        let mut stats = ExecutorStats::default();
        stats.successful_steals = 80;
        stats.failed_steals = 20;
        stats.tasks_executed = 900;
        stats.tasks_spawned = 1000;
        stats.active_threads = 6;
        stats.idle_threads = 2;

        assert_eq!(stats.steal_success_rate(), 0.8);
        assert_eq!(stats.task_completion_rate(), 0.9);
        assert_eq!(stats.thread_utilization(), 0.75);
        assert_eq!(stats.throughput(10.0), 90.0);
    }

    #[test]
    fn test_task_stats() {
        let spawn_time = core::time::Instant::now();
        let start_time = spawn_time + core::time::Duration::from_millis(10);
        let completion_time = start_time + core::time::Duration::from_millis(50);

        let stats = TaskStats {
            spawn_time,
            start_time: Some(start_time),
            completion_time: Some(completion_time),
            status: TaskStatus::Completed,
            priority: Priority::Normal,
            estimated_cost: 100,
            cpu_time: Some(core::time::Duration::from_millis(45)),
        };

        assert_eq!(stats.queue_time(), Some(core::time::Duration::from_millis(10)));
        assert_eq!(stats.execution_time(), Some(core::time::Duration::from_millis(50)));
        assert_eq!(stats.total_time(), Some(core::time::Duration::from_millis(60)));
    }
}