//! Executor trait definitions and configuration.

use crate::{Task, AsyncTask, TaskHandle, TaskId, Priority};
use core::future::Future;

/// The core trait for task executors in the Moirai runtime.
pub trait Executor: Send + Sync + 'static {
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

    /// Block the current thread until all tasks complete.
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future;

    /// Attempt to run tasks without blocking.
    fn try_run(&self) -> bool;

    /// Shutdown the executor gracefully.
    fn shutdown(&self);

    /// Check if the executor is shutting down.
    fn is_shutting_down(&self) -> bool;

    /// Get the number of worker threads.
    fn worker_count(&self) -> usize;

    /// Get the current load (number of pending tasks).
    fn load(&self) -> usize;
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
    pub thread_name_prefix: String,
    /// Whether to enable NUMA awareness
    pub numa_aware: bool,
    /// Stack size for worker threads (bytes)
    pub stack_size: Option<usize>,
    /// Whether to enable metrics collection
    pub enable_metrics: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus(),
            async_threads: (num_cpus() / 4).max(1),
            max_global_queue_size: 1024,
            max_local_queue_size: 256,
            work_stealing: WorkStealingConfig::default(),
            thread_name_prefix: "moirai-worker".to_string(),
            numa_aware: false,
            stack_size: None,
            enable_metrics: cfg!(feature = "metrics"),
        }
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
    pub fn thread_name_prefix(mut self, prefix: String) -> Self {
        self.config.thread_name_prefix = prefix;
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
}

impl ExecutorStats {
    /// Calculate the steal success rate.
    pub fn steal_success_rate(&self) -> f64 {
        let total_attempts = self.successful_steals + self.failed_steals;
        if total_attempts == 0 {
            0.0
        } else {
            self.successful_steals as f64 / total_attempts as f64
        }
    }

    /// Calculate the task completion rate.
    pub fn task_completion_rate(&self) -> f64 {
        if self.tasks_spawned == 0 {
            0.0
        } else {
            self.tasks_executed as f64 / self.tasks_spawned as f64
        }
    }

    /// Calculate thread utilization.
    pub fn thread_utilization(&self) -> f64 {
        let total_threads = self.active_threads + self.idle_threads;
        if total_threads == 0 {
            0.0
        } else {
            self.active_threads as f64 / total_threads as f64
        }
    }
}

/// A trait for objects that can provide executor statistics.
pub trait ExecutorMetrics {
    /// Get current executor statistics.
    fn stats(&self) -> ExecutorStats;

    /// Reset statistics counters.
    fn reset_stats(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert!(config.worker_threads > 0);
        assert!(config.async_threads > 0);
        assert!(config.max_global_queue_size > 0);
        assert!(config.max_local_queue_size > 0);
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
    }
}