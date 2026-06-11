//! Task spawning interface.

use crate::error::ExecutorResult;
use crate::{Priority, Task, TaskHandle};

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
