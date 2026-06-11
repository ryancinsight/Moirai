//! Task management and status tracking.

use crate::error::ExecutorResult;
use crate::platform::Instant;
use crate::{Priority, TaskId};

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
/// - Status queries: O(1) lookup time, < 50ns typical latency
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
    fn wait_for_task(
        &self,
        id: TaskId,
        timeout: Option<core::time::Duration>,
    ) -> impl core::future::Future<Output = ExecutorResult<()>> + Send;

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
        matches!(
            self.status,
            TaskStatus::Completed | TaskStatus::Cancelled | TaskStatus::Failed
        )
    }
}
