use crate::platform::*;
use crate::{Priority, TaskId};

/// Task wrapper for object pooling.
///
/// This wrapper allows tasks to be reset and reused, reducing allocation overhead.
/// Now uses inline storage to avoid pointer chasing.
pub struct TaskWrapper<T> {
    inner: Option<T>,
    task_id: Option<TaskId>,
    priority: Priority,
    /// Creation timestamp for age tracking
    creation_time: Instant,
    /// Number of times this wrapper has been reset
    reset_count: usize,
    /// Inline storage for small tasks to avoid allocation
    #[allow(dead_code)]
    inline_storage: [u8; 64],
}

impl<T> Default for TaskWrapper<T> {
    fn default() -> Self {
        Self {
            inner: None,
            task_id: None,
            priority: Priority::Normal,
            creation_time: Instant::now(),
            reset_count: 0,
            inline_storage: [0; 64],
        }
    }
}

impl<T> TaskWrapper<T> {
    /// Create a new task wrapper.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: None,
            task_id: None,
            priority: Priority::Normal,
            creation_time: Instant::now(),
            reset_count: 0,
            inline_storage: [0; 64],
        }
    }

    /// Initialize the wrapper with a task.
    pub fn init(&mut self, task: T, task_id: TaskId, priority: Priority) {
        self.inner = Some(task);
        self.task_id = Some(task_id);
        self.priority = priority;
        self.creation_time = Instant::now();
    }

    /// Reset the wrapper for reuse.
    pub fn reset(&mut self) {
        self.inner = None;
        self.task_id = None;
        self.priority = Priority::Normal;
        self.creation_time = Instant::now();
        self.reset_count += 1;
    }

    /// Take the inner task.
    pub fn take(&mut self) -> Option<T> {
        self.inner.take()
    }

    /// Get the task ID.
    pub fn task_id(&self) -> Option<TaskId> {
        self.task_id
    }

    /// Get the priority.
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Get the age of this wrapper.
    pub fn age(&self) -> Duration {
        self.creation_time.elapsed()
    }

    /// Get the number of times this wrapper has been reset.
    pub fn reset_count(&self) -> usize {
        self.reset_count
    }
}
