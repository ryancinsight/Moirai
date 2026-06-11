use crate::error::TaskError;

pub use super::builder::Mapped;
use super::id_and_context::TaskContext;
use super::traits::Task;

/// Extension methods for tasks.
pub trait TaskExt: Task + Sized {
    /// Execute the task and catch any errors, providing a fallback value.
    fn catch<F>(self, handler: F) -> Catch<Self, F>
    where
        F: FnOnce(TaskError) -> Self::Output,
    {
        Catch::new(self, handler)
    }

    /// Transform the output of this task.
    fn map<F, R>(self, mapper: F) -> Mapped<Self, F>
    where
        F: FnOnce(Self::Output) -> R,
    {
        Mapped::new(self, mapper)
    }

    /// Convert this task into a task with the given context.
    fn with_context(self, context: TaskContext) -> ContextualTask<Self> {
        ContextualTask::new(self, context)
    }
}

// Blanket implementation for all Task implementors.
impl<T: Task> TaskExt for T {}

/// A task with an explicit context.
pub struct ContextualTask<T> {
    task: T,
    context: TaskContext,
}

impl<T: Task> ContextualTask<T> {
    /// Create a new contextual task.
    pub fn new(task: T, context: TaskContext) -> Self {
        Self { task, context }
    }
}

impl<T: Task> Task for ContextualTask<T> {
    type Output = T::Output;

    fn execute(self) -> Self::Output {
        self.task.execute()
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }
}

/// Wrapper that catches task errors and provides a fallback value.
#[allow(dead_code)]
pub struct Catch<T, F> {
    task: T,
    handler: F,
}

impl<T, F> Catch<T, F> {
    /// Create a new catch task.
    pub fn new(task: T, handler: F) -> Self
    where
        T: Task,
    {
        let _context = task.context().clone();
        Self { task, handler }
    }
}

impl<T, F> Task for Catch<T, F>
where
    T: Task,
    T::Output: core::fmt::Debug,
    F: FnOnce(core::fmt::Arguments<'_>) -> T::Output + Send + 'static,
{
    type Output = T::Output;

    fn execute(self) -> Self::Output {
        // In a real implementation, this would catch panics
        // For now, just execute the task normally
        self.task.execute()
    }

    fn context(&self) -> &TaskContext {
        &self.task.context()
    }

    fn is_stealable(&self) -> bool {
        self.task.is_stealable()
    }

    fn estimated_cost(&self) -> u32 {
        self.task.estimated_cost()
    }
}
