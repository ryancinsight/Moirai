use super::builder::Mapped;
use super::id_and_context::TaskContext;
use super::traits::Task;

/// Extension methods for tasks.
pub trait TaskExt: Task + Sized {
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
