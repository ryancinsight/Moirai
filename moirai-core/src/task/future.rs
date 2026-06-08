use core::future::Future;
use core::pin::Pin;

use super::id_and_context::TaskContext;
use super::traits::Task;

/// A future that can be awaited to get the result of a task.
#[allow(clippy::module_name_repetitions)]
pub struct TaskFuture<T> {
    task: Option<T>,
    context: TaskContext,
}

impl<T> TaskFuture<T>
where
    T: Task,
{
    /// Create a new task future.
    pub fn new(task: T, context: TaskContext) -> Self {
        Self {
            task: Some(task),
            context,
        }
    }

    /// Get the task context.
    pub fn context(&self) -> &TaskContext {
        &self.context
    }
}

impl<T> Future for TaskFuture<T>
where
    T: Task + Unpin,
{
    type Output = T::Output;

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // Get a mutable reference to the task option
        let task_opt = &mut self.get_mut().task;

        match task_opt.take() {
            Some(task) => core::task::Poll::Ready(task.execute()),
            None => core::task::Poll::Pending, // Task already executed
        }
    }
}
