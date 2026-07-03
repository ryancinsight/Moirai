use core::future::Future;
use core::pin::Pin;

use super::id_and_context::TaskContext;
use super::traits::Task;

/// A future adapter that executes a [`Task`] on first poll.
///
/// # Execution semantics
///
/// The wrapped task runs **synchronously inside the first `poll` call** on the
/// polling thread — this adapter does not offload work to an executor, does
/// not yield mid-task, and never returns `Poll::Pending`. It exists so a
/// synchronous task can be awaited from async code; the `.await` completes in
/// one poll, blocking the async worker for the task's full duration. Offload
/// long-running tasks to a blocking pool instead of awaiting them directly on
/// an async executor.
///
/// # Fused
///
/// Like [`core::future::Ready`] and other std-convention one-shot futures,
/// polling again after completion is a contract violation and panics.
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

    /// Executes the task synchronously and returns `Poll::Ready` on the first
    /// call; see the type-level docs for the blocking semantics.
    ///
    /// # Panics
    /// Panics if polled again after it has returned `Poll::Ready` (fused
    /// one-shot contract; a completed future has no result to hand out and no
    /// waker-based path by which `Pending` could ever resolve).
    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        let task = self
            .get_mut()
            .task
            .take()
            .expect("TaskFuture polled after completion");
        core::task::Poll::Ready(task.execute())
    }
}
