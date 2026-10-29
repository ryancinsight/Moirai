use std::sync::Arc;
use std::{future::Future, pin::Pin};

use moirai_core::{
    error::{ExecutorError, ExecutorResult, TaskError, TaskErrorKind},
    executor::{TaskManager, TaskStats, TaskStatus},
    task::TaskId,
};

use super::HybridExecutor;
use crate::registry::{CancelOutcome, TaskRegistry, TaskWaiter};
use crate::schedule::WorkScheduler;
use crate::task::TaskMetadata;

/// Derive the observable [`TaskStatus`] from registry metadata.
fn status_of(metadata: &TaskMetadata) -> TaskStatus {
    if metadata.cancelled {
        TaskStatus::Cancelled
    } else if metadata.completed_at.is_some() {
        TaskStatus::Completed
    } else if metadata.started_at.is_some() {
        TaskStatus::Running
    } else {
        TaskStatus::Queued
    }
}

/// Build the owned completion wait around a caller-supplied deadline future.
///
/// Production supplies the PAL timer. Tests supply an event-driven future so
/// expiry and cleanup do not depend on wall-clock progress.
pub(super) fn completion_wait<F, E>(
    registry: Arc<TaskRegistry>,
    id: TaskId,
    mut deadline: Option<F>,
) -> impl Future<Output = ExecutorResult<()>> + Send
where
    F: Future<Output = Result<(), E>> + Send + Unpin,
{
    let mut waiter: Option<TaskWaiter> = None;

    std::future::poll_fn(move |context| {
        match registry.completion(id.0) {
            Some(true) => {
                waiter.take();
                deadline.take();
                return std::task::Poll::Ready(Ok(()));
            }
            Some(false) => {}
            None => {
                waiter.take();
                deadline.take();
                return std::task::Poll::Ready(Err(ExecutorError::SpawnFailed(
                    TaskError::InvalidOperation,
                )));
            }
        }

        let deadline_poll = deadline
            .as_mut()
            .map(|deadline| Pin::new(deadline).poll(context));
        match deadline_poll {
            Some(std::task::Poll::Ready(Ok(()))) => {
                waiter.take();
                deadline.take();
                return std::task::Poll::Ready(Err(ExecutorError::SpawnFailed(TaskError::Timeout)));
            }
            Some(std::task::Poll::Ready(Err(_))) => {
                waiter.take();
                deadline.take();
                return std::task::Poll::Ready(Err(ExecutorError::SpawnFailed(
                    TaskError::ExecutionFailed(TaskErrorKind::Io),
                )));
            }
            Some(std::task::Poll::Pending) | None => {}
        }

        let registration_live = if let Some(waiter) = waiter.as_mut() {
            waiter.update(context.waker())
        } else {
            waiter = registry.register_waker(id.0, context.waker());
            waiter.is_some()
        };
        if !registration_live {
            waiter.take();
        }

        // Completion publishes before closing the waiter set. This recheck
        // covers completion racing either initial registration or an update.
        match registry.completion(id.0) {
            Some(true) => {
                waiter.take();
                deadline.take();
                std::task::Poll::Ready(Ok(()))
            }
            Some(false) => std::task::Poll::Pending,
            None => {
                waiter.take();
                deadline.take();
                std::task::Poll::Ready(Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation)))
            }
        }
    })
}

impl<S: WorkScheduler> TaskManager for HybridExecutor<S> {
    /// Cooperative cancellation.
    ///
    /// Contract: a queued task that has not started skips its body when a
    /// worker dequeues it — the task completes with `TaskError::Cancelled` and
    /// status `Cancelled`. A running synchronous closure is not preempted and
    /// runs to completion. An async task receives cancellation at its next
    /// poll boundary. Cancelling an already-completed task is a no-op `Ok(())`.
    /// An unknown task ID is an error.
    fn cancel_task(&self, id: TaskId) -> ExecutorResult<()> {
        match self.task_registry.request_cancel(id.0) {
            Some(CancelOutcome::Requested | CancelOutcome::AlreadyCompleted) => {
                self.async_tasks.cancel(id.0)
            }
            None => Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation)),
        }
    }

    fn task_status(&self, id: TaskId) -> Option<TaskStatus> {
        self.task_registry
            .get_metadata(id.0)
            .map(|metadata| status_of(&metadata))
    }

    /// Event-driven completion wait.
    ///
    /// The future registers a completion waker with the task registry and
    /// returns `Pending`; `mark_completed`/`mark_cancelled` wake it exactly
    /// (no polling loop and no thread-blocking sleep). A positive timeout owns
    /// one PAL timer, which wakes the same task when its deadline expires.
    /// Completion is observed before the deadline on every poll, so a task
    /// already published as complete wins a simultaneous timeout observation.
    fn wait_for_task(
        &self,
        id: TaskId,
        timeout: Option<core::time::Duration>,
    ) -> impl core::future::Future<Output = ExecutorResult<()>> + Send {
        completion_wait(
            Arc::clone(&self.task_registry),
            id,
            timeout.map(moirai_pal::timer::sleep),
        )
    }

    /// Statistics limited to what the executor actually tracks.
    ///
    /// `priority` is the value recorded at spawn; timing fields come from the
    /// registry's lifecycle timestamps.
    fn task_stats(&self, id: TaskId) -> Option<TaskStats> {
        self.task_registry
            .get_metadata(id.0)
            .map(|metadata| TaskStats {
                id,
                priority: metadata.priority,
                status: status_of(&metadata),
                spawn_time: metadata.created_at,
                start_time: metadata.started_at,
                completion_time: metadata.completed_at,
                cpu_time_ns: metadata
                    .execution_duration()
                    .map_or(0, |duration| duration.as_nanos() as u64),
            })
    }
}
