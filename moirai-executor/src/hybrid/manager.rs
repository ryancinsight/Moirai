use std::sync::Arc;
use std::{future::Future, pin::Pin};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
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

pin_project_lite::pin_project! {
    #[project = DeadlineStateProj]
    #[project_replace = DeadlineStateProjReplace]
    enum DeadlineState<F> {
        Inactive,
        Running {
            #[pin]
            future: F,
        },
        Failed { source: std::io::Error },
    }
}

pin_project_lite::pin_project! {
    /// Owns a completion registration and its caller-supplied deadline future.
    pub(super) struct CompletionWait<F> {
        registry: Arc<TaskRegistry>,
        id: TaskId,
        #[pin]
        deadline: DeadlineState<F>,
        waiter: Option<TaskWaiter>,
    }
}

fn release_resources<F>(mut deadline: Pin<&mut DeadlineState<F>>, waiter: &mut Option<TaskWaiter>) {
    waiter.take();
    deadline.set(DeadlineState::Inactive);
}

impl<F> Future for CompletionWait<F>
where
    F: Future<Output = std::io::Result<()>> + Send,
{
    type Output = ExecutorResult<()>;

    fn poll(
        self: Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut this = self.project();
        let id = *this.id;

        match this.registry.completion(id.0) {
            Some(true) => {
                release_resources(this.deadline.as_mut(), this.waiter);
                return std::task::Poll::Ready(Ok(()));
            }
            Some(false) => {}
            None => {
                release_resources(this.deadline.as_mut(), this.waiter);
                return std::task::Poll::Ready(Err(ExecutorError::UnknownTask { id }));
            }
        }

        if matches!(
            this.deadline.as_ref().get_ref(),
            DeadlineState::Failed { .. }
        ) {
            let source = match this
                .deadline
                .as_mut()
                .project_replace(DeadlineState::Inactive)
            {
                DeadlineStateProjReplace::Failed { source } => source,
                DeadlineStateProjReplace::Inactive | DeadlineStateProjReplace::Running { .. } => {
                    unreachable!("invariant: a construction failure remains present until consumed")
                }
            };
            release_resources(this.deadline.as_mut(), this.waiter);
            return std::task::Poll::Ready(Err(ExecutorError::TimerFailed { id, source }));
        }

        let deadline_poll = match this.deadline.as_mut().project() {
            DeadlineStateProj::Inactive => None,
            DeadlineStateProj::Running { future } => Some(future.poll(context)),
            DeadlineStateProj::Failed { .. } => {
                unreachable!("invariant: a construction failure is consumed before polling")
            }
        };
        match deadline_poll {
            Some(std::task::Poll::Ready(Ok(()))) => {
                release_resources(this.deadline.as_mut(), this.waiter);
                return std::task::Poll::Ready(Err(ExecutorError::WaitTimedOut { id }));
            }
            Some(std::task::Poll::Ready(Err(source))) => {
                release_resources(this.deadline.as_mut(), this.waiter);
                return std::task::Poll::Ready(Err(ExecutorError::TimerFailed { id, source }));
            }
            Some(std::task::Poll::Pending) | None => {}
        }

        let registration_live = if let Some(waiter) = this.waiter.as_mut() {
            waiter.update(context.waker())
        } else {
            *this.waiter = this.registry.register_waker(id.0, context.waker());
            this.waiter.is_some()
        };
        if !registration_live {
            this.waiter.take();
        }

        // Completion publishes before closing the waiter set. This recheck
        // covers completion racing either initial registration or an update.
        match this.registry.completion(id.0) {
            Some(true) => {
                release_resources(this.deadline.as_mut(), this.waiter);
                std::task::Poll::Ready(Ok(()))
            }
            Some(false) => std::task::Poll::Pending,
            None => {
                release_resources(this.deadline.as_mut(), this.waiter);
                std::task::Poll::Ready(Err(ExecutorError::UnknownTask { id }))
            }
        }
    }
}

pub(super) fn completion_wait<F>(
    registry: Arc<TaskRegistry>,
    id: TaskId,
    deadline: Option<std::io::Result<F>>,
) -> CompletionWait<F>
where
    F: Future<Output = std::io::Result<()>> + Send,
{
    let deadline = match deadline {
        Some(Ok(future)) => DeadlineState::Running { future },
        Some(Err(source)) => DeadlineState::Failed { source },
        None => DeadlineState::Inactive,
    };
    CompletionWait {
        registry,
        id,
        deadline,
        waiter: None,
    }
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
            None => Err(ExecutorError::UnknownTask { id }),
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
        let deadline = timeout.map(moirai_pal::timer::sleep);
        completion_wait(Arc::clone(&self.task_registry), id, deadline)
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
