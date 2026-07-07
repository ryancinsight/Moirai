use std::sync::{Arc, Mutex};

use moirai_core::{
    error::{ExecutorError, ExecutorResult, TaskError},
    executor::{TaskManager, TaskStats, TaskStatus},
    task::TaskId,
};

use super::HybridExecutor;
use crate::registry::{CancelOutcome, TaskRegistry};
use crate::schedule::WorkScheduler;
use crate::task::TaskMetadata;

fn lock_registry(
    registry: &Mutex<TaskRegistry>,
) -> ExecutorResult<std::sync::MutexGuard<'_, TaskRegistry>> {
    registry
        .lock()
        .map_err(|_| ExecutorError::ResourceExhausted("task registry lock poisoned".to_string()))
}

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

impl<S: WorkScheduler> TaskManager for HybridExecutor<S> {
    /// Cooperative cancellation.
    ///
    /// Contract: a queued task that has not started skips its body when a
    /// worker dequeues it — the task completes with `TaskError::Cancelled` and
    /// status `Cancelled`. A task that already started is **not** preempted; it
    /// runs to completion and this call still returns `Ok(())` (the request is
    /// recorded but has no effect). Cancelling an already-completed task is a
    /// no-op `Ok(())`. An unknown task ID is an error.
    fn cancel_task(&self, id: TaskId) -> ExecutorResult<()> {
        let registry = lock_registry(&self.task_registry)?;
        match registry.request_cancel(id.0) {
            Some(CancelOutcome::Requested | CancelOutcome::AlreadyCompleted) => Ok(()),
            None => Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation)),
        }
    }

    fn task_status(&self, id: TaskId) -> Option<TaskStatus> {
        let registry = self.task_registry.lock().ok()?;
        registry
            .get_metadata(id.0)
            .map(|metadata| status_of(&metadata))
    }

    /// Event-driven completion wait.
    ///
    /// The future registers a completion waker with the task registry and
    /// returns `Pending`; `mark_completed`/`mark_cancelled` wake it exactly
    /// (no polling loop and no thread-blocking sleep). The deadline is checked
    /// on every poll — at creation, on each completion wake, and on any
    /// external poll after expiry. No in-scope timer exists (the async timer
    /// lives in `moirai-async`), so if the task never completes, observing the
    /// expiry requires the caller to poll after the deadline (e.g. a
    /// timeout-aware runtime); a completion wake always resolves promptly.
    fn wait_for_task(
        &self,
        id: TaskId,
        timeout: Option<core::time::Duration>,
    ) -> impl core::future::Future<Output = ExecutorResult<()>> + Send {
        let registry = Arc::clone(&self.task_registry);
        let deadline = timeout.and_then(|timeout| std::time::Instant::now().checked_add(timeout));

        std::future::poll_fn(move |context| {
            let registry = match lock_registry(&registry) {
                Ok(registry) => registry,
                Err(error) => return std::task::Poll::Ready(Err(error)),
            };

            if registry.is_completed(id.0) {
                return std::task::Poll::Ready(Ok(()));
            }
            if registry.get_metadata(id.0).is_none() {
                return std::task::Poll::Ready(Err(ExecutorError::SpawnFailed(
                    TaskError::InvalidOperation,
                )));
            }
            if deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline) {
                return std::task::Poll::Ready(Err(ExecutorError::SpawnFailed(TaskError::Timeout)));
            }

            registry.register_waker(id.0, context.waker());
            // Re-check after registration: completion publishes the timestamp
            // before taking the waker, so a completion that raced ahead of the
            // registration is visible here and must not be lost.
            if registry.is_completed(id.0) {
                return std::task::Poll::Ready(Ok(()));
            }

            std::task::Poll::Pending
        })
    }

    /// Statistics limited to what the executor actually tracks.
    ///
    /// `priority` is the value recorded at spawn; timing fields come from the
    /// registry's lifecycle timestamps.
    fn task_stats(&self, id: TaskId) -> Option<TaskStats> {
        let registry = self.task_registry.lock().ok()?;
        registry.get_metadata(id.0).map(|metadata| TaskStats {
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
