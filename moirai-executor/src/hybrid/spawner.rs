use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use moirai_core::{
    error::ExecutorResult,
    executor::TaskSpawner,
    task::{Task, TaskHandle},
    Priority,
};

use super::{async_state::AsyncFutureState, HybridExecutor, MetricsRef};
use crate::schedule::{BlockingTask, SyncTask, WorkScheduler};

impl<S: WorkScheduler> HybridExecutor<S> {
    /// `Task`-typed adapter over the canonical closure-based
    /// [`HybridExecutor::spawn_result`] path, shared by `spawn` and
    /// `spawn_with_priority`: the task executes inside the same
    /// cancellation-aware scheduled job with panic containment, lifecycle
    /// timing, and result publication.
    fn spawn_task_job<T>(
        &self,
        task: T,
        priority: Priority,
        locality_hint: Option<usize>,
    ) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
        T::Output: Send + 'static,
    {
        self.spawn_result::<SyncTask, _>(priority, locality_hint, move || task.execute())
    }
}

impl<S: WorkScheduler> TaskSpawner for HybridExecutor<S> {
    fn spawn<T>(&self, task: T) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
        T::Output: Send + 'static,
    {
        let priority = task.context().priority;
        self.spawn_task_job(task, priority, None)
    }

    fn spawn_async<F>(&self, future: F) -> ExecutorResult<TaskHandle<F::Output>>
    where
        F: core::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (task_id, lifecycle) = self.register_task(Priority::Normal)?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let state = AsyncFutureState::new(
            self.scheduler.clone(),
            future,
            lifecycle,
            result_sender,
            Arc::clone(&self.metrics),
        );
        Arc::clone(&state).schedule()?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }

    fn spawn_blocking<F, R>(&self, func: F) -> ExecutorResult<TaskHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // No caller-supplied priority exists on this surface, so the spawn
        // records the default priority rather than a fabricated one.
        self.spawn_result::<BlockingTask, _>(Priority::Normal, None, func)
    }

    fn spawn_detached<F>(&self, func: F) -> ExecutorResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        // No result is collected, so no `TaskHandle::new_pending` and therefore
        // no `Arc<TaskResultSlot>` heap allocation or atomic refcount — the win
        // over routing through `spawn_result`. Lifecycle tracking and metrics
        // are preserved so shutdown drain and counters stay accurate.
        let (_task_id, lifecycle) = self.register_task(Priority::Normal)?;
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |worker_id| {
                let Some(running) = lifecycle.start_unless_cancelled(worker_id) else {
                    metrics.get().record_task_cancelled();
                    return;
                };
                // Catch here (not only at the job level) so `complete()` runs and
                // the executor-level completed/failed metric is recorded, matching
                // `send_task_result`.
                match catch_unwind(AssertUnwindSafe(func)) {
                    Ok(()) => {
                        metrics.get().record_task_completed(running.complete());
                    }
                    Err(_) => {
                        running.complete();
                        metrics.get().record_task_failed();
                    }
                }
            })?;

        self.metrics.record_task_spawned();
        Ok(())
    }

    fn spawn_with_priority<T>(
        &self,
        task: T,
        priority: Priority,
        locality_hint: Option<usize>,
    ) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
        T::Output: Send + 'static,
    {
        self.spawn_task_job(task, priority, locality_hint)
    }
}
