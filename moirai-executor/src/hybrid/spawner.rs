use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use moirai_core::{
    error::ExecutorResult,
    executor::TaskSpawner,
    task::{Task, TaskHandle},
    Priority,
};

use super::{async_state::AsyncFutureState, send_task_result, HybridExecutor, MetricsRef};
use crate::schedule::{BlockingTask, SyncTask};

impl TaskSpawner for HybridExecutor {
    fn spawn<T>(&self, task: T) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
        T::Output: Send + 'static,
    {
        let priority = task.context().priority;
        let (task_id, lifecycle) = self.register_task()?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<SyncTask, _>(priority, None, move |worker_id| {
                let running = lifecycle.start(worker_id);
                let result = catch_unwind(AssertUnwindSafe(|| task.execute()));
                let execution_time = running.complete();
                send_task_result(result, result_sender, metrics.get(), execution_time);
            })?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }

    fn spawn_async<F>(&self, future: F) -> ExecutorResult<TaskHandle<F::Output>>
    where
        F: core::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (task_id, lifecycle) = self.register_task()?;

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
        let (task_id, lifecycle) = self.register_task()?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |worker_id| {
                let running = lifecycle.start(worker_id);
                let result = catch_unwind(AssertUnwindSafe(func));
                let execution_time = running.complete();
                send_task_result(result, result_sender, metrics.get(), execution_time);
            })?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }

    fn spawn_detached<F>(&self, func: F) -> ExecutorResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        // No result is collected, so no `TaskHandle::new_pending` and therefore
        // no `Arc<TaskResultSlot>` heap allocation or atomic refcount — the win
        // over the default routing through `spawn_blocking`. Lifecycle tracking
        // and metrics are preserved so shutdown drain and counters stay accurate.
        let (_task_id, lifecycle) = self.register_task()?;
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |worker_id| {
                let running = lifecycle.start(worker_id);
                // Catch here (not only at the job level) so `complete()` runs and
                // the executor-level completed/failed metric is recorded, matching
                // `send_task_result`.
                let outcome = catch_unwind(AssertUnwindSafe(func));
                let execution_time = running.complete();
                match outcome {
                    Ok(()) => metrics.get().record_task_completed(execution_time),
                    Err(_) => metrics.get().record_task_failed(),
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
        let (task_id, lifecycle) = self.register_task()?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<SyncTask, _>(priority, locality_hint, move |worker_id| {
                let running = lifecycle.start(worker_id);
                let result = catch_unwind(AssertUnwindSafe(|| task.execute()));
                let execution_time = running.complete();
                send_task_result(result, result_sender, metrics.get(), execution_time);
            })?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }
}
