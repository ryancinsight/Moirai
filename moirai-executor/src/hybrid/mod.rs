//! Main hybrid executor implementation.
//!
//! `HybridExecutor` exposes one public execution surface while delegating sync,
//! async, and blocking work to one scheduler facade. Sync and async-ready work
//! use the compute worker pool; blocking work uses the facade's bounded lane.
//! The work-shape choice is encoded by zero-sized marker types in
//! `crate::schedule`.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult, TaskError},
    executor::ExecutorConfig,
    task::{TaskHandle, TaskId, TaskResultSender},
    Priority,
};

use crate::{
    metrics::ExecutorMetrics,
    registry::{SchedulerStateLease, TaskLifecycleToken, TaskRegistry},
    schedule::{SchedulerScope, SyncTask, ThreadScheduler, WorkClass, WorkScheduler},
};

mod async_state;
pub(crate) mod control;
pub(crate) mod manager;
pub(crate) mod spawner;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
struct MetricsRef {
    metrics: NonNull<ExecutorMetrics>,
}

// Safety: `MetricsRef` points at `HybridExecutor.metrics`. Construction retains
// the metrics Arc in the scheduler's lifetime owner, and each scheduled job
// retains scheduler state until its final metrics access completes.
unsafe impl Send for MetricsRef {}

impl MetricsRef {
    #[inline]
    fn new(metrics: &Arc<ExecutorMetrics>) -> Self {
        Self {
            metrics: NonNull::from(metrics.as_ref()),
        }
    }

    #[inline]
    fn get(self) -> &'static ExecutorMetrics {
        // Safety: see the `Send` impl invariant. The returned reference is used
        // only inside scheduled jobs that complete before executor destruction.
        unsafe { self.metrics.as_ref() }
    }
}

/// Main hybrid executor that coordinates sync, async, and blocking tasks.
///
/// Generic over the work-stealing runtime `S` behind the
/// [`WorkScheduler`] seam; the default
/// [`ThreadScheduler`] backs the production runtime, while the parameter lets a
/// substitute (e.g. a single-threaded `wasm32` scheduler) be plugged in without
/// touching this façade.
pub struct HybridExecutor<S: WorkScheduler = ThreadScheduler> {
    config: ExecutorConfig,
    scheduler: S,
    task_registry: Arc<Mutex<TaskRegistry>>,
    metrics: Arc<ExecutorMetrics>,
    shutdown_signal: Arc<AtomicBool>,
}

impl HybridExecutor<ThreadScheduler> {
    /// Create a new hybrid executor with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::InvalidConfiguration`] when the configured
    /// global admission bound cannot supply at least two slots per worker,
    /// [`ExecutorError::InvalidLocalQueueInitialCapacity`] when local capacity
    /// cannot normalize or form the required deque allocation layouts, or
    /// propagates scheduler construction failures.
    pub fn new(config: ExecutorConfig) -> ExecutorResult<Self> {
        let numa_aware = {
            #[cfg(feature = "numa")]
            {
                config.numa_aware
            }
            #[cfg(not(feature = "numa"))]
            {
                true
            }
        };
        let scheduler = ThreadScheduler::from_executor_config(&config, numa_aware)?;
        let task_registry = Arc::new(Mutex::new(TaskRegistry::new()));
        let metrics = Arc::new(ExecutorMetrics::new());
        scheduler.retain_lifetime_owner((Arc::clone(&task_registry), Arc::clone(&metrics)));
        metrics.update_worker_counts(0, scheduler.worker_count(), scheduler.worker_count());

        Ok(Self {
            config,
            scheduler,
            task_registry,
            metrics,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Run a scoped fan-out directly on the unified scheduler.
    ///
    /// This path is for completion-only work that does not require per-task
    /// result handles or lifecycle metadata. It preserves borrowing semantics
    /// by waiting for all spawned jobs before returning.
    ///
    /// `scope` is inherent to the default [`ThreadScheduler`] backing because its
    /// signature exposes a concrete [`SchedulerScope`] borrow handle, which is
    /// outside the substitutable [`WorkScheduler`]
    /// seam.
    pub fn scope<'scope, C, F>(&'scope self, body: F) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(&SchedulerScope<'scope, C>) -> ExecutorResult<()>,
    {
        self.scheduler.scope::<C, _>(Priority::Normal, None, body)
    }
}

impl<S: WorkScheduler> HybridExecutor<S> {
    /// Get executor configuration.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Shutdown the executor gracefully.
    pub fn shutdown(&mut self) -> ExecutorResult<()> {
        self.shutdown_signal.store(true, Ordering::Release);
        self.scheduler.shutdown();
        Ok(())
    }

    /// Get executor metrics.
    pub fn metrics(&self) -> &ExecutorMetrics {
        self.refresh_scheduler_metrics();
        &self.metrics
    }

    /// Submit an untyped synchronous job.
    pub fn submit_task<F>(&self, task: F) -> ExecutorResult<TaskId>
    where
        F: FnOnce() + Send + 'static,
    {
        self.spawn_result::<SyncTask, _>(Priority::Normal, None, task)
            .map(|handle| handle.id())
    }

    /// Canonical result-producing spawn path shared by every closure-based
    /// spawn surface (`spawn_blocking`, `submit_task`, and — via a
    /// task-executing sibling in `spawner` — the `Task`-typed surfaces).
    ///
    /// Registers the task at `priority`, allocates its pending handle, and
    /// schedules one job that honors queued-task cancellation, contains
    /// panics, records lifecycle timing, and publishes the result.
    pub(super) fn spawn_result<C, R>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        func: impl FnOnce() -> R + Send + 'static,
    ) -> ExecutorResult<TaskHandle<R>>
    where
        C: WorkClass,
        R: Send + 'static,
    {
        let (task_id, lifecycle) = self.register_scheduled_task(priority)?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<C, _>(priority, locality_hint, move |worker_id| {
                let Some(running) = lifecycle.start_unless_cancelled(worker_id) else {
                    // Record before publishing the result so a joiner observes
                    // the cancelled counter as soon as the handle resolves.
                    metrics.get().record_task_cancelled();
                    result_sender.send(Err(TaskError::Cancelled));
                    return;
                };
                let result = catch_unwind(AssertUnwindSafe(func));
                let execution_time = running.complete();
                send_task_result(result, result_sender, metrics.get(), execution_time);
            })?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }

    /// Run indexed work in worker-sized chunks on the unified scheduler.
    ///
    /// This path avoids per-item task handles and lifecycle metadata when the
    /// caller only needs completion for a bounded index domain.
    pub fn for_each_indexed<'scope, C, F>(&'scope self, count: usize, task: F) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: Fn(usize) + Send + Sync + 'scope,
    {
        self.scheduler
            .for_each_indexed::<C, _>(Priority::Normal, None, count, task)
    }

    /// Run indexed map/reduce in worker-sized chunks on the unified scheduler.
    pub fn map_reduce_indexed<'scope, C, T, Map, Reduce>(
        &'scope self,
        count: usize,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> ExecutorResult<T>
    where
        C: WorkClass,
        T: Send + Clone + 'scope,
        Map: Fn(usize) -> T + Send + Sync + 'scope,
        Reduce: Fn(T, T) -> T + Send + Sync + 'scope,
    {
        self.scheduler.map_reduce_indexed::<C, _, _, _>(
            Priority::Normal,
            None,
            count,
            identity,
            map,
            reduce,
        )
    }

    /// Get the number of active workers.
    pub fn active_workers(&self) -> usize {
        self.scheduler.active_workers()
    }

    /// Get the total number of workers.
    pub fn total_workers(&self) -> usize {
        self.scheduler.worker_count()
    }

    /// Get pending task count across all workers.
    pub fn pending_tasks(&self) -> usize {
        self.scheduler.pending_tasks()
    }

    /// Returns true when queued or active scheduler work exists.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Wait until queued and active scheduler work completes without shutting down workers.
    pub fn join(&self) -> ExecutorResult<()> {
        self.scheduler.join()?;
        self.refresh_scheduler_metrics();
        Ok(())
    }

    fn register_scheduled_task(
        &self,
        priority: Priority,
    ) -> ExecutorResult<(TaskId, TaskLifecycleToken<SchedulerStateLease>)> {
        let mut registry = self.task_registry.lock().map_err(|_| {
            ExecutorError::ResourceExhausted("task registry lock poisoned".to_string())
        })?;
        // SAFETY: synchronous and blocking lifecycle tokens move only into
        // scheduler-owned jobs. Construction installs the registry and metrics
        // as the scheduler's lifetime owner; each worker holds scheduler state
        // until its current job returns, including re-entrant destruction.
        let (task_id, lifecycle) = unsafe { registry.register_next_scheduled_task() };
        lifecycle.set_priority(priority);
        Ok((TaskId::new(task_id), lifecycle))
    }

    fn refresh_scheduler_metrics(&self) {
        let snapshot = self.scheduler.metrics();
        self.metrics.update_worker_counts(
            snapshot.active_workers,
            snapshot
                .worker_count
                .saturating_sub(snapshot.active_workers),
            snapshot.worker_count,
        );
        self.metrics.update_queue_metrics(snapshot.pending_tasks);
    }
}

impl<S: WorkScheduler> Drop for HybridExecutor<S> {
    fn drop(&mut self) {
        let _ = Self::shutdown(self);
    }
}

#[inline]
fn send_task_result<T>(
    result: Result<T, Box<dyn std::any::Any + Send>>,
    sender: TaskResultSender<T>,
    metrics: &ExecutorMetrics,
    execution_time: core::time::Duration,
) where
    T: Send + 'static,
{
    match result {
        Ok(value) => {
            sender.send(Ok(value));
            metrics.record_task_completed(execution_time);
        }
        Err(_) => {
            sender.send(Err(TaskError::Panicked));
            metrics.record_task_failed();
        }
    }
}
