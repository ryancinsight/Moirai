//! Main hybrid executor implementation.
//!
//! `HybridExecutor` exposes one public execution surface while delegating sync,
//! async, and blocking work to the same thread scheduler. The work-shape choice
//! is encoded by zero-sized marker types in `crate::schedule`.

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
    task::{TaskId, TaskResultSender},
    Priority,
};

use crate::{
    metrics::ExecutorMetrics,
    registry::{TaskLifecycleToken, TaskRegistry},
    schedule::{SchedulerScope, SyncTask, ThreadScheduler, WorkClass},
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

// Safety: `MetricsRef` points at `HybridExecutor.metrics`. The executor owns
// the scheduler and drains scheduled jobs during shutdown/drop before dropping
// the metrics allocation, so scheduled synchronous/blocking jobs cannot observe
// a dangling metrics pointer.
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
pub struct HybridExecutor {
    config: ExecutorConfig,
    scheduler: ThreadScheduler,
    task_registry: Arc<Mutex<TaskRegistry>>,
    metrics: Arc<ExecutorMetrics>,
    shutdown_signal: Arc<AtomicBool>,
}

impl HybridExecutor {
    /// Create a new hybrid executor with the given configuration.
    pub fn new(config: ExecutorConfig) -> ExecutorResult<Self> {
        let scheduler = ThreadScheduler::new(config.worker_threads, &config.thread_name_prefix)?;
        let metrics = Arc::new(ExecutorMetrics::new());
        metrics.update_worker_counts(0, scheduler.worker_count(), scheduler.worker_count());

        Ok(Self {
            config,
            scheduler,
            task_registry: Arc::new(Mutex::new(TaskRegistry::new())),
            metrics,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Get executor configuration.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Start the executor.
    pub fn start(&mut self) -> ExecutorResult<()> {
        Ok(())
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
        let (task_id, lifecycle) = self.register_task()?;

        let metrics = MetricsRef::new(&self.metrics);
        self.scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |worker_id| {
                let running = lifecycle.start(worker_id);
                let succeeded = catch_unwind(AssertUnwindSafe(task)).is_ok();
                if succeeded {
                    metrics.get().record_task_completed(running.complete());
                } else {
                    running.complete();
                    metrics.get().record_task_failed();
                }
            })?;

        self.metrics.record_task_spawned();
        Ok(task_id)
    }

    /// Run a scoped fan-out directly on the unified scheduler.
    ///
    /// This path is for completion-only work that does not require per-task
    /// result handles or lifecycle metadata. It preserves borrowing semantics
    /// by waiting for all spawned jobs before returning.
    pub fn scope<'scope, C, F>(&'scope self, body: F) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(&SchedulerScope<'scope, C>) -> ExecutorResult<()>,
    {
        self.scheduler.scope::<C, _>(Priority::Normal, None, body)
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

    fn register_task(&self) -> ExecutorResult<(TaskId, TaskLifecycleToken)> {
        let mut registry = self.task_registry.lock().map_err(|_| {
            ExecutorError::ResourceExhausted("task registry lock poisoned".to_string())
        })?;
        let (task_id, lifecycle) = registry.register_next_task();
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

impl Drop for HybridExecutor {
    fn drop(&mut self) {
        let _ = HybridExecutor::shutdown(self);
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
