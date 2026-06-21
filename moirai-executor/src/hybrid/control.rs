use std::sync::atomic::Ordering;

use moirai_core::executor::{Executor, ExecutorControl, ExecutorStats};

use super::HybridExecutor;

impl ExecutorControl for HybridExecutor {
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: core::future::Future,
    {
        crate::schedule::wake::block_on_current_thread(future)
    }

    fn try_run(&self) -> bool {
        self.refresh_scheduler_metrics();
        self.scheduler.has_work()
    }

    fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Release);
        self.scheduler.shutdown();
    }

    fn shutdown_timeout(&self, _timeout: core::time::Duration) {
        self.shutdown();
    }

    fn is_shutting_down(&self) -> bool {
        self.shutdown_signal.load(Ordering::Acquire)
    }

    fn worker_count(&self) -> usize {
        self.scheduler.worker_count()
    }

    fn load(&self) -> usize {
        self.scheduler.pending_tasks()
    }
}

impl Executor for HybridExecutor {
    #[cfg(feature = "metrics")]
    fn stats(&self) -> ExecutorStats {
        self.refresh_scheduler_metrics();
        ExecutorStats {
            tasks_executed: self.metrics.tasks_completed.load(Ordering::Acquire),
            tasks_queued: self.scheduler.pending_tasks(),
            avg_execution_time_ns: self.metrics.average_task_duration().as_nanos() as u64,
        }
    }
}
