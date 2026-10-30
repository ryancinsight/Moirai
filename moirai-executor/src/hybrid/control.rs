use std::sync::atomic::Ordering;

#[cfg(feature = "metrics")]
use moirai_core::executor::ExecutorStats;
use moirai_core::executor::{Executor, ExecutorControl};

use super::HybridExecutor;
use crate::schedule::WorkScheduler;

impl<S: WorkScheduler> HybridExecutor<S> {
    pub(super) fn request_shutdown(&self) {
        self.shutdown
            .request(&self.admission, self.scheduler.is_current_worker(), None);
    }
}

impl<S: WorkScheduler> ExecutorControl for HybridExecutor<S> {
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: core::future::Future,
    {
        // This local future driver cannot reject shutdown through its generic
        // `F::Output`, so it remains outside facade admission.
        crate::schedule::wake::block_on_current_thread(future)
    }

    fn try_run(&self) -> bool {
        self.refresh_scheduler_metrics();
        self.scheduler.has_work()
    }

    fn shutdown(&self) {
        self.request_shutdown();
    }

    /// Graceful shutdown bounded by `timeout` for the *caller*.
    ///
    /// This call returns once the shared drain completes or `timeout` elapses,
    /// whichever comes first. A lapsed deadline does not stop the drain.
    fn shutdown_timeout(&self, timeout: core::time::Duration) {
        let started = std::time::Instant::now();
        self.shutdown.request(
            &self.admission,
            self.scheduler.is_current_worker(),
            Some((started, timeout)),
        );
    }

    fn is_shutting_down(&self) -> bool {
        self.admission.is_closed()
    }

    fn worker_count(&self) -> usize {
        self.scheduler.worker_count()
    }

    fn load(&self) -> usize {
        self.scheduler.pending_tasks()
    }
}

impl<S: WorkScheduler> Executor for HybridExecutor<S> {
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
