use std::sync::atomic::Ordering;

#[cfg(feature = "metrics")]
use moirai_core::executor::ExecutorStats;
use moirai_core::executor::{Executor, ExecutorControl};

use super::HybridExecutor;
use crate::schedule::WorkScheduler;

impl<S: WorkScheduler> ExecutorControl for HybridExecutor<S> {
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

    /// Graceful shutdown bounded by `timeout` for the *caller*.
    ///
    /// The drain runs on a helper thread; this call returns once the drain
    /// completes or `timeout` elapses, whichever comes first. If the deadline
    /// lapses, workers keep draining in the background and the (idempotent)
    /// scheduler shutdown is re-joined on executor drop.
    fn shutdown_timeout(&self, timeout: core::time::Duration) {
        self.shutdown_signal.store(true, Ordering::Release);

        let scheduler = self.scheduler.clone();
        let (done_sender, done_receiver) = std::sync::mpsc::sync_channel::<()>(1);
        std::thread::spawn(move || {
            scheduler.shutdown();
            let _ = done_sender.send(());
        });

        // A timeout here is the expected bounded outcome, not a failure to
        // mask: the drain continues in the background by contract (above).
        let _ = done_receiver.recv_timeout(timeout);
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
