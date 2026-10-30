use core::convert::Infallible;
use std::sync::{Arc, atomic::Ordering, mpsc};

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
        // This local future driver cannot reject shutdown through its generic
        // `F::Output`, so it remains outside facade admission.
        crate::schedule::wake::block_on_current_thread(future)
    }

    fn try_run(&self) -> bool {
        self.refresh_scheduler_metrics();
        self.scheduler.has_work()
    }

    fn shutdown(&self) {
        self.admission.close();
        self.async_tasks.shutdown(&self.task_registry);
        self.scheduler.shutdown();
    }

    /// Graceful shutdown bounded by `timeout` for the *caller*.
    ///
    /// The drain runs on a helper thread; this call returns once the drain
    /// completes or `timeout` elapses, whichever comes first. If the deadline
    /// lapses, workers keep draining in the background and the (idempotent)
    /// scheduler shutdown is re-joined on executor drop.
    fn shutdown_timeout(&self, timeout: core::time::Duration) {
        let wait_started = std::time::Instant::now();
        self.admission.close();

        let async_tasks = Arc::clone(&self.async_tasks);
        let task_registry = Arc::clone(&self.task_registry);
        let scheduler = self.scheduler.clone();
        let (done_sender, done_receiver) = mpsc::channel::<Infallible>();
        drop(std::thread::spawn(move || {
            let completion = done_sender;
            async_tasks.shutdown(&task_registry);
            scheduler.shutdown();
            drop(completion);
        }));

        let remaining = timeout.saturating_sub(wait_started.elapsed());
        match done_receiver.recv_timeout(remaining) {
            Ok(completion) => match completion {},
            Err(mpsc::RecvTimeoutError::Disconnected | mpsc::RecvTimeoutError::Timeout) => {}
        }
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
