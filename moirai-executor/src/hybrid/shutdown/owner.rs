use super::{Admission, Context};
use crate::{
    hybrid::async_state::AsyncTasks, metrics::ExecutorMetrics, registry::TaskRegistry,
    schedule::WorkScheduler,
};
use moirai_core::error::{ExecutorError, ExecutorResult};
#[cfg(test)]
use std::io;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    sync::{Arc, Condvar, Mutex},
    thread,
    time::{Duration, Instant},
};
#[derive(Clone)]
pub(in crate::hybrid) struct Shutdown {
    completion: Arc<Completion>,
}
#[derive(Default)]
struct Completion {
    complete: Mutex<bool>,
    changed: Condvar,
    #[cfg(test)]
    waiting: AtomicUsize,
}

impl Completion {
    fn wait(&self, deadline: Option<(Instant, Duration)>) {
        let complete = self
            .complete
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        #[cfg(test)]
        let counted = !*complete;
        #[cfg(test)]
        if counted {
            self.waiting.fetch_add(1, Ordering::Relaxed);
            self.changed.notify_all();
        }
        let complete = match deadline {
            Some((started, timeout)) => {
                self.changed
                    .wait_timeout_while(
                        complete,
                        timeout.saturating_sub(started.elapsed()),
                        |complete| !*complete,
                    )
                    .unwrap_or_else(|error| error.into_inner())
                    .0
            }
            None => self
                .changed
                .wait_while(complete, |complete| !*complete)
                .unwrap_or_else(|error| error.into_inner()),
        };
        #[cfg(test)]
        if counted {
            self.waiting.fetch_sub(1, Ordering::Relaxed);
            self.changed.notify_all();
        }
        drop(complete);
    }
    fn publish(&self) {
        *self
            .complete
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = true;
        self.changed.notify_all();
    }

    #[cfg(test)]
    fn wait_for_waiters(&self, minimum: usize, timeout: Duration) -> bool {
        let complete = self
            .complete
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (complete, _) = self
            .changed
            .wait_timeout_while(complete, timeout, |_| {
                self.waiting.load(Ordering::Relaxed) < minimum
            })
            .unwrap_or_else(|error| error.into_inner());
        let observed = self.waiting.load(Ordering::Relaxed) >= minimum;
        drop(complete);
        observed
    }

    #[cfg(test)]
    fn skip_wait_mutant(&self) {
        let complete = self
            .complete
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.waiting.fetch_add(1, Ordering::Relaxed);
        self.changed.notify_all();
        self.waiting.fetch_sub(1, Ordering::Relaxed);
        self.changed.notify_all();
        drop(complete);
    }
}
impl Shutdown {
    pub(in crate::hybrid) fn new<S: WorkScheduler>(
        thread_name: String,
        scheduler: &S,
        tasks: &Arc<AsyncTasks>,
        registry: &Arc<TaskRegistry>,
        metrics: &Arc<ExecutorMetrics>,
        admission: &Arc<Admission>,
    ) -> ExecutorResult<Self> {
        Self::start(
            thread_name,
            scheduler,
            tasks,
            registry,
            metrics,
            admission,
            #[cfg(test)]
            None,
        )
    }
    fn start<S: WorkScheduler>(
        thread_name: String,
        scheduler: &S,
        tasks: &Arc<AsyncTasks>,
        registry: &Arc<TaskRegistry>,
        metrics: &Arc<ExecutorMetrics>,
        admission: &Arc<Admission>,
        #[cfg(test)] creation_failure: Option<io::Error>,
    ) -> ExecutorResult<Self> {
        let completion = Arc::new(Completion::default());
        let worker_scheduler = scheduler.clone();
        let worker_tasks = Arc::clone(tasks);
        let worker_registry = Arc::clone(registry);
        let worker_metrics = Arc::clone(metrics);
        let worker_admission = Arc::clone(admission);
        let worker_completion = Arc::clone(&completion);
        let worker = move || {
            while !worker_admission.is_closed() {
                thread::park();
            }
            Context::with(&worker_admission, || {
                worker_tasks.shutdown(&worker_registry);
                worker_scheduler.shutdown();
                while worker_admission.active_calls() != 0 || !worker_tasks.is_empty() {
                    thread::park();
                }
            });
            drop(worker_scheduler);
            drop(worker_tasks);
            drop(worker_registry);
            drop(worker_metrics);
            drop(worker_admission);
            worker_completion.publish();
        };
        #[cfg(test)]
        let created = match creation_failure {
            Some(source) => Err(source),
            None => thread::Builder::new().name(thread_name).spawn(worker),
        };
        #[cfg(not(test))]
        let created = thread::Builder::new().name(thread_name).spawn(worker);
        let handle = match created {
            Ok(handle) => handle,
            Err(source) => {
                scheduler.shutdown();
                return Err(ExecutorError::ShutdownWorkerCreationFailed { source });
            }
        };
        admission.install_worker(handle.thread().clone());
        drop(handle);
        Ok(Self { completion })
    }
    pub(in crate::hybrid) fn request(
        &self,
        admission: &Admission,
        scheduler_worker: bool,
        deadline: Option<(Instant, Duration)>,
    ) {
        admission.close();
        if !scheduler_worker && !admission.is_shutdown_worker() && !Context::contains(admission) {
            self.completion.wait(deadline);
        }
    }

    #[cfg(test)]
    pub(in crate::hybrid) fn wait_for_waiters(&self, minimum: usize, timeout: Duration) -> bool {
        self.completion.wait_for_waiters(minimum, timeout)
    }
}
#[cfg(test)]
mod tests {
    use std::sync::{Arc, mpsc};

    use super::*;
    use crate::schedule::{SyncTask, ThreadScheduler};
    use moirai_core::Priority;
    struct ReentrantCleanup {
        admission: Arc<Admission>,
        shutdown: Shutdown,
        observed: mpsc::SyncSender<bool>,
    }
    impl Drop for ReentrantCleanup {
        fn drop(&mut self) {
            self.shutdown.request(&self.admission, false, None);
            self.observed
                .send(self.admission.is_shutdown_worker())
                .expect("cleanup observer remains connected");
        }
    }
    fn resources() -> (
        Arc<AsyncTasks>,
        Arc<TaskRegistry>,
        Arc<ExecutorMetrics>,
        Arc<Admission>,
    ) {
        (
            Arc::new(AsyncTasks::default()),
            Arc::new(TaskRegistry::new()),
            Arc::new(ExecutorMetrics::new()),
            Arc::new(Admission::new()),
        )
    }
    #[test]
    fn creation_failure_joins_scheduler_and_preserves_source() {
        let scheduler = ThreadScheduler::new(1, "shutdown-rollback")
            .expect("shutdown rollback scheduler constructs");
        let (tasks, registry, metrics, admission) = resources();
        let error = Shutdown::start(
            "injected-shutdown-worker".into(),
            &scheduler,
            &tasks,
            &registry,
            &metrics,
            &admission,
            Some(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "worker denied",
            )),
        );
        let Err(error) = error else {
            panic!("injected worker creation must fail");
        };
        assert!(matches!(
            error,
            ExecutorError::ShutdownWorkerCreationFailed { ref source }
                if source.kind() == io::ErrorKind::PermissionDenied
                    && source.to_string() == "worker denied"
        ));
        assert!(matches!(
            scheduler.schedule::<SyncTask, _>(Priority::Normal, None, |_| {}),
            Err(ExecutorError::ShuttingDown)
        ));
    }
    #[test]
    fn shutdown_worker_cleanup_reentry_is_request_only() {
        let scheduler = ThreadScheduler::new(1, "shutdown-cleanup")
            .expect("shutdown cleanup scheduler constructs");
        let (tasks, registry, metrics, admission) = resources();
        let shutdown = Shutdown::new(
            "shutdown-cleanup-owner".into(),
            &scheduler,
            &tasks,
            &registry,
            &metrics,
            &admission,
        )
        .expect("shutdown worker constructs");
        let (observed_sender, observed_receiver) = mpsc::sync_channel(1);
        scheduler.retain_lifetime_owner(ReentrantCleanup {
            admission: Arc::clone(&admission),
            shutdown: shutdown.clone(),
            observed: observed_sender,
        });
        drop(scheduler);
        shutdown.request(&admission, false, None);
        assert!(
            observed_receiver
                .recv()
                .expect("shutdown worker cleanup reports its identity")
        );
    }

    #[test]
    fn waiter_witness_rejects_a_compiled_skip_wait_mutant() {
        let completion = Completion::default();
        completion.skip_wait_mutant();
        assert!(!completion.wait_for_waiters(1, Duration::ZERO));
    }
}
