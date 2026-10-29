use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
    thread,
    time::Duration,
};

use moirai_core::{
    Priority,
    error::{ExecutorError, ExecutorResult, TaskError},
    executor::ExecutorConfig,
    task::{TaskHandle, TaskId},
};

use super::super::super::HybridExecutor;
use super::super::{AsyncFutureState, AsyncTasks};
use super::fixtures::{
    AlwaysSelfWake, GatedInjector, PendingTask, WakePeerThenReady, pending_async_state,
};
use crate::metrics::ExecutorMetrics;
use crate::registry::{CancelOutcome, TaskRegistry};
use crate::schedule::{SyncTask, ThreadScheduler, WorkClass, WorkSubmit};

const EVENT_DEADLINE: Duration = Duration::from_secs(5);

struct UnpolledFuture {
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Future for UnpolledFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Poll::Pending
    }
}

impl Drop for UnpolledFuture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct RejectionGate {
    scheduler: ThreadScheduler,
    entered: mpsc::SyncSender<()>,
    release: Mutex<mpsc::Receiver<()>>,
    attempts: AtomicUsize,
}

struct RejectionFixture {
    scheduler: Arc<RejectionGate>,
    entered: mpsc::Receiver<()>,
    release: mpsc::SyncSender<()>,
}

impl RejectionFixture {
    fn new(scheduler: ThreadScheduler) -> Self {
        let (entered_sender, entered) = mpsc::sync_channel(1);
        let (release, release_receiver) = mpsc::sync_channel(1);
        let scheduler = Arc::new(RejectionGate {
            scheduler,
            entered: entered_sender,
            release: Mutex::new(release_receiver),
            attempts: AtomicUsize::new(0),
        });
        Self {
            scheduler,
            entered,
            release,
        }
    }
}

impl WorkSubmit for Arc<RejectionGate> {
    fn schedule<C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'static,
    {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        if attempt == 0 {
            self.entered
                .send(())
                .expect("admission observer remains connected");
            self.release
                .lock()
                .unwrap()
                .recv_timeout(EVENT_DEADLINE)
                .expect("test controller releases admission before the deadline");
        }
        self.scheduler
            .schedule::<C, F>(priority, locality_hint, task)
    }
}

struct WorkerRelease(Option<mpsc::SyncSender<()>>);

impl WorkerRelease {
    fn release(&mut self) {
        self.0
            .take()
            .expect("worker gate releases once")
            .send(())
            .expect("gated worker remains alive");
    }
}

struct SaturatedExecutor {
    worker_release: WorkerRelease,
    executor: HybridExecutor,
    filler_runs: Arc<AtomicUsize>,
}

impl SaturatedExecutor {
    fn new() -> Self {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            max_global_queue_size: 2,
            ..ExecutorConfig::default()
        })
        .expect("executor");
        let scheduler = executor.scheduler.clone();
        let (worker_entered_tx, worker_entered_rx) = mpsc::sync_channel(1);
        let (worker_release_tx, worker_release_rx) = mpsc::sync_channel(1);
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |worker_id| {
                worker_entered_tx
                    .send(worker_id)
                    .expect("worker-gate observer remains connected");
                worker_release_rx
                    .recv_timeout(EVENT_DEADLINE)
                    .expect("test controller releases worker before the deadline");
            })
            .expect("worker gate admits");
        assert_eq!(
            worker_entered_rx
                .recv_timeout(EVENT_DEADLINE)
                .expect("worker enters gate before the deadline"),
            0
        );

        let filler_runs = Arc::new(AtomicUsize::new(0));
        for _ in 0..2 {
            let filler_runs = Arc::clone(&filler_runs);
            scheduler
                .schedule::<SyncTask, _>(Priority::Normal, None, move |worker_id| {
                    assert_eq!(worker_id, 0);
                    filler_runs.fetch_add(1, Ordering::SeqCst);
                })
                .expect("bounded queue slot admits filler");
        }
        assert_eq!(scheduler.pending_tasks(), 2);

        Self {
            worker_release: WorkerRelease(Some(worker_release_tx)),
            executor,
            filler_runs,
        }
    }

    fn scheduler(&self) -> ThreadScheduler {
        self.executor.scheduler.clone()
    }

    fn release_and_join(&mut self) {
        self.worker_release.release();
        self.executor
            .scheduler
            .join()
            .expect("accepted filler jobs drain");
        assert_eq!(self.filler_runs.load(Ordering::SeqCst), 2);
    }
}

impl Drop for WorkerRelease {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            match release.send(()) {
                Ok(()) | Err(_) => {}
            }
        }
    }
}

#[test]
fn cancellation_wake_cannot_claim_first_admission() {
    let mut saturated = SaturatedExecutor::new();
    let registry = TaskRegistry::new();
    let (task_id, lifecycle) = registry.register_next_task();
    let (handle, result_sender) = TaskHandle::new_pending(TaskId(task_id));
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let state = AsyncFutureState::new(
        saturated.scheduler(),
        UnpolledFuture {
            polls: Arc::clone(&polls),
            drops: Arc::clone(&drops),
        },
        lifecycle,
        result_sender,
        Arc::new(ExecutorMetrics::new()),
    );
    let tasks = Arc::new(AsyncTasks::default());
    state.track(&tasks, task_id).expect("table admits task");

    assert_eq!(
        registry.request_cancel(task_id),
        Some(CancelOutcome::Requested)
    );
    tasks.cancel(task_id).expect("cancellation wake completes");
    assert!(!handle.is_finished());
    assert_eq!(saturated.executor.scheduler.pending_tasks(), 2);

    let admission = Arc::clone(&state).schedule();
    assert!(matches!(
        admission,
        Err(ExecutorError::ResourceExhausted(_))
    ));
    assert_eq!(state.complete_rejected(), Ok(()));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    assert!(tasks.close_and_snapshot().is_empty());
    saturated.release_and_join();
}

#[test]
fn shutdown_wake_during_first_rejection_preserves_completion_owner() {
    let mut saturated = SaturatedExecutor::new();
    let rejection = RejectionFixture::new(saturated.scheduler());
    let registry = TaskRegistry::new();
    let (task_id, lifecycle) = registry.register_next_task();
    let (handle, result_sender) = TaskHandle::new_pending(TaskId(task_id));
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let state = AsyncFutureState::new(
        Arc::clone(&rejection.scheduler),
        UnpolledFuture {
            polls: Arc::clone(&polls),
            drops: Arc::clone(&drops),
        },
        lifecycle,
        result_sender,
        Arc::new(ExecutorMetrics::new()),
    );
    let tasks = Arc::new(AsyncTasks::default());
    state.track(&tasks, task_id).expect("table admits task");

    let scheduled_state = Arc::clone(&state);
    let (admission_sender, admission_receiver) = mpsc::sync_channel(1);
    let admission = thread::spawn(move || {
        admission_sender
            .send(scheduled_state.schedule())
            .expect("admission observer remains connected");
    });
    rejection
        .entered
        .recv_timeout(EVENT_DEADLINE)
        .expect("first admission enters gate before the deadline");

    let shutdown_wakers = tasks.close_and_snapshot();
    assert_eq!(shutdown_wakers.len(), 1);
    assert_eq!(
        registry.request_cancel(task_id),
        Some(CancelOutcome::Requested)
    );
    for wake in shutdown_wakers.into_values() {
        wake.wake();
    }

    rejection
        .release
        .send(())
        .expect("admission gate remains connected");
    let admission_result = admission_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("first admission rejects before the deadline");
    admission.join().expect("first admission thread exits");
    assert!(matches!(
        admission_result,
        Err(ExecutorError::ResourceExhausted(_))
    ));
    assert_eq!(rejection.scheduler.attempts.load(Ordering::SeqCst), 1);
    assert_eq!(state.complete_rejected(), Ok(()));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    assert!(tasks.close_and_snapshot().is_empty());
    saturated.release_and_join();
}

/// A rejected wake polls inline with no retry or lost output.
fn wake_survives_admission_rejection(output: i32) {
    let injector = GatedInjector::new();
    let PendingTask {
        state,
        handle,
        waker,
        polls,
    } = pending_async_state(Arc::clone(&injector), output);

    Arc::clone(&state).schedule().expect("first poll admits");
    injector.drain();
    let waker = waker
        .lock()
        .unwrap()
        .take()
        .expect("first poll published its waker");
    assert_eq!(polls.load(Ordering::SeqCst), 1);

    injector.refuse_next.store(1, Ordering::SeqCst);
    waker.wake();
    assert_eq!(
        injector.rejections.load(Ordering::SeqCst),
        1,
        "the full injector must reject exactly one admission"
    );
    assert_eq!(
        injector.refuse_next.load(Ordering::SeqCst),
        0,
        "the rejection must be consumed"
    );

    assert_eq!(polls.load(Ordering::SeqCst), 2);
    assert_eq!(handle.join(), Some(Ok(output)));
}

#[test]
fn wake_polls_inline_after_admission_rejection() {
    wake_survives_admission_rejection(41);
}

#[test]
fn repeated_self_wake_reports_saturated_reschedule_without_recursion() {
    let injector = GatedInjector::new();
    let registry = TaskRegistry::new();
    let (task_id, lifecycle) = registry.register_next_task();
    let (handle, result_sender) = TaskHandle::new_pending(TaskId(task_id));
    let polls = Arc::new(AtomicUsize::new(0));
    let state = AsyncFutureState::new(
        Arc::clone(&injector),
        AlwaysSelfWake {
            polls: Arc::clone(&polls),
        },
        lifecycle,
        result_sender,
        Arc::new(ExecutorMetrics::new()),
    );

    Arc::clone(&state).schedule().expect("first poll admits");
    injector.refuse_next.store(1, Ordering::SeqCst);
    injector.drain();

    assert_eq!(polls.load(Ordering::SeqCst), 2);
    assert_eq!(injector.rejections.load(Ordering::SeqCst), 1);
    assert!(handle.is_finished());
    assert_eq!(handle.join(), Some(Err(TaskError::ResourceExhausted)));
}

#[test]
fn cross_task_wake_respects_inline_poll_depth_bound() {
    let injector = GatedInjector::new();
    let follower = pending_async_state(Arc::clone(&injector), 19);
    let recovery = pending_async_state(Arc::clone(&injector), 23);
    let registry = TaskRegistry::new();
    let (leader_id, leader_lifecycle) = registry.register_next_task();
    let (leader_handle, leader_sender) = TaskHandle::new_pending(TaskId(leader_id));
    let leader_polls = Arc::new(AtomicUsize::new(0));
    let leader_waker = Arc::new(Mutex::new(None));
    let leader = AsyncFutureState::new(
        Arc::clone(&injector),
        WakePeerThenReady {
            output: 17,
            polls: Arc::clone(&leader_polls),
            waker: Arc::clone(&leader_waker),
            peer_waker: Arc::clone(&follower.waker),
        },
        leader_lifecycle,
        leader_sender,
        Arc::new(ExecutorMetrics::new()),
    );

    Arc::clone(&follower.state)
        .schedule()
        .expect("follower first poll admits");
    Arc::clone(&recovery.state)
        .schedule()
        .expect("recovery first poll admits");
    Arc::clone(&leader)
        .schedule()
        .expect("leader first poll admits");
    injector.drain();

    injector.refuse_next.store(2, Ordering::SeqCst);
    leader_waker
        .lock()
        .unwrap()
        .take()
        .expect("leader first poll must publish its waker")
        .wake();

    assert_eq!(leader_polls.load(Ordering::SeqCst), 2);
    assert_eq!(follower.polls.load(Ordering::SeqCst), 1);
    assert_eq!(leader_handle.join(), Some(Ok(17)));
    assert_eq!(
        follower.handle.join(),
        Some(Err(TaskError::ResourceExhausted))
    );
    assert_eq!(injector.rejections.load(Ordering::SeqCst), 2);

    injector.refuse_next.store(1, Ordering::SeqCst);
    recovery
        .waker
        .lock()
        .unwrap()
        .take()
        .expect("recovery first poll must publish its waker")
        .wake();
    assert_eq!(recovery.polls.load(Ordering::SeqCst), 2);
    assert_eq!(recovery.handle.join(), Some(Ok(23)));
}
