use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Wake, Waker},
};

use moirai_core::{
    error::TaskError,
    task::{TaskHandle, TaskId},
};

use super::super::AsyncFutureState;
use super::fixtures::{AlwaysSelfWake, GatedInjector, WakeThenReady};
use crate::metrics::ExecutorMetrics;
use crate::registry::TaskRegistry;

/// Waker standing in for a `wait_for_task` future parked on the task.
struct CompletionWaiter(AtomicUsize);

impl Wake for CompletionWaiter {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A wake that scheduler shutdown refuses can never be honored, so the task
/// ends there: its waiters wake, its handle resolves to `Cancelled`, and the
/// registry records it cancelled. Leaving it idle would hold all three until
/// every waker clone dropped, which for a task parked on an external event
/// may be never.
#[test]
fn wake_refused_by_shutdown_completes_the_task_as_cancelled() {
    let injector = GatedInjector::new();
    let registry = TaskRegistry::new();
    let (task_id, lifecycle) = registry.register_next_task();
    let (handle, result_sender) = TaskHandle::new_pending(TaskId(task_id));
    let polls = Arc::new(AtomicUsize::new(0));
    let waker = Arc::new(Mutex::new(None));
    let metrics = Arc::new(ExecutorMetrics::new());
    let state = AsyncFutureState::new(
        Arc::clone(&injector),
        WakeThenReady {
            output: 5,
            polls: Arc::clone(&polls),
            waker: Arc::clone(&waker),
            first_poll_sender: None,
        },
        lifecycle,
        result_sender,
        Arc::clone(&metrics),
    );

    Arc::clone(&state).schedule().expect("first poll admits");
    injector.drain();
    let waker = waker
        .lock()
        .unwrap()
        .take()
        .expect("first poll published its waker");
    let waiter = Arc::new(CompletionWaiter(AtomicUsize::new(0)));
    assert!(registry.register_waker(task_id, &Waker::from(Arc::clone(&waiter))));
    assert!(!registry.is_completed(task_id));

    injector.shutting_down.store(true, Ordering::SeqCst);
    waker.wake();

    assert_eq!(
        waiter.0.load(Ordering::SeqCst),
        1,
        "the refused wake must complete the task and wake its waiter"
    );
    let metadata = registry.get_metadata(task_id).unwrap();
    assert!(metadata.cancelled);
    assert!(registry.is_completed(task_id));
    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(metrics.tasks_cancelled.load(Ordering::Relaxed), 1);
    assert_eq!(
        polls.load(Ordering::SeqCst),
        1,
        "the body never polls again"
    );
}

/// The reschedule that follows an exhausted inline-repoll budget meets the
/// same refusal with the lifecycle already `Running`, and ends the same way.
#[test]
fn reschedule_refused_by_shutdown_completes_the_running_task_as_cancelled() {
    let injector = GatedInjector::new();
    let registry = TaskRegistry::new();
    let (task_id, lifecycle) = registry.register_next_task();
    let (handle, result_sender) = TaskHandle::new_pending(TaskId(task_id));
    let polls = Arc::new(AtomicUsize::new(0));
    let metrics = Arc::new(ExecutorMetrics::new());
    let state = AsyncFutureState::new(
        Arc::clone(&injector),
        AlwaysSelfWake {
            polls: Arc::clone(&polls),
        },
        lifecycle,
        result_sender,
        Arc::clone(&metrics),
    );

    Arc::clone(&state).schedule().expect("first poll admits");
    injector.shutting_down.store(true, Ordering::SeqCst);
    injector.drain();

    assert_eq!(polls.load(Ordering::SeqCst), 2);
    let metadata = registry.get_metadata(task_id).unwrap();
    assert!(metadata.cancelled);
    let started_at = metadata
        .started_at
        .expect("the first poll started the task");
    let completed_at = metadata
        .completed_at
        .expect("the refused reschedule completes the task");
    assert!(completed_at >= started_at);
    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(metrics.tasks_cancelled.load(Ordering::Relaxed), 1);
}
