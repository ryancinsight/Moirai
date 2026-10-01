use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use moirai_core::{
    error::TaskError,
    task::{TaskHandle, TaskId},
};

use super::super::AsyncFutureState;
use super::fixtures::{
    AlwaysSelfWake, GatedInjector, PendingTask, WakePeerThenReady, pending_async_state,
};
use crate::metrics::ExecutorMetrics;
use crate::registry::TaskRegistry;

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
