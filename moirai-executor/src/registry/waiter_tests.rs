#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! Several waiters on one task: completion wakes every distinct waker, a
//! repeated registration of one waker counts once, and nothing registered is
//! retained past completion or past the task state.

use std::{
    mem::ManuallyDrop,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{RawWaker, RawWakerVTable, Waker},
    thread,
    time::Duration,
};

use crate::counting_wake::counting_waker;

use super::registry::TaskRegistry;
use super::state::{TaskState, elapsed_nanos_since};
use super::token::TaskLifecycleToken;

#[derive(Clone, Copy)]
enum CallbackAction {
    CloneRegisters,
    ClonePanics,
    DropCompletes,
}

struct CallbackProbe {
    state: Arc<TaskState>,
    action: CallbackAction,
    clone_calls: AtomicUsize,
    drop_calls: AtomicUsize,
    wake_calls: AtomicUsize,
    drop_reentry_armed: AtomicBool,
}

static CALLBACK_VTABLE: RawWakerVTable = RawWakerVTable::new(
    clone_callback,
    wake_callback,
    wake_by_ref_callback,
    drop_callback,
);

fn callback_waker(state: Arc<TaskState>, action: CallbackAction) -> (Arc<CallbackProbe>, Waker) {
    let probe = Arc::new(CallbackProbe {
        state,
        action,
        clone_calls: AtomicUsize::new(0),
        drop_calls: AtomicUsize::new(0),
        wake_calls: AtomicUsize::new(0),
        drop_reentry_armed: AtomicBool::new(true),
    });
    let data = Arc::into_raw(Arc::clone(&probe)).cast::<()>();
    // SAFETY: `data` owns one `Arc<CallbackProbe>` strong reference and every
    // callback below preserves the `RawWakerVTable` ownership rules.
    let waker = unsafe { Waker::from_raw(RawWaker::new(data, &CALLBACK_VTABLE)) };
    (probe, waker)
}

unsafe fn clone_callback(data: *const ()) -> RawWaker {
    // SAFETY: `data` is one live `Arc<CallbackProbe>` reference. This
    // temporary view does not consume it.
    let probe = ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<CallbackProbe>()) });
    probe.clone_calls.fetch_add(1, Ordering::Relaxed);
    if probe.state.waker.is_locked() {
        panic!("clone ran while slot lock held");
    }
    match probe.action {
        CallbackAction::CloneRegisters => probe.state.register_waker(Waker::noop()),
        CallbackAction::ClonePanics => panic!("clone callback panic"),
        CallbackAction::DropCompletes => {}
    }
    let cloned = Arc::clone(&probe);
    RawWaker::new(Arc::into_raw(cloned).cast::<()>(), &CALLBACK_VTABLE)
}

unsafe fn wake_callback(data: *const ()) {
    // SAFETY: value wake consumes the callback's live Arc reference.
    let probe = unsafe { Arc::from_raw(data.cast::<CallbackProbe>()) };
    probe.wake_calls.fetch_add(1, Ordering::Relaxed);
}

unsafe fn wake_by_ref_callback(data: *const ()) {
    // SAFETY: reference wake borrows the callback's live Arc reference.
    let probe = ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<CallbackProbe>()) });
    probe.wake_calls.fetch_add(1, Ordering::Relaxed);
}

unsafe fn drop_callback(data: *const ()) {
    // SAFETY: drop consumes the callback's live Arc reference.
    let probe = unsafe { Arc::from_raw(data.cast::<CallbackProbe>()) };
    probe.drop_calls.fetch_add(1, Ordering::Relaxed);
    if matches!(probe.action, CallbackAction::DropCompletes)
        && probe.drop_reentry_armed.swap(false, Ordering::AcqRel)
    {
        assert!(!probe.state.waker.is_locked());
        probe.state.mark_completed();
    }
}

/// Register a task and keep its lifecycle token, so the task stays pending
/// until the token completes it (a dropped unstarted token completes the task).
fn pending_task(registry: &TaskRegistry) -> (u64, TaskLifecycleToken) {
    let (id, lifecycle) = registry.register_next_task();
    assert!(
        !registry.is_completed(id),
        "a held token leaves the task pending"
    );
    (id, lifecycle)
}

#[test]
fn completion_wakes_two_waiters_on_one_task() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (first, first_waker) = counting_waker();
    let (second, second_waker) = counting_waker();

    assert!(registry.register_waker(id, &first_waker));
    assert!(registry.register_waker(id, &second_waker));
    assert_eq!(first.wakes() + second.wakes(), 0, "nothing completed yet");

    lifecycle.start(0).complete();

    assert_eq!(first.wakes(), 1, "the first waiter was overwritten");
    assert_eq!(second.wakes(), 1);
}

#[test]
fn completion_wakes_every_distinct_waiter_once() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let waiters: Vec<_> = (0..5).map(|_| counting_waker()).collect();

    // A second round with the same wakers models waiters that are polled
    // again: each must still be woken once, and none may accumulate.
    for _round in 0..2 {
        for (_, waker) in &waiters {
            assert!(registry.register_waker(id, waker));
        }
    }
    lifecycle.start(0).complete();

    for (index, (target, _)) in waiters.iter().enumerate() {
        assert_eq!(target.wakes(), 1, "waiter {index}");
    }
}

#[test]
fn one_waker_registered_repeatedly_is_held_and_woken_once() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (target, waker) = counting_waker();

    for _ in 0..3 {
        assert!(registry.register_waker(id, &waker));
    }
    assert_eq!(
        Arc::strong_count(&target),
        3,
        "the test's handle, the waker, and exactly one registry entry"
    );
    lifecycle.start(0).complete();

    assert_eq!(target.wakes(), 1);
    assert_eq!(
        Arc::strong_count(&target),
        2,
        "completion releases the registry entry"
    );
}

#[test]
fn waiters_registered_after_completion_are_woken_and_not_retained() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (early, early_waker) = counting_waker();
    let (early_second, early_second_waker) = counting_waker();
    registry.register_waker(id, &early_waker);
    registry.register_waker(id, &early_second_waker);
    lifecycle.start(0).complete();

    let (late, late_waker) = counting_waker();
    let (late_second, late_second_waker) = counting_waker();
    assert!(registry.register_waker(id, &late_waker));
    assert!(registry.register_waker(id, &late_second_waker));
    drop((
        early_waker,
        early_second_waker,
        late_waker,
        late_second_waker,
    ));

    for target in [&early, &early_second, &late, &late_second] {
        assert_eq!(target.wakes(), 1);
        assert_eq!(
            Arc::strong_count(target),
            1,
            "a waker registered for a completed task is held by the registry"
        );
    }
}

#[test]
fn dropping_an_unfinished_state_releases_every_registered_waker() {
    let state = TaskState::new();
    let waiters: Vec<_> = (0..3).map(|_| counting_waker()).collect();
    for (_, waker) in &waiters {
        state.register_waker(waker);
    }
    let targets: Vec<_> = waiters
        .into_iter()
        .map(|(target, waker)| {
            drop(waker);
            target
        })
        .collect();

    drop(state);

    for target in &targets {
        assert_eq!(target.wakes(), 0, "an unfinished task wakes nobody");
        assert_eq!(Arc::strong_count(target), 1, "a waker outlived its state");
    }
}

#[test]
fn completion_clamps_to_recorded_start_offset() {
    let state = TaskState::new();
    let future_start = elapsed_nanos_since(state.created_at).saturating_add(1_000_000);
    state
        .started_after_ns
        .store(future_start, Ordering::Release);

    let elapsed = state.mark_completed_since(future_start);
    let snapshot = state.snapshot(7);

    assert_eq!(elapsed, Duration::ZERO);
    assert_eq!(snapshot.started_at, snapshot.completed_at);
    assert_eq!(snapshot.execution_duration(), Some(Duration::ZERO));
}

#[test]
fn raw_waker_clone_reenters_registration_before_lock() {
    let state = Arc::new(TaskState::new());
    let (probe, waker) = callback_waker(Arc::clone(&state), CallbackAction::CloneRegisters);

    state.register_waker(&waker);
    let started = state.mark_started(3);
    let elapsed = state.mark_completed_since(started);
    let snapshot = state.snapshot(41);

    assert_eq!(probe.clone_calls.load(Ordering::Relaxed), 1);
    assert_eq!(probe.wake_calls.load(Ordering::Relaxed), 1);
    assert_eq!(snapshot.worker_id, Some(3));
    assert_eq!(snapshot.execution_duration(), Some(elapsed));
}

#[test]
fn raw_waker_clone_panic_does_not_strand_slot_lock() {
    let state = Arc::new(TaskState::new());
    let (probe, waker) = callback_waker(Arc::clone(&state), CallbackAction::ClonePanics);

    let panic = catch_unwind(AssertUnwindSafe(|| state.register_waker(&waker)))
        .expect_err("invariant: configured raw waker clone panics");

    assert_eq!(panic.downcast_ref::<&str>(), Some(&"clone callback panic"));
    assert_eq!(probe.clone_calls.load(Ordering::Relaxed), 1);
    assert!(!state.waker.is_locked());
    state.register_waker(Waker::noop());
    state.mark_completed();
    assert!(state.is_completed());
}

#[test]
fn discarded_raw_waker_drops_after_unlock_and_completes() {
    let state = Arc::new(TaskState::new());
    let (probe, waker) = callback_waker(Arc::clone(&state), CallbackAction::DropCompletes);

    state.register_waker(&waker);
    state.register_waker(&waker);

    assert!(state.is_completed());
    assert_eq!(probe.clone_calls.load(Ordering::Relaxed), 2);
    assert_eq!(probe.drop_calls.load(Ordering::Relaxed), 1);
    assert_eq!(probe.wake_calls.load(Ordering::Relaxed), 1);
    assert_eq!(state.snapshot(42).completed_at, state.completed_at());
}

/// Registrations from different threads all land before completion, so every
/// waiter's post-registration recheck is `false` and only the completion wake
/// can reach it.
#[test]
fn waiters_on_separate_threads_are_all_woken_by_completion() {
    const WAITERS: usize = 6;
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let registered = Arc::new(Barrier::new(WAITERS + 1));
    let waiters: Vec<_> = (0..WAITERS)
        .map(|_| {
            let (target, waker) = counting_waker();
            let registry = Arc::clone(&registry);
            let registered = Arc::clone(&registered);
            let thread = thread::spawn(move || {
                assert!(registry.register_waker(id, &waker));
                let completed_at_recheck = registry.is_completed(id);
                registered.wait();
                completed_at_recheck
            });
            (target, thread)
        })
        .collect();

    registered.wait();
    lifecycle.start(0).complete();

    for (index, (target, thread)) in waiters.into_iter().enumerate() {
        assert!(
            !thread.join().unwrap(),
            "waiter {index} saw completion early"
        );
        assert_eq!(target.wakes(), 1, "waiter {index}");
    }
}

/// Completion races the registrations. Whatever the interleaving, every waiter
/// ends up woken exactly once: by the completion, or by the registration or
/// another waiter's recheck that finds the task complete.
#[test]
fn completion_racing_registrations_never_strands_a_waiter() {
    const WAITERS: usize = 4;
    const ROUNDS: usize = 200;
    for round in 0..ROUNDS {
        let registry = Arc::new(TaskRegistry::new());
        let (id, lifecycle) = pending_task(&registry);
        let start = Arc::new(Barrier::new(WAITERS + 1));
        let (targets, threads): (Vec<_>, Vec<_>) = (0..WAITERS)
            .map(|_| {
                let (target, waker) = counting_waker();
                let registry = Arc::clone(&registry);
                let start = Arc::clone(&start);
                let thread = thread::spawn(move || {
                    start.wait();
                    assert!(registry.register_waker(id, &waker));
                });
                (target, thread)
            })
            .unzip();

        start.wait();
        lifecycle.start(0).complete();

        // Join every thread before asserting: a recheck on one thread may take
        // another waiter's waker and wake it before that waiter's thread
        // returns.
        for thread in threads {
            thread.join().unwrap();
        }
        for (index, target) in targets.iter().enumerate() {
            assert_eq!(target.wakes(), 1, "round {round}: waiter {index}");
        }
    }
}
