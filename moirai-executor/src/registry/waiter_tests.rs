#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! Several waiters on one task: completion wakes every distinct waker, a
//! repeated registration of one waker counts once, and nothing registered is
//! retained past completion or past the task state.

use std::{
    future::Future,
    mem::ManuallyDrop,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    sync::{Arc, Barrier},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    thread,
};

use moirai_core::executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner};

use crate::HybridExecutor;
use crate::counting_wake::counting_waker;

use super::registry::TaskRegistry;
use super::state::TaskState;
use super::token::TaskLifecycleToken;

const CLONE_NONE: u8 = 0;
const CLONE_REENTER: u8 = 1;
const CLONE_PANIC: u8 = 2;
const WAKE_NONE: u8 = 0;
const WAKE_PANIC: u8 = 1;
const WAKE_TRACKED_PANIC: u8 = 2;
const WAKE_DROP_PANIC: u8 = 3;

struct CallbackWaker {
    registry: Arc<TaskRegistry>,
    task_id: u64,
    nested: Waker,
    clone_action: AtomicU8,
    drop_reenters: AtomicBool,
    drop_panics: AtomicBool,
    wake_action: AtomicU8,
    wakes: AtomicUsize,
    payload_drops: Arc<AtomicUsize>,
}

struct TrackedPanic(Arc<AtomicUsize>);

impl Drop for TrackedPanic {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

struct DropPanics(Arc<AtomicUsize>);

impl Drop for DropPanics {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::AcqRel);
        panic!("secondary panic payload drop");
    }
}

static CALLBACK_VTABLE: RawWakerVTable = RawWakerVTable::new(
    callback_clone,
    callback_wake,
    callback_wake_by_ref,
    callback_drop,
);

fn callback_waker(state: Arc<CallbackWaker>) -> Waker {
    let data = Arc::into_raw(state).cast::<()>();
    // SAFETY: `data` owns one `Arc<CallbackWaker>` strong reference and every
    // vtable function preserves or consumes exactly that ownership.
    unsafe { Waker::from_raw(RawWaker::new(data, &CALLBACK_VTABLE)) }
}

fn callback_state(data: *const ()) -> ManuallyDrop<Arc<CallbackWaker>> {
    // SAFETY: every `CALLBACK_VTABLE` invocation receives the pointer created
    // by `callback_waker` or `callback_clone`. `ManuallyDrop` preserves the
    // vtable-owned strong reference for by-reference operations.
    ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<CallbackWaker>()) })
}

unsafe fn callback_clone(data: *const ()) -> RawWaker {
    let state = callback_state(data);
    match state.clone_action.swap(CLONE_NONE, Ordering::AcqRel) {
        CLONE_REENTER => {
            assert!(state.registry.register_waker(state.task_id, &state.nested));
        }
        CLONE_PANIC => panic!("clone callback panic"),
        _ => {}
    }
    // SAFETY: the source waker retains its strong reference; this increments
    // once for the independently owned waker returned by the clone callback.
    unsafe { Arc::increment_strong_count(data.cast::<CallbackWaker>()) };
    RawWaker::new(data, &CALLBACK_VTABLE)
}

unsafe fn callback_wake(data: *const ()) {
    // SAFETY: `RawWakerVTable::wake` consumes the one strong reference carried
    // by this raw waker.
    let state = unsafe { Arc::from_raw(data.cast::<CallbackWaker>()) };
    run_wake_callback(&state);
}

unsafe fn callback_wake_by_ref(data: *const ()) {
    let state = callback_state(data);
    run_wake_callback(&state);
}

fn run_wake_callback(state: &CallbackWaker) {
    state.wakes.fetch_add(1, Ordering::AcqRel);
    match state.wake_action.load(Ordering::Acquire) {
        WAKE_PANIC => panic!("first wake callback panic"),
        WAKE_TRACKED_PANIC => {
            std::panic::panic_any(TrackedPanic(Arc::clone(&state.payload_drops)));
        }
        WAKE_DROP_PANIC => {
            std::panic::panic_any(DropPanics(Arc::clone(&state.payload_drops)));
        }
        _ => {}
    }
}

unsafe fn callback_drop(data: *const ()) {
    // SAFETY: `RawWakerVTable::drop` consumes the one strong reference carried
    // by this raw waker.
    let state = unsafe { Arc::from_raw(data.cast::<CallbackWaker>()) };
    if state.drop_reenters.swap(false, Ordering::AcqRel) {
        assert!(state.registry.register_waker(state.task_id, &state.nested));
    }
    if state.drop_panics.swap(false, Ordering::AcqRel) {
        panic!("duplicate drop callback panic");
    }
}

fn callback_target(
    registry: Arc<TaskRegistry>,
    task_id: u64,
    nested: Waker,
    clone_action: u8,
    drop_reenters: bool,
    drop_panics: bool,
    wake_action: u8,
) -> Arc<CallbackWaker> {
    Arc::new(CallbackWaker {
        registry,
        task_id,
        nested,
        clone_action: AtomicU8::new(clone_action),
        drop_reenters: AtomicBool::new(drop_reenters),
        drop_panics: AtomicBool::new(drop_panics),
        wake_action: AtomicU8::new(wake_action),
        wakes: AtomicUsize::new(0),
        payload_drops: Arc::new(AtomicUsize::new(0)),
    })
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
fn clone_callback_reenters_the_same_task_without_stranding_either_waiter() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let (nested_target, nested) = counting_waker();
    let state = callback_target(
        Arc::clone(&registry),
        id,
        nested,
        CLONE_REENTER,
        false,
        false,
        WAKE_NONE,
    );
    let waker = callback_waker(Arc::clone(&state));

    assert!(registry.register_waker(id, &waker));
    lifecycle.start(0).complete().notify();

    assert_eq!(state.wakes.load(Ordering::Acquire), 1);
    assert_eq!(nested_target.wakes(), 1);
    assert!(registry.is_completed(id));
}

#[test]
fn clone_panic_leaves_the_slot_usable_for_later_registration_and_completion() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let state = callback_target(
        Arc::clone(&registry),
        id,
        Waker::noop().clone(),
        CLONE_PANIC,
        false,
        false,
        WAKE_NONE,
    );
    let panicking = callback_waker(Arc::clone(&state));

    let panic = catch_unwind(AssertUnwindSafe(|| {
        registry.register_waker(id, &panicking);
    }));
    assert!(panic.is_err(), "the adversarial clone must panic");

    let (normal, normal_waker) = counting_waker();
    assert!(registry.register_waker(id, &normal_waker));
    lifecycle.start(0).complete().notify();

    assert_eq!(normal.wakes(), 1);
    assert_eq!(state.wakes.load(Ordering::Acquire), 0);
    assert!(registry.is_completed(id));
}

#[test]
fn task_manager_wait_recovers_after_its_waker_clone_panics() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let handle = executor
        .spawn_blocking(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            7_usize
        })
        .unwrap();
    started_rx.recv().unwrap();

    let callback = callback_target(
        Arc::new(TaskRegistry::new()),
        0,
        Waker::noop().clone(),
        CLONE_PANIC,
        false,
        false,
        WAKE_NONE,
    );
    let panicking = callback_waker(callback);
    let mut panicking_context = Context::from_waker(&panicking);
    let mut wait = std::pin::pin!(executor.wait_for_task(handle.id(), None));
    let panic = catch_unwind(AssertUnwindSafe(|| {
        wait.as_mut().poll(&mut panicking_context)
    }));
    assert!(panic.is_err(), "the adversarial clone must panic");

    let (normal, normal_waker) = counting_waker();
    let mut normal_context = Context::from_waker(&normal_waker);
    assert_eq!(wait.as_mut().poll(&mut normal_context), Poll::Pending);
    release_tx.send(()).unwrap();
    assert_eq!(handle.join().unwrap().unwrap(), 7);
    executor.join().unwrap();
    assert_eq!(normal.wakes(), 1);
    assert_eq!(wait.as_mut().poll(&mut normal_context), Poll::Ready(Ok(())));
    executor.shutdown();
}

#[test]
fn duplicate_drop_callback_reenters_after_the_slot_is_released() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let (nested_target, nested) = counting_waker();
    let state = callback_target(
        Arc::clone(&registry),
        id,
        nested,
        CLONE_NONE,
        true,
        false,
        WAKE_NONE,
    );
    let waker = callback_waker(Arc::clone(&state));

    assert!(registry.register_waker(id, &waker));
    assert!(registry.register_waker(id, &waker));
    lifecycle.start(0).complete().notify();

    assert_eq!(state.wakes.load(Ordering::Acquire), 1);
    assert_eq!(nested_target.wakes(), 1);
    assert!(registry.is_completed(id));
}

#[test]
fn panicking_duplicate_drop_preserves_every_registered_waiter() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let state = callback_target(
        Arc::clone(&registry),
        id,
        Waker::noop().clone(),
        CLONE_NONE,
        false,
        true,
        WAKE_NONE,
    );
    let waker = callback_waker(Arc::clone(&state));
    let (normal, normal_waker) = counting_waker();
    assert!(registry.register_waker(id, &waker));
    assert!(registry.register_waker(id, &normal_waker));

    let panic = catch_unwind(AssertUnwindSafe(|| {
        registry.register_waker(id, &waker);
    }));
    assert!(panic.is_err(), "the adversarial drop must panic");
    lifecycle.start(0).complete().notify();

    assert_eq!(state.wakes.load(Ordering::Acquire), 1);
    assert_eq!(normal.wakes(), 1);
    assert!(registry.is_completed(id));
}

#[test]
fn fan_out_attempts_every_wake_and_resumes_the_first_panic() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let first = callback_target(
        Arc::clone(&registry),
        id,
        Waker::noop().clone(),
        CLONE_NONE,
        false,
        false,
        WAKE_PANIC,
    );
    let secondary = callback_target(
        Arc::clone(&registry),
        id,
        Waker::noop().clone(),
        CLONE_NONE,
        false,
        false,
        WAKE_TRACKED_PANIC,
    );
    let cleanup_panicker = callback_target(
        Arc::clone(&registry),
        id,
        Waker::noop().clone(),
        CLONE_NONE,
        false,
        false,
        WAKE_DROP_PANIC,
    );
    let first_waker = callback_waker(Arc::clone(&first));
    let secondary_waker = callback_waker(Arc::clone(&secondary));
    let cleanup_panicker_waker = callback_waker(Arc::clone(&cleanup_panicker));
    let (normal, normal_waker) = counting_waker();
    assert!(registry.register_waker(id, &first_waker));
    assert!(registry.register_waker(id, &secondary_waker));
    assert!(registry.register_waker(id, &cleanup_panicker_waker));
    assert!(registry.register_waker(id, &normal_waker));

    let panic = catch_unwind(AssertUnwindSafe(|| {
        lifecycle.start(0).complete().notify();
    }))
    .expect_err("the first wake panic must propagate");

    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"first wake callback panic")
    );
    assert_eq!(first.wakes.load(Ordering::Acquire), 1);
    assert_eq!(secondary.wakes.load(Ordering::Acquire), 1);
    assert_eq!(secondary.payload_drops.load(Ordering::Acquire), 1);
    assert_eq!(cleanup_panicker.wakes.load(Ordering::Acquire), 1);
    assert_eq!(cleanup_panicker.payload_drops.load(Ordering::Acquire), 1);
    assert_eq!(normal.wakes(), 1);
    assert!(registry.is_completed(id));
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

    lifecycle.start(0).complete().notify();

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
    lifecycle.start(0).complete().notify();

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
    lifecycle.start(0).complete().notify();

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
    lifecycle.start(0).complete().notify();

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
    lifecycle.start(0).complete().notify();

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
        lifecycle.start(0).complete().notify();

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
