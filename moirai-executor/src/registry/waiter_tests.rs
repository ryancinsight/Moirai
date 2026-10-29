#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! Owned task-completion registrations and adversarial callback behavior.

use std::{
    future::Future,
    mem::ManuallyDrop,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    sync::{Arc, Barrier, Mutex},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    thread,
};

use moirai_core::executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner};

use crate::HybridExecutor;
use crate::counting_wake::counting_waker;

use super::TaskWaiter;
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
const WAKE_REENTER: u8 = 4;

struct CallbackWaker {
    registry: Arc<TaskRegistry>,
    task_id: u64,
    nested: Waker,
    clone_action: AtomicU8,
    drop_reenters: AtomicBool,
    drop_panics: AtomicBool,
    wake_action: AtomicU8,
    wakes: AtomicUsize,
    waker_drops: AtomicUsize,
    payload_drops: Arc<AtomicUsize>,
    nested_waiter: Mutex<Option<TaskWaiter>>,
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
            let waiter = state
                .registry
                .register_waker(state.task_id, &state.nested)
                .expect("the reentrant target remains pending");
            *state.nested_waiter.lock().unwrap() = Some(waiter);
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
        WAKE_REENTER => {
            assert!(
                state
                    .registry
                    .register_waker(state.task_id, &state.nested)
                    .is_none(),
                "completion closes registration before wake callbacks"
            );
        }
        _ => {}
    }
}

unsafe fn callback_drop(data: *const ()) {
    // SAFETY: `RawWakerVTable::drop` consumes the one strong reference carried
    // by this raw waker.
    let state = unsafe { Arc::from_raw(data.cast::<CallbackWaker>()) };
    state.waker_drops.fetch_add(1, Ordering::AcqRel);
    if state.drop_reenters.swap(false, Ordering::AcqRel) {
        let waiter = state
            .registry
            .register_waker(state.task_id, &state.nested)
            .expect("the reentrant target remains pending");
        *state.nested_waiter.lock().unwrap() = Some(waiter);
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
        waker_drops: AtomicUsize::new(0),
        payload_drops: Arc::new(AtomicUsize::new(0)),
        nested_waiter: Mutex::new(None),
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
        WAKE_REENTER,
    );
    let waker = callback_waker(Arc::clone(&state));

    let _waiter = registry.register_waker(id, &waker).unwrap();
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
        drop(registry.register_waker(id, &panicking));
    }));
    assert!(panic.is_err(), "the adversarial clone must panic");

    let (normal, normal_waker) = counting_waker();
    let _normal_waiter = registry.register_waker(id, &normal_waker).unwrap();
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

    let mut waiter = registry.register_waker(id, &waker).unwrap();
    assert!(waiter.update(&waker));
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
    let mut waiter = registry.register_waker(id, &waker).unwrap();
    let _normal_waiter = registry.register_waker(id, &normal_waker).unwrap();

    let panic = catch_unwind(AssertUnwindSafe(|| {
        waiter.update(&waker);
    }));
    assert!(panic.is_err(), "the adversarial drop must panic");
    lifecycle.start(0).complete().notify();

    assert_eq!(state.wakes.load(Ordering::Acquire), 1);
    assert_eq!(normal.wakes(), 1);
    assert!(registry.is_completed(id));
}

#[test]
fn completion_attempts_every_wake_and_resumes_the_first_panic() {
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
    let _first_waiter = registry.register_waker(id, &first_waker).unwrap();
    let _secondary_waiter = registry.register_waker(id, &secondary_waker).unwrap();
    let _cleanup_waiter = registry
        .register_waker(id, &cleanup_panicker_waker)
        .unwrap();
    let _normal_waiter = registry.register_waker(id, &normal_waker).unwrap();

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

struct GatedWake {
    entered: std::sync::mpsc::SyncSender<()>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl std::task::Wake for GatedWake {
    fn wake(self: Arc<Self>) {
        self.entered.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
    }
}

#[test]
fn dropping_an_unclaimed_closed_waiter_prevents_its_notification() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let (entered_sender, entered_receiver) = std::sync::mpsc::sync_channel(1);
    let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(1);
    let gated_waker = Waker::from(Arc::new(GatedWake {
        entered: entered_sender,
        release: Mutex::new(release_receiver),
    }));
    let _gated_waiter = registry.register_waker(id, &gated_waker).unwrap();
    let pending = callback_target(
        Arc::clone(&registry),
        id,
        Waker::noop().clone(),
        CLONE_NONE,
        false,
        false,
        WAKE_NONE,
    );
    let pending_waker = callback_waker(Arc::clone(&pending));
    let pending_waiter = registry.register_waker(id, &pending_waker).unwrap();

    let completion = thread::spawn(move || lifecycle.start(0).complete().notify());
    entered_receiver.recv().unwrap();
    drop(pending_waiter);
    assert_eq!(
        pending.waker_drops.load(Ordering::Acquire),
        1,
        "dropping the owner releases its unclaimed closed registration"
    );
    assert_eq!(pending.wakes.load(Ordering::Acquire), 0);

    release_sender.send(()).unwrap();
    completion.join().unwrap();
    assert_eq!(pending.wakes.load(Ordering::Acquire), 0);
}

#[test]
fn completion_wakes_two_waiters_on_one_task() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (first, first_waker) = counting_waker();
    let (second, second_waker) = counting_waker();

    let _first_waiter = registry.register_waker(id, &first_waker).unwrap();
    let _second_waiter = registry.register_waker(id, &second_waker).unwrap();
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

    let mut registrations: Vec<_> = waiters
        .iter()
        .map(|(_, waker)| registry.register_waker(id, waker).unwrap())
        .collect();
    // Re-polling updates each owned registration instead of adding history.
    for (registration, (_, waker)) in registrations.iter_mut().zip(&waiters) {
        assert!(registration.update(waker));
    }
    lifecycle.start(0).complete().notify();

    for (index, (target, _)) in waiters.iter().enumerate() {
        assert_eq!(target.wakes(), 1, "waiter {index}");
    }
}

#[test]
fn independent_waiters_with_one_caller_waker_are_each_preserved() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (target, waker) = counting_waker();

    let registrations: Vec<_> = (0..3)
        .map(|_| registry.register_waker(id, &waker).unwrap())
        .collect();
    assert_eq!(
        Arc::strong_count(&target),
        5,
        "the handle, caller waker, and three independent registrations"
    );
    lifecycle.start(0).complete().notify();

    assert_eq!(target.wakes(), 3);
    drop(registrations);
    assert_eq!(
        Arc::strong_count(&target),
        2,
        "completion releases the registry entry"
    );
}

#[test]
fn completed_tasks_reject_new_waiters_without_retaining_callbacks() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (early, early_waker) = counting_waker();
    let (early_second, early_second_waker) = counting_waker();
    let _early_waiter = registry.register_waker(id, &early_waker).unwrap();
    let _early_second_waiter = registry.register_waker(id, &early_second_waker).unwrap();
    lifecycle.start(0).complete().notify();

    let (late, late_waker) = counting_waker();
    let (late_second, late_second_waker) = counting_waker();
    assert!(registry.register_waker(id, &late_waker).is_none());
    assert!(registry.register_waker(id, &late_second_waker).is_none());
    drop((
        early_waker,
        early_second_waker,
        late_waker,
        late_second_waker,
    ));

    for target in [&early, &early_second] {
        assert_eq!(target.wakes(), 1);
        assert_eq!(Arc::strong_count(target), 1, "completion retains no waker");
    }
    for target in [&late, &late_second] {
        assert_eq!(target.wakes(), 0);
        assert_eq!(Arc::strong_count(target), 1);
    }
}

#[test]
fn waiter_ownership_outlives_state_storage_without_a_state_pointer() {
    let state = TaskState::new();
    let waiters: Vec<_> = (0..3).map(|_| counting_waker()).collect();
    let registrations: Vec<_> = waiters
        .iter()
        .map(|(_, waker)| state.register_waker(waker).unwrap())
        .collect();
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
        assert_eq!(Arc::strong_count(target), 2, "the owned waiter remains");
    }
    drop(registrations);

    for target in &targets {
        assert_eq!(target.wakes(), 0, "an unfinished task wakes nobody");
        assert_eq!(Arc::strong_count(target), 1, "a waker outlived its state");
    }
}

#[test]
fn repeated_abandonment_reuses_one_spill_slot_and_releases_each_payload() {
    const ABANDONED: usize = 1_000;
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (survivor, survivor_waker) = counting_waker();
    let survivor_waiter = registry.register_waker(id, &survivor_waker).unwrap();

    for index in 0..ABANDONED {
        let (target, waker) = counting_waker();
        let waiter = registry.register_waker(id, &waker).unwrap();
        assert_eq!(waiter.spill_slots(), 1, "abandonment {index}");
        drop(waiter);
        drop(waker);
        assert_eq!(
            Arc::strong_count(&target),
            1,
            "abandonment {index} retained its waker payload"
        );
    }

    lifecycle.start(0).complete().notify();
    assert_eq!(survivor.wakes(), 1);
    drop(survivor_waiter);
}

#[test]
fn an_empty_set_releases_its_spill_storage() {
    let registry = TaskRegistry::new();
    let (id, _lifecycle) = pending_task(&registry);
    let waiters: Vec<_> = (0..8).map(|_| counting_waker()).collect();
    let registrations: Vec<_> = waiters
        .iter()
        .map(|(_, waker)| registry.register_waker(id, waker).unwrap())
        .collect();
    assert_eq!(registrations[0].spill_slots(), 7);
    drop(registrations);

    let (_, waker) = counting_waker();
    let waiter = registry.register_waker(id, &waker).unwrap();
    assert_eq!(waiter.spill_slots(), 0);
}

#[test]
fn repoll_replaces_and_releases_the_previous_waker() {
    let registry = TaskRegistry::new();
    let (id, lifecycle) = pending_task(&registry);
    let (old_target, old_waker) = counting_waker();
    let (current_target, current_waker) = counting_waker();
    let mut waiter = registry.register_waker(id, &old_waker).unwrap();

    assert!(waiter.update(&current_waker));
    assert_eq!(Arc::strong_count(&old_target), 2);
    assert_eq!(Arc::strong_count(&current_target), 3);

    lifecycle.start(0).complete().notify();
    assert_eq!(old_target.wakes(), 0);
    assert_eq!(current_target.wakes(), 1);
}

#[test]
fn waiter_drop_reenters_after_removal_and_preserves_a_survivor() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let (nested, nested_waker) = counting_waker();
    let callback = callback_target(
        Arc::clone(&registry),
        id,
        nested_waker,
        CLONE_NONE,
        true,
        false,
        WAKE_NONE,
    );
    let callback_waker = callback_waker(Arc::clone(&callback));
    let callback_waiter = registry.register_waker(id, &callback_waker).unwrap();
    let (survivor, survivor_waker) = counting_waker();
    let _survivor_waiter = registry.register_waker(id, &survivor_waker).unwrap();

    drop(callback_waiter);
    lifecycle.start(0).complete().notify();

    assert_eq!(callback.wakes.load(Ordering::Acquire), 0);
    assert_eq!(nested.wakes(), 1);
    assert_eq!(survivor.wakes(), 1);
}

#[test]
fn waiter_drop_contains_callback_panic_and_preserves_a_survivor() {
    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let callback = callback_target(
        Arc::clone(&registry),
        id,
        Waker::noop().clone(),
        CLONE_NONE,
        false,
        true,
        WAKE_NONE,
    );
    let callback_waker = callback_waker(Arc::clone(&callback));
    let callback_waiter = registry.register_waker(id, &callback_waker).unwrap();
    let (survivor, survivor_waker) = counting_waker();
    let _survivor_waiter = registry.register_waker(id, &survivor_waker).unwrap();

    drop(callback_waiter);
    lifecycle.start(0).complete().notify();

    assert_eq!(callback.wakes.load(Ordering::Acquire), 0);
    assert_eq!(survivor.wakes(), 1);
}

#[test]
fn completion_and_unregister_can_race_block_retirement() {
    use std::time::Duration;

    let registry = Arc::new(TaskRegistry::new());
    let (id, lifecycle) = pending_task(&registry);
    let (target, waker) = counting_waker();
    let waiter = registry.register_waker(id, &waker).unwrap();
    for _ in 1..super::state::TASK_STATE_BLOCK_SIZE {
        let (_, other) = pending_task(&registry);
        other.start(0).complete().notify();
    }
    let start = Arc::new(Barrier::new(3));
    let completion_start = Arc::clone(&start);
    let completion = thread::spawn(move || {
        completion_start.wait();
        lifecycle.start(0).complete().notify();
    });
    let removal_start = Arc::clone(&start);
    let removal = thread::spawn(move || {
        removal_start.wait();
        drop(waiter);
    });

    start.wait();
    completion.join().unwrap();
    removal.join().unwrap();
    assert!(target.wakes() <= 1);
    assert_eq!(registry.cleanup_completed(Duration::ZERO), 1);
    assert!(registry.is_completed(id));
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
                let waiter = registry.register_waker(id, &waker).unwrap();
                let completed_at_recheck = registry.is_completed(id);
                registered.wait();
                (completed_at_recheck, waiter)
            });
            (target, thread)
        })
        .collect();

    registered.wait();
    lifecycle.start(0).complete().notify();

    for (index, (target, thread)) in waiters.into_iter().enumerate() {
        let (completed_at_recheck, waiter) = thread.join().unwrap();
        assert!(!completed_at_recheck, "waiter {index} saw completion early");
        assert_eq!(target.wakes(), 1, "waiter {index}");
        drop(waiter);
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
                    let waiter = registry.register_waker(id, &waker);
                    let completed = registry.is_completed(id);
                    (waiter, completed)
                });
                (target, thread)
            })
            .unzip();

        start.wait();
        lifecycle.start(0).complete().notify();

        // Join every thread before asserting: a recheck on one thread may take
        // another waiter's waker and wake it before that waiter's thread
        // returns.
        for (index, (target, thread)) in targets.iter().zip(threads).enumerate() {
            let (waiter, completed) = thread.join().unwrap();
            if waiter.is_some() {
                assert_eq!(target.wakes(), 1, "round {round}: waiter {index}");
            } else {
                assert!(completed, "a rejected registration observes completion");
                assert_eq!(target.wakes(), 0, "round {round}: waiter {index}");
            }
        }
    }
}
