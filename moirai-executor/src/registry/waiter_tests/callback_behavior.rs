use std::{
    future::Future,
    mem::ManuallyDrop,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    sync::{Arc, Mutex},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use moirai_core::executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner};

use crate::HybridExecutor;
use crate::counting_wake::counting_waker;

use super::super::TaskWaiter;
use super::super::registry::TaskRegistry;
use super::owned_registration::pending_task;

pub(super) const CLONE_NONE: u8 = 0;
const CLONE_REENTER: u8 = 1;
const CLONE_PANIC: u8 = 2;
pub(super) const WAKE_NONE: u8 = 0;
const WAKE_PANIC: u8 = 1;
const WAKE_TRACKED_PANIC: u8 = 2;
const WAKE_DROP_PANIC: u8 = 3;
const WAKE_REENTER: u8 = 4;

pub(super) struct CallbackWaker {
    registry: Arc<TaskRegistry>,
    task_id: u64,
    nested: Waker,
    clone_action: AtomicU8,
    drop_reenters: AtomicBool,
    drop_panics: AtomicBool,
    wake_action: AtomicU8,
    pub(super) wakes: AtomicUsize,
    pub(super) waker_drops: AtomicUsize,
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

pub(super) fn callback_waker(state: Arc<CallbackWaker>) -> Waker {
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

pub(super) fn callback_target(
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
    assert!(matches!(
        wait.as_mut().poll(&mut normal_context),
        Poll::Pending
    ));
    release_tx.send(()).unwrap();
    assert_eq!(handle.join().unwrap().unwrap(), 7);
    executor.join().unwrap();
    assert_eq!(normal.wakes(), 1);
    assert!(matches!(
        wait.as_mut().poll(&mut normal_context),
        Poll::Ready(Ok(()))
    ));
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
