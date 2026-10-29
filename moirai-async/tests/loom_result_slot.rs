//! Bounded interleaving models for the production [`ResultCell`] protocol.
//!
//! Unlike the former copied state machine, these models call
//! `moirai_utils::ResultCell` directly. Under `cfg(loom)`, the cell's private
//! waiter-state and waiter-presence atomics use Loom. A test-local [`StateWord`]
//! forwards the result-state operations and their exact orderings to a Loom
//! atomic. The production payloads remain in `core::cell::UnsafeCell`, so Loom
//! does not check their raw accesses; these models cover the atomic ownership,
//! publication, replacement, wakeup, and liveness protocol that guards them.
//!
//! Every model uses a finite four-preemption bound. The covered schedules
//! include completion against check-register-recheck, completion against waker
//! replacement, result consumption versus cell drop, waiter clone/drop
//! accounting, and waiter callbacks that re-enter after the waiter machine has
//! returned to idle.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-async --test loom_result_slot --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty.

#![cfg(loom)]

use core::ptr;
use core::sync::atomic::Ordering;

use loom::sync::Arc;
use loom::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize};
use loom::thread;
use moirai_utils::result_cell::{ResultCell, StateWord, Waiter};

const MODEL_PREEMPTIONS: usize = 4;
const VALUE: u32 = 0xA5A5_5A5A;

/// Loom-backed result state used by the production cell in these models.
struct LoomStateWord(AtomicU8);

// SAFETY: every operation forwards the requested ordering unchanged to one
// Loom atomic. `exclusive_state` reads the same cell through Loom's exclusive
// mutable access API after `ResultCell` has exclusive ownership.
unsafe impl StateWord for LoomStateWord {
    fn pending() -> Self {
        Self(AtomicU8::new(0))
    }

    fn load(&self, order: Ordering) -> u8 {
        self.0.load(order)
    }

    fn store(&self, value: u8, order: Ordering) {
        self.0.store(value, order);
    }

    fn compare_exchange(
        &self,
        current: u8,
        new: u8,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u8, u8> {
        self.0.compare_exchange(current, new, success, failure)
    }

    fn exclusive_state(&mut self) -> u8 {
        self.0.with_mut(|state| *state)
    }
}

struct TrackedValue {
    value: u32,
    drops: Arc<AtomicUsize>,
}

impl Drop for TrackedValue {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct WaiterProbe {
    wakes: AtomicUsize,
    wake_mask: AtomicUsize,
    clones: AtomicUsize,
    drops: AtomicUsize,
    reentries: AtomicUsize,
    cell: AtomicPtr<ModelCell>,
}

impl WaiterProbe {
    fn new() -> Self {
        Self {
            wakes: AtomicUsize::new(0),
            wake_mask: AtomicUsize::new(0),
            clones: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            reentries: AtomicUsize::new(0),
            cell: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

struct ModelWaiter {
    id: usize,
    probe: Arc<WaiterProbe>,
    cloned_waiter_reenters_on_drop: bool,
    reenter_on_drop: bool,
    inspect_on_wake: bool,
}

impl ModelWaiter {
    fn passive(id: usize, probe: &Arc<WaiterProbe>) -> Self {
        Self {
            id,
            probe: Arc::clone(probe),
            cloned_waiter_reenters_on_drop: false,
            reenter_on_drop: false,
            inspect_on_wake: false,
        }
    }
}

impl Clone for ModelWaiter {
    fn clone(&self) -> Self {
        self.probe.clones.fetch_add(1, Ordering::Relaxed);
        Self {
            id: self.id,
            probe: Arc::clone(&self.probe),
            cloned_waiter_reenters_on_drop: false,
            reenter_on_drop: self.cloned_waiter_reenters_on_drop,
            inspect_on_wake: self.inspect_on_wake,
        }
    }
}

impl Drop for ModelWaiter {
    fn drop(&mut self) {
        self.probe.drops.fetch_add(1, Ordering::Relaxed);
        if !self.reenter_on_drop {
            return;
        }

        let cell = self.probe.cell.load(Ordering::Acquire);
        assert!(!cell.is_null(), "reentrant waiter has a live cell");
        let mut replacement = Self::passive(3, &self.probe);
        replacement.inspect_on_wake = true;
        // SAFETY: the outer registration restores the waiter machine to idle
        // before dropping the replaced waiter. This callback is the serialized
        // consumer and the replacement does not recursively re-enter.
        unsafe {
            (*cell).register(&replacement);
        }
        self.probe.reentries.fetch_add(1, Ordering::Relaxed);
    }
}

impl Waiter for ModelWaiter {
    const REPLACE_ON_REPEAT: bool = true;

    fn wake(self) {
        self.probe.wakes.fetch_add(1, Ordering::Relaxed);
        self.probe
            .wake_mask
            .fetch_or(1_usize << self.id, Ordering::Relaxed);

        if self.inspect_on_wake {
            let cell = self.probe.cell.load(Ordering::Acquire);
            assert!(!cell.is_null(), "woken waiter has a live cell");
            // SAFETY: `cell` points to the live `Arc<ModelCell>` installed by
            // the model before registration and cleared only after completion.
            let cell = unsafe { &*cell };
            assert!(cell.is_completed());
            assert!(!cell.has_registered_waiter());
        }
    }
}

type ModelCell = ResultCell<TrackedValue, ModelWaiter, LoomStateWord>;

fn tracked_value(drops: &Arc<AtomicUsize>) -> TrackedValue {
    TrackedValue {
        value: VALUE,
        drops: Arc::clone(drops),
    }
}

fn check_model<F>(model: F)
where
    F: Fn() + Sync + Send + 'static,
{
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(MODEL_PREEMPTIONS);
    builder.check(model);
}

/// The consumer's production check-register-recheck sequence.
fn poll(cell: &ModelCell, waiter: &ModelWaiter) -> Option<TrackedValue> {
    if let Some(value) = cell.try_take_ready() {
        return Some(value);
    }
    // SAFETY: each model has one serialized consumer.
    unsafe {
        cell.register(waiter);
    }
    cell.try_take_ready()
}

/// `complete` races a single check-register-recheck poll.
#[test]
fn complete_races_first_poll() {
    check_model(|| {
        let value_drops = Arc::new(AtomicUsize::new(0));
        let probe = Arc::new(WaiterProbe::new());
        let cell = Arc::new(ModelCell::new());
        let waiter = ModelWaiter::passive(0, &probe);

        let producer = {
            let cell = Arc::clone(&cell);
            let value_drops = Arc::clone(&value_drops);
            thread::spawn(move || cell.complete(tracked_value(&value_drops)))
        };
        let consumer = {
            let cell = Arc::clone(&cell);
            let waiter = waiter.clone();
            thread::spawn(move || poll(&cell, &waiter))
        };

        producer.join().expect("invariant: producer completes");
        let observed = consumer.join().expect("invariant: consumer completes");
        let value = match observed {
            Some(value) => value,
            None => {
                assert_eq!(
                    probe.wakes.load(Ordering::Acquire),
                    1,
                    "a pending poll must be notified before it can run again"
                );
                cell.try_take_ready()
                    .expect("invariant: notification follows result publication")
            }
        };
        assert_eq!(value.value, VALUE);
        assert!(cell.try_take_ready().is_none());
        drop(value);
        drop(cell);
        drop(waiter);

        assert_eq!(value_drops.load(Ordering::Relaxed), 1);
        assert_eq!(
            probe.drops.load(Ordering::Relaxed),
            probe.clones.load(Ordering::Relaxed) + 1
        );
    });
}

/// Completion races replacement of an already-registered waiter.
#[test]
fn complete_races_reregistering_poll() {
    check_model(|| {
        let value_drops = Arc::new(AtomicUsize::new(0));
        let probe = Arc::new(WaiterProbe::new());
        let cell = Arc::new(ModelCell::new());
        let first = ModelWaiter::passive(0, &probe);
        let second = ModelWaiter::passive(1, &probe);

        // SAFETY: this is the only consumer registration before the race.
        unsafe {
            cell.register(&first);
        }
        assert!(cell.has_registered_waiter());

        let producer = {
            let cell = Arc::clone(&cell);
            let value_drops = Arc::clone(&value_drops);
            thread::spawn(move || cell.complete(tracked_value(&value_drops)))
        };
        let replacer = {
            let cell = Arc::clone(&cell);
            let second = second.clone();
            thread::spawn(move || {
                // SAFETY: this is the serialized consumer's next registration.
                unsafe {
                    cell.register(&second);
                }
            })
        };

        producer.join().expect("invariant: producer completes");
        replacer.join().expect("invariant: replacer completes");
        let value = cell
            .try_take_ready()
            .expect("invariant: completion publishes one value");
        assert_eq!(value.value, VALUE);
        assert!(cell.try_take_ready().is_none());
        assert_eq!(probe.wakes.load(Ordering::Relaxed), 1);
        assert!(matches!(probe.wake_mask.load(Ordering::Relaxed), 1 | 2));

        drop(value);
        drop(cell);
        drop(first);
        drop(second);
        assert_eq!(value_drops.load(Ordering::Relaxed), 1);
        assert_eq!(
            probe.drops.load(Ordering::Relaxed),
            probe.clones.load(Ordering::Relaxed) + 2
        );
    });
}

/// Published values and all waiter instances are dropped exactly once.
#[test]
fn value_and_waiter_drop_once() {
    check_model(|| {
        let value_drops = Arc::new(AtomicUsize::new(0));
        let probe = Arc::new(WaiterProbe::new());
        let cell = Arc::new(ModelCell::new());
        let waiter = ModelWaiter::passive(0, &probe);

        let producer = {
            let cell = Arc::clone(&cell);
            let value_drops = Arc::clone(&value_drops);
            thread::spawn(move || cell.complete(tracked_value(&value_drops)))
        };
        let consumer = {
            let cell = Arc::clone(&cell);
            let waiter = waiter.clone();
            thread::spawn(move || {
                // SAFETY: this is the model's only consumer.
                unsafe {
                    cell.register(&waiter);
                }
                drop(cell.try_take_ready());
            })
        };

        producer.join().expect("invariant: producer completes");
        consumer.join().expect("invariant: consumer completes");
        drop(cell);
        drop(waiter);

        assert_eq!(value_drops.load(Ordering::Relaxed), 1);
        assert_eq!(
            probe.drops.load(Ordering::Relaxed),
            probe.clones.load(Ordering::Relaxed) + 1
        );
    });
}

/// Replaced-waiter drop and wake callbacks may re-enter after slot release.
#[test]
fn waiter_callbacks_reenter_after_slot_release() {
    check_model(|| {
        let value_drops = Arc::new(AtomicUsize::new(0));
        let probe = Arc::new(WaiterProbe::new());
        let cell = Arc::new(ModelCell::new());
        probe
            .cell
            .store(Arc::as_ptr(&cell).cast_mut(), Ordering::Release);

        let first = ModelWaiter {
            id: 0,
            probe: Arc::clone(&probe),
            cloned_waiter_reenters_on_drop: true,
            reenter_on_drop: false,
            inspect_on_wake: false,
        };
        let second = ModelWaiter {
            id: 1,
            probe: Arc::clone(&probe),
            cloned_waiter_reenters_on_drop: false,
            reenter_on_drop: false,
            inspect_on_wake: true,
        };

        // SAFETY: both calls belong to one serialized consumer. Dropping the
        // replaced clone re-enters only after the outer call releases the slot.
        unsafe {
            cell.register(&first);
            cell.register(&second);
        }
        assert_eq!(probe.reentries.load(Ordering::Relaxed), 1);

        cell.complete(tracked_value(&value_drops));
        assert_eq!(probe.wakes.load(Ordering::Relaxed), 1);
        let value = cell
            .try_take_ready()
            .expect("invariant: completion publishes one value");
        assert_eq!(value.value, VALUE);

        drop(value);
        drop(first);
        drop(second);
        probe.cell.store(ptr::null_mut(), Ordering::Release);
        drop(cell);
        assert_eq!(value_drops.load(Ordering::Relaxed), 1);
    });
}
