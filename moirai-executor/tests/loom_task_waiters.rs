//! loom model of owned waiter registration racing task completion.
//!
//! `TaskState::register_waker` publishes a lazily allocated waiter set through
//! `WaiterSetSlot`'s atomic lock. Completion publishes its state, detaches that
//! set under the same lock, closes registration, and claims each occupied slot
//! separately. An owner may remove its slot until notification claims it.
//!
//! Two bounded scenarios use at most two identities. One owner unregisters
//! before completion can claim it and must not be notified. On a fresh task,
//! one live registration races completion across lazy set publication; it
//! either observes completion or is accepted and then notified.
//!
//! loom tracks the slot lock, set mutex, and completion/wake atomics. It cannot
//! instrument the production registry block, `std::sync::Mutex`, or callback
//! vtables. Keep the protocol in sync with `registry/state.rs` and
//! `registry/waiter.rs`.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo nextest run --locked -p moirai-executor`
//! `--test loom_task_waiters --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty.

#![cfg(loom)]
#![allow(clippy::unwrap_used, reason = "test scope")]

use loom::cell::UnsafeCell;
use loom::sync::atomic::{AtomicBool, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

const WAITERS: usize = 2;

/// The completion state, lazily published waiter set, and observed wakes.
struct Task {
    completed: AtomicBool,
    waiters: WaiterSetSlot,
    woken: [AtomicBool; WAITERS],
}

/// Faithful bounded model of the production atomic slot lock and set pointer.
struct WaiterSetSlot {
    locked: AtomicBool,
    set: UnsafeCell<Option<Arc<WaiterSet>>>,
}

struct WaiterSetSlotGuard<'a> {
    slot: &'a WaiterSetSlot,
}

impl Drop for WaiterSetSlotGuard<'_> {
    fn drop(&mut self) {
        self.slot.locked.store(false, Ordering::Release);
    }
}

impl WaiterSetSlot {
    fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
            set: UnsafeCell::new(None),
        }
    }

    fn lock(&self) -> WaiterSetSlotGuard<'_> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            thread::yield_now();
        }
        WaiterSetSlotGuard { slot: self }
    }

    fn get_or_create(&self, completed: impl FnOnce() -> bool) -> Option<Arc<WaiterSet>> {
        let guard = self.lock();
        if completed() {
            drop(guard);
            return None;
        }
        let set = self.set.with_mut(|set| {
            // SAFETY: `guard` makes this the only access to `set`.
            let set = unsafe { &mut *set };
            Arc::clone(set.get_or_insert_with(|| Arc::new(WaiterSet::new())))
        });
        drop(guard);
        Some(set)
    }

    fn take(&self) -> Option<Arc<WaiterSet>> {
        let guard = self.lock();
        let set = self.set.with_mut(|set| {
            // SAFETY: `guard` makes this the only access to `set`.
            unsafe { (&mut *set).take() }
        });
        drop(guard);
        set
    }
}

struct WaiterSet {
    slots: Mutex<WaiterSlots>,
}

struct WaiterSlots {
    closed: bool,
    entries: [Option<usize>; WAITERS],
}

impl WaiterSet {
    fn new() -> Self {
        Self {
            slots: Mutex::new(WaiterSlots {
                closed: false,
                entries: [None; WAITERS],
            }),
        }
    }

    fn register(set: &Arc<Self>, waiter: usize) -> Option<Waiter> {
        let mut slots = set.slots.lock().unwrap();
        if slots.closed {
            return None;
        }
        let key = slots.entries.iter().position(Option::is_none)?;
        slots.entries[key] = Some(waiter);
        Some(Waiter {
            set: Arc::clone(set),
            key,
            registered: true,
        })
    }

    fn remove(&self, key: usize) -> Option<usize> {
        self.slots.lock().unwrap().entries[key].take()
    }

    fn close(set: &Arc<Self>) -> Notifications {
        set.slots.lock().unwrap().closed = true;
        Notifications {
            set: Arc::clone(set),
        }
    }
}

struct Waiter {
    set: Arc<WaiterSet>,
    key: usize,
    registered: bool,
}

impl Waiter {
    fn unregister(mut self) -> bool {
        let removed = self.set.remove(self.key).is_some();
        self.registered = false;
        removed
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        if self.registered {
            assert!(
                self.set.remove(self.key).is_none(),
                "an accepted live waiter must be claimed before owner drop"
            );
        }
    }
}

struct Notifications {
    set: Arc<WaiterSet>,
}

impl Notifications {
    fn notify(self, woken: &[AtomicBool; WAITERS]) {
        for key in 0..WAITERS {
            if let Some(waiter) = self.set.remove(key) {
                woken[waiter].store(true, Ordering::Release);
            }
        }
    }
}

impl Task {
    fn new() -> Self {
        Self {
            completed: AtomicBool::new(false),
            waiters: WaiterSetSlot::new(),
            woken: [AtomicBool::new(false), AtomicBool::new(false)],
        }
    }

    /// Mirrors `TaskState::register_waker`.
    fn register(&self, waiter: usize) -> Option<Waiter> {
        if self.completed.load(Ordering::Acquire) {
            return None;
        }
        let set = self
            .waiters
            .get_or_create(|| self.completed.load(Ordering::Acquire))?;
        WaiterSet::register(&set, waiter)
    }

    /// Mirrors publication, detachment, closing, and deferred notification.
    fn complete(&self) {
        self.completed.store(true, Ordering::Release);
        if let Some(set) = self.waiters.take() {
            WaiterSet::close(&set).notify(&self.woken);
        }
    }
}

#[test]
fn removed_waiter_stays_quiet() {
    loom::model(|| {
        let task = Arc::new(Task::new());

        let removed = task
            .register(0)
            .expect("completion has not started")
            .unregister();
        assert!(removed, "the owner removes its slot before notification");

        task.complete();

        assert!(
            !task.woken[0].load(Ordering::Acquire),
            "an unregistered waiter was notified"
        );
    });
}

#[test]
fn live_waiter_is_not_stranded_during_publication() {
    loom::model(|| {
        let task = Arc::new(Task::new());

        let live = {
            let task = Arc::clone(&task);
            thread::spawn(move || task.register(1))
        };

        task.complete();
        let live_waiter = live.join().unwrap();

        assert!(
            live_waiter.is_none() || task.woken[1].load(Ordering::Acquire),
            "a live waiter was accepted but never notified"
        );
    });
}
