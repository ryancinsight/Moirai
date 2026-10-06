//! loom exhaustive-interleaving model of two waiters racing one completion.
//!
//! `TaskState::register_waker` stores a waker in the task's slot under the slot
//! mutex, then re-checks the completion flag; `mark_completed_since` publishes
//! the completion flag, then takes the slot under the same mutex and wakes what
//! it took. Two waiters on one task share that slot (the second turns it into a
//! fan-out waker owning both, `registry/fan_out.rs`), so the slot's content is
//! modeled as the set of registered waiter ids.
//!
//! The danger is a stranded waiter: it registered, its recheck saw the task
//! still pending, and completion drained the slot before its registration —
//! or drained without it. Either the recheck sees the completion, or the
//! completion's drain sees the registration; loom confirms no interleaving
//! leaves a waiter with neither.
//!
//! The model covers the ordering protocol over loom-tracked atoms and a loom
//! mutex; loom cannot instrument the production `TaskState` storage, and the
//! fan-out waker's vtable is covered by ordinary unit tests. Keep the orderings
//! in sync with `registry/state.rs`.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-executor --test loom_task_waiters --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty.

#![cfg(loom)]
#![allow(clippy::unwrap_used, reason = "test scope")]

use loom::sync::atomic::{AtomicBool, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;

const WAITERS: usize = 2;

/// The completion flag, the slot's registered waiters, and which were woken.
struct Task {
    completed: AtomicBool,
    slot: Mutex<Vec<usize>>,
    woken: [AtomicBool; WAITERS],
}

impl Task {
    fn new() -> Self {
        Self {
            completed: AtomicBool::new(false),
            slot: Mutex::new(Vec::new()),
            woken: [AtomicBool::new(false), AtomicBool::new(false)],
        }
    }

    /// Take everything in the slot and wake it, outside the slot lock.
    fn take_and_wake(&self) {
        let taken = std::mem::take(&mut *self.slot.lock().unwrap());
        for waiter in taken {
            self.woken[waiter].store(true, Ordering::Release);
        }
    }

    /// Mirrors `TaskState::register_waker`; returns whether the recheck saw the
    /// completion.
    fn register(&self, waiter: usize) -> bool {
        self.slot.lock().unwrap().push(waiter);
        let completed = self.completed.load(Ordering::Acquire);
        if completed {
            self.take_and_wake();
        }
        completed
    }

    /// Mirrors `TaskState::mark_completed_since`.
    fn complete(&self) {
        self.completed.store(true, Ordering::Release);
        self.take_and_wake();
    }
}

#[test]
fn two_waiters_are_never_stranded_by_completion() {
    loom::model(|| {
        let task = Arc::new(Task::new());

        let waiters: Vec<_> = (0..WAITERS)
            .map(|waiter| {
                let task = Arc::clone(&task);
                thread::spawn(move || task.register(waiter))
            })
            .collect();

        task.complete();

        // Join every thread first: a wake recorded by another waiter's thread is
        // visible only once that thread is joined.
        let outcomes: Vec<bool> = waiters
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        for (waiter, saw_completion) in outcomes.into_iter().enumerate() {
            assert!(
                saw_completion || task.woken[waiter].load(Ordering::Acquire),
                "waiter {waiter} registered, saw a pending task, and was never woken"
            );
        }
    });
}
