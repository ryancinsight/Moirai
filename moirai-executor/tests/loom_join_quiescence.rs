//! loom model of the `join()` / `notify_quiescent` quiescence handshake.
//!
//! A worker finishing the last active job and a thread in `ThreadScheduler::join`
//! race through a store-buffer (Dekker) handshake across two atoms:
//!   * `active_workers` — the worker publishes quiescence by decrementing it to 0
//!     (`execute_job`, `worker.rs`), then `notify_quiescent` reads `join_waiters`
//!     to decide whether to signal.
//!   * `join_waiters` — the joiner registers by incrementing it (`core.rs::join`),
//!     then re-checks `is_quiescent` (which loads `active_workers`) to decide
//!     whether to park on the condvar.
//!
//! The lost-wakeup outcome: the joiner observes `active != 0` (parks) while the
//! worker observes `join_waiters == 0` (does not signal) — the quiescent
//! scheduler never wakes the joiner, hanging `join()`.
//!
//! This models the *ordering protocol* over loom atoms (loom cannot instrument
//! the production `CacheAligned` storage). As with `loom_wake_handshake.rs`, the
//! SC StoreLoad barrier that production must rely on is represented to loom by an
//! explicit `fence(SeqCst)` between each side's store and its load.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-executor --test loom_join_quiescence --release`

#![cfg(loom)]

use loom::sync::atomic::{fence, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

struct Quiescence {
    /// Mirrors `active_workers`; starts at 1 (the worker running the last job).
    active: AtomicUsize,
    /// Mirrors `join_waiters`.
    join_waiters: AtomicUsize,
}

impl Quiescence {
    fn new() -> Self {
        Self {
            active: AtomicUsize::new(1),
            join_waiters: AtomicUsize::new(0),
        }
    }

    /// Worker half: finish the last job (`active` 1 -> 0), then read
    /// `join_waiters` to decide whether to signal. Returns whether it would
    /// signal the condvar. (`pending` is 0 throughout this scenario.)
    fn finish_last_job_and_maybe_notify(&self) -> bool {
        let was_active = self.active.fetch_sub(1, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        // notify_quiescent: only the last worker to go idle checks join_waiters.
        was_active == 1 && self.join_waiters.load(Ordering::SeqCst) != 0
    }

    /// Joiner half: register interest, then check quiescence. Returns whether it
    /// would park (observed the scheduler as not-yet-quiescent).
    fn register_and_would_park(&self) -> bool {
        self.join_waiters.fetch_add(1, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        // is_quiescent: pending(0) && active == 0.
        self.active.load(Ordering::SeqCst) != 0
    }
}

/// The quiescent scheduler must always wake a registering joiner: the joiner
/// either observes quiescence itself (does not park) or the worker observes the
/// registration (signals). `park && !notify` is the lost wakeup — a hung
/// `join()`. The SC StoreLoad barrier across `active`/`join_waiters` must make it
/// unreachable.
#[test]
fn join_quiescence_never_loses_a_wakeup() {
    loom::model(|| {
        let q = Arc::new(Quiescence::new());

        let worker = {
            let q = q.clone();
            thread::spawn(move || q.finish_last_job_and_maybe_notify())
        };

        let parked = q.register_and_would_park();
        let notified = worker.join().unwrap();

        assert!(
            !(parked && !notified),
            "lost wakeup: join() parked on active!=0 while the quiescent worker did not signal"
        );
    });
}
