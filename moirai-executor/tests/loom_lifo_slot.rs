//! loom exhaustive-interleaving model of the worker `LifoSlot` protocol.
//!
//! `LifoSlot` (`schedule/runtime/types.rs`) is the single-slot LIFO fast path in
//! front of each worker's deque: the owning worker `push`es and `pop`s it, and
//! thieves `steal` from it. It is `unsafe impl Sync` and moves the job in and out
//! with `ptr::read`/`MaybeUninit`, so the load-bearing safety property is that a
//! published job is handed to **exactly one** taker. Two takers reading the same
//! slot would `ptr::read` the same `ScheduledJob` twice — a double-move /
//! use-after-free / double-free, i.e. the heap-corruption class this crate's
//! nested-scope work has been chasing.
//!
//! The slot is a four-state machine over one `AtomicU8`:
//!   0 EMPTY · 1 LOCKED (owner mid-push/pop) · 2 READY · 3 STEALING (thief).
//! Mutual exclusion is enforced by CAS on the `state`:
//!   * `push`  (empty):   CAS 0->1 (Acquire) · write · store 2 (Release)
//!   * `push`  (replace): CAS 2->1 (AcqRel)  · read old + write · store 2 (Release)
//!   * `pop`:             load==2 · CAS 2->1 (Acquire) · read · store 0 (Release)
//!   * `steal`:           load==2 · CAS 2->3 (Acquire) · read · store 0 (Release)
//!
//! Only the thread that wins the `2->{1,3}` CAS reaches the read, so at most one
//! taker consumes the READY job. The `Acquire` on the take CAS pairs with the
//! `Release` `store(2)` of the publishing push (job write happens-before the
//! read); the `Acquire` on push's `0->1` CAS pairs with the `Release` `store(0)`
//! of the prior taker (the taker's read happens-before the next push's write, so
//! a refill never races an in-flight take).
//!
//! This file mirrors that protocol over loom atoms and a `loom::cell::UnsafeCell`
//! (loom cannot instrument the production `MaybeUninit`/`fn`-pointer storage
//! directly); loom enumerates every interleaving, its `UnsafeCell` flags any
//! concurrent access, and the assertions pin "exactly one taker, nothing lost."
//! Keep the orderings here in sync with `LifoSlot` in `types.rs`.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-executor --test loom_lifo_slot --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty, so it
//! never affects the standard test suite or pulls in the `loom` dependency.

#![cfg(loom)]

use loom::cell::UnsafeCell;
use loom::sync::atomic::{AtomicU8, Ordering};
use loom::sync::Arc;
use loom::thread;

const EMPTY: u8 = 0;
const LOCKED: u8 = 1;
const READY: u8 = 2;
const STEALING: u8 = 3;

/// Faithful model of `LifoSlot`, storing a `u32` token instead of a
/// `ScheduledJob`. A value of `0` is never published, so it doubles as an
/// "uninitialized" sentinel for post-run inspection.
struct Slot {
    state: AtomicU8,
    job: UnsafeCell<u32>,
}

impl Slot {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(EMPTY),
            job: UnsafeCell::new(0),
        }
    }

    /// Mirrors `LifoSlot::push`: store into an empty slot (returns `None`) or
    /// replace a READY slot (returns the evicted job); on contention the job is
    /// handed back to the caller (`Some(v)`), which production routes to the
    /// deque.
    fn push(&self, v: u32) -> Option<u32> {
        let current = self.state.load(Ordering::Relaxed);
        if current == EMPTY {
            if self
                .state
                .compare_exchange(EMPTY, LOCKED, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                self.job.with_mut(|p| unsafe { *p = v });
                self.state.store(READY, Ordering::Release);
                return None;
            }
        } else if current == READY
            && self
                .state
                .compare_exchange(READY, LOCKED, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            let old = self.job.with_mut(|p| unsafe { core::mem::replace(&mut *p, v) });
            self.state.store(READY, Ordering::Release);
            return Some(old);
        }
        Some(v)
    }

    /// Mirrors `LifoSlot::pop`: owner take via the `2->1` CAS.
    fn pop(&self) -> Option<u32> {
        if self.state.load(Ordering::Relaxed) == READY
            && self
                .state
                .compare_exchange(READY, LOCKED, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            let v = self.job.with_mut(|p| unsafe { *p });
            self.state.store(EMPTY, Ordering::Release);
            Some(v)
        } else {
            None
        }
    }

    /// Mirrors `LifoSlot::steal`: thief take via the `2->3` CAS.
    fn steal(&self) -> Option<u32> {
        if self.state.load(Ordering::Relaxed) == READY
            && self
                .state
                .compare_exchange(READY, STEALING, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            let v = self.job.with_mut(|p| unsafe { *p });
            self.state.store(EMPTY, Ordering::Release);
            Some(v)
        } else {
            None
        }
    }

    /// Single-threaded post-run inspection: the job still resident in a READY
    /// slot, if any.
    fn resident(&self) -> Option<u32> {
        if self.state.load(Ordering::Acquire) == READY {
            Some(self.job.with(|p| unsafe { *p }))
        } else {
            None
        }
    }
}

/// Owner `pop` racing a thief `steal` over one published job: the job must be
/// taken by exactly one of them (never both — that is the double-`ptr::read`
/// heap-corruption outcome — and never neither).
#[test]
fn pop_and_steal_take_the_job_exactly_once() {
    loom::model(|| {
        let slot = Arc::new(Slot::new());
        assert!(slot.push(42).is_none(), "publish into an empty slot");

        let thief = {
            let slot = slot.clone();
            thread::spawn(move || slot.steal())
        };
        let popped = slot.pop();
        let stolen = thief.join().unwrap();

        match (popped, stolen) {
            (Some(v), None) | (None, Some(v)) => assert_eq!(v, 42, "wrong job taken"),
            (None, None) => panic!("job lost: neither pop nor steal took the published job"),
            (Some(_), Some(_)) => {
                panic!("double take: pop and steal both consumed the same slot")
            }
        }
        assert_eq!(slot.resident(), None, "slot must be empty after the take");
    });
}

/// Two thieves stealing the same published job: exactly one wins the `2->3` CAS.
#[test]
fn concurrent_steals_take_the_job_exactly_once() {
    loom::model(|| {
        let slot = Arc::new(Slot::new());
        assert!(slot.push(7).is_none());

        let a = {
            let slot = slot.clone();
            thread::spawn(move || slot.steal())
        };
        let b = {
            let slot = slot.clone();
            thread::spawn(move || slot.steal())
        };
        let (ra, rb) = (a.join().unwrap(), b.join().unwrap());

        match (ra, rb) {
            (Some(v), None) | (None, Some(v)) => assert_eq!(v, 7),
            (None, None) => panic!("job lost: neither thief stole the published job"),
            (Some(_), Some(_)) => panic!("double steal: two thieves consumed the same slot"),
        }
    });
}

/// Owner replace-`push` racing a thief `steal` over a resident job: the old job
/// must be consumed exactly once (by whichever wins), the new job must never be
/// lost (it is either resident or handed back to the pusher), and the two jobs
/// never alias.
#[test]
fn replace_push_and_steal_conserve_both_jobs() {
    const OLD: u32 = 1;
    const NEW: u32 = 2;
    loom::model(|| {
        let slot = Arc::new(Slot::new());
        assert!(slot.push(OLD).is_none());

        let thief = {
            let slot = slot.clone();
            thread::spawn(move || slot.steal())
        };
        let evicted = slot.push(NEW);
        let stolen = thief.join().unwrap();

        // Every job that left the system, plus whatever remains resident, must
        // be exactly {OLD, NEW} — no loss, no duplication (a duplicate is the
        // double-take heap hazard).
        let mut seen: Vec<u32> = Vec::new();
        seen.extend(evicted);
        seen.extend(stolen);
        seen.extend(slot.resident());
        seen.sort_unstable();
        assert_eq!(
            seen,
            vec![OLD, NEW],
            "replace/steal must conserve both jobs exactly once (evicted/stolen/resident)"
        );
    });
}
