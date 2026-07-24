//! loom exhaustive-interleaving model of `AsyncResultSlot`'s producer/consumer
//! protocol (`moirai-async/src/executor/result_slot.rs`).
//!
//! The production slot backs its `result`/`waiter` payload with
//! `UnsafeCell<MaybeUninit<_>>`; this file models the *state protocol* — the
//! six-state `AtomicU8` and the exact `Acquire`/`Release`/`Relaxed` orderings of
//! `complete`, `try_take_ready`, and `register_waker` — over loom-tracked cells,
//! plus the `check -> register -> re-check` sequence that `AsyncHandle::poll`
//! wraps them in. loom enumerates every producer/consumer interleaving and checks
//! the two invariants the hand proof in `result_slot.rs` claims:
//!
//! - **exactly-once**: the result is delivered once — never lost, never taken
//!   twice;
//! - **no lost wakeup**: a consumer that parks (polls to `Pending`) is always
//!   woken, so it is guaranteed to be re-polled and take the result.
//!
//! loom's `UnsafeCell` additionally fails the run on any unsynchronized access to
//! the `result`/`waiter` cells, so the model also proves the state transitions
//! serialize every cell access (the property the per-site `// Safety:` comments
//! assert).
//!
//! Keep the orderings here in sync with the production code: any change to the
//! `Acquire`/`Release`/`Relaxed` on `state`, or to the poll sequence, must be
//! mirrored.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p moirai-async --test loom_result_slot --release`
//!
//! Under a normal build the `#![cfg(loom)]` gate makes this file empty, so it
//! never affects the standard test suite or pulls in the `loom` dependency.

#![cfg(loom)]

use loom::cell::UnsafeCell;
use loom::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use loom::sync::Arc;
use loom::thread;

const PENDING: u8 = 0;
const WAITING: u8 = 1;
const UPDATING_WAKER: u8 = 2;
const WRITING: u8 = 3;
const READY: u8 = 4;
const TAKEN: u8 = 5;

const VALUE: u32 = 0xA5A5_5A5A;

/// Model of `AsyncResultSlot<u32>`: the `state` atomic plus the two payload
/// cells, using the production orderings. `woken` models the effect of
/// `Waker::wake` so the no-lost-wakeup invariant is observable after the run.
struct Slot {
    result: UnsafeCell<Option<u32>>,
    state: AtomicU8,
    /// `true` == a waker is currently registered in the cell.
    waiter: UnsafeCell<bool>,
    woken: AtomicBool,
}

impl Slot {
    fn new() -> Self {
        Self {
            result: UnsafeCell::new(None),
            state: AtomicU8::new(PENDING),
            waiter: UnsafeCell::new(false),
            woken: AtomicBool::new(false),
        }
    }

    /// Mirror of `AsyncResultSlot::complete` + `begin_completion`. Returns whether
    /// it observed a registered waiter (the `waiting` branch).
    fn complete(&self, value: u32) -> bool {
        let waiting = loop {
            match self.state.load(Ordering::Acquire) {
                PENDING => {
                    if self
                        .state
                        .compare_exchange(PENDING, WRITING, Ordering::Relaxed, Ordering::Acquire)
                        .is_ok()
                    {
                        break false;
                    }
                }
                WAITING => {
                    if self
                        .state
                        .compare_exchange(WAITING, WRITING, Ordering::Acquire, Ordering::Acquire)
                        .is_ok()
                    {
                        break true;
                    }
                }
                // The consumer holds the waiter cell mid-swap; yield so loom
                // advances the consumer instead of enumerating unbounded spin
                // iterations (the production code spins with `spin_loop`).
                UPDATING_WAKER => thread::yield_now(),
                // A single producer completes once, so it never observes its own
                // WRITING or the post-completion READY/TAKEN at the loop head.
                other => unreachable!("producer saw unexpected state {other}"),
            }
        };

        // WRITING is exclusive to this producer: initialize the result cell.
        self.result.with_mut(|p| unsafe { *p = Some(value) });
        self.state.store(READY, Ordering::Release);

        if waiting {
            // WAITING -> WRITING acquired the consumer's release of the waiter;
            // read it exactly once and model the wake.
            let registered = self.waiter.with_mut(|p| unsafe {
                let was = *p;
                *p = false; // assume_init_read moves the waker out of the cell
                was
            });
            assert!(registered, "producer read an unregistered waiter cell");
            self.woken.store(true, Ordering::Release);
        }
        waiting
    }

    /// Mirror of `AsyncResultSlot::try_take_ready`.
    fn try_take_ready(&self) -> Option<u32> {
        if self
            .state
            .compare_exchange(READY, TAKEN, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            // READY -> TAKEN is the unique consumer transition: read once.
            self.result.with_mut(|p| unsafe { (*p).take() })
        } else {
            None
        }
    }

    /// Mirror of `AsyncResultSlot::register_waker`.
    fn register_waker(&self) {
        loop {
            match self.state.load(Ordering::Acquire) {
                PENDING => {
                    self.waiter.with_mut(|p| unsafe { *p = true });
                    if self
                        .state
                        .compare_exchange(PENDING, WAITING, Ordering::Release, Ordering::Acquire)
                        .is_ok()
                    {
                        return;
                    }
                    // Publish failed: no producer observed this write, so drop it.
                    self.waiter.with_mut(|p| unsafe { *p = false });
                }
                WAITING => {
                    if self
                        .state
                        .compare_exchange(
                            WAITING,
                            UPDATING_WAKER,
                            Ordering::Acquire,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        // UPDATING_WAKER excludes the producer: swap the stale
                        // waker for the fresh one.
                        self.waiter.with_mut(|p| unsafe {
                            *p = false; // drop stale
                            *p = true; // write fresh
                        });
                        self.state.store(WAITING, Ordering::Release);
                        return;
                    }
                }
                // WRITING (producer writing) or UPDATING_WAKER (unreachable with a
                // single consumer): yield so loom advances the producer instead of
                // enumerating unbounded spin iterations.
                WRITING | UPDATING_WAKER => thread::yield_now(),
                // READY / TAKEN: the result is already available; do not register.
                _ => return,
            }
        }
    }
}

/// The consumer's `AsyncHandle::poll` body: check, register, re-check.
fn poll(slot: &Slot) -> Option<u32> {
    if let Some(v) = slot.try_take_ready() {
        return Some(v);
    }
    slot.register_waker();
    slot.try_take_ready()
}

/// Assert exactly-once delivery and no lost wakeup after both threads join.
///
/// `took` is what the consumer's `poll` returned across the run.
fn assert_delivered_once(slot: &Slot, took: Option<u32>) {
    match took {
        Some(v) => {
            // Delivered straight to a poll. Exactly-once: nothing remains.
            assert_eq!(v, VALUE);
            assert_eq!(slot.try_take_ready(), None, "result taken twice");
        }
        None => {
            // Parked. No lost wakeup: the consumer must have been woken, and the
            // guaranteed re-poll takes the single result.
            assert!(
                slot.woken.load(Ordering::Relaxed),
                "lost wakeup: consumer parked but was never woken"
            );
            assert_eq!(slot.try_take_ready(), Some(VALUE), "result lost");
            assert_eq!(slot.try_take_ready(), None, "result taken twice");
        }
    }
}

/// `complete` races a single `poll` (check -> register -> re-check).
#[test]
fn complete_races_first_poll() {
    loom::model(|| {
        let slot = Arc::new(Slot::new());

        let producer = {
            let slot = Arc::clone(&slot);
            thread::spawn(move || slot.complete(VALUE))
        };
        let consumer = {
            let slot = Arc::clone(&slot);
            thread::spawn(move || poll(&slot))
        };

        producer.join().unwrap();
        let took = consumer.join().unwrap();

        // Both children joined: the main thread now has exclusive access.
        assert_delivered_once(&slot, took);
    });
}

/// `complete` races a consumer that polls twice, so the second poll re-registers
/// through `WAITING -> UPDATING_WAKER -> WAITING` against the producer — exercising
/// the waiter-swap-under-lock and the producer's spin past `UPDATING_WAKER`.
#[test]
fn complete_races_reregistering_poll() {
    loom::model(|| {
        let slot = Arc::new(Slot::new());

        let producer = {
            let slot = Arc::clone(&slot);
            thread::spawn(move || slot.complete(VALUE))
        };
        let consumer = {
            let slot = Arc::clone(&slot);
            thread::spawn(move || {
                if let Some(v) = poll(&slot) {
                    return Some(v);
                }
                // A spurious re-poll is always permitted; here it drives the
                // re-registration path.
                poll(&slot)
            })
        };

        producer.join().unwrap();
        let took = consumer.join().unwrap();

        assert_delivered_once(&slot, took);
    });
}
