//! One-shot completion cell: one result, one waiter, one hand-off.
//!
//! `ResultCell` carries a producer's single output to a single consumer and
//! wakes the parked waiter across that hand-off. It is the completion path
//! shared by `moirai-core`'s blocking `TaskResultSlot` and `moirai-async`'s
//! `AsyncResultSlot`: both need this atomic hand-off, and neither may reintroduce
//! a lock or a per-task waker map on the result path.
//!
//! # Roles
//!
//! The cell is reached through an `Arc` shared by exactly two owners:
//!
//! - the **producer** calls [`complete`](ResultCell::complete) exactly once — the tail
//!   of a spawned unit of work;
//! - the **consumer** calls [`try_take_ready`](ResultCell::try_take_ready) and
//!   [`register`](ResultCell::register) from its own poll or park loop.
//!
//! Because those are the only two owners, the `result` and `waiter` cells need
//! no lock. The type does not enforce that, so [`register`](ResultCell::register)
//! is `unsafe` and its caller vouches for the single consumer. The producer runs
//! once; the consumer is serialized with itself (`poll` takes `Pin<&mut Self>`
//! on the async side, and the blocking side registers once per wait); and `Drop`
//! runs only after the last `Arc`, so it has exclusive access and races neither
//! side.
//!
//! # State machines
//!
//! The result and waiter have independent atomic machines so registering a
//! waiter never waits for producer progress (C = consumer, P = producer):
//!
//! ```text
//!   PENDING ──P complete──▶ WRITING ──P publish──▶ READY ──C take──▶ TAKEN
//!
//!   IDLE ──C register──▶ REGISTERING ──C publish──▶ IDLE
//!     │                       │
//!     │ P wake                │ P requests wake
//!     ▼                       ▼
//!   WAKING                 REGISTERING | WAKING
//!     │                       │
//!     └──── owner wakes ──────┴───────────────▶ IDLE
//! ```
//!
//! `WRITING` is the producer's exclusive claim on `result`, and `READY`
//! publishes it before the producer touches the waiter machine. `REGISTERING`
//! is the consumer's exclusive claim on the waiter slot. A producer that sees
//! that claim sets `WAKING` and returns; the consumer then publishes, removes,
//! and wakes the new waiter. No path spins for the other role.
//!
//! # Cell-access invariants (what the per-site `// Safety:` comments rely on)
//!
//! 1. **`result`: written once, read once.** Only the producer writes it, only
//!    under `WRITING` (entered by winning the producer transition, so no
//!    consumer can see it yet). It is read exactly once — by the unique
//!    `READY -> TAKEN` consumer transition, or by `Drop` at `READY` when the
//!    consumer never took it.
//! 2. **`waiter`: accessed only by the waiter-machine owner.** The consumer owns
//!    the slot after `IDLE -> REGISTERING`; the producer owns it when `fetch_or`
//!    observes `IDLE`. If the producer instead observes `REGISTERING`, it marks
//!    the pending wake and leaves the slot to the consumer. Both roles restore
//!    `IDLE` before waking or dropping a waiter, because those operations may
//!    re-enter the cell. A separate atomic presence flag mirrors the slot only
//!    for safe diagnostics; it grants no access to the waiter.
//! 3. **No lost wakeup.** [`complete`](ResultCell::complete) publishes `READY`
//!    before it attempts waiter notification. Registration that races with
//!    completion may observe `READY` or `WAKING` and install nothing. Liveness
//!    therefore requires the consumer to check → [`register`](ResultCell::register)
//!    → **re-check**; the re-check observes `READY` even when no waiter was
//!    installed or woken. It is load-bearing, not defensive.
//!
//! # Ordering
//!
//! `READY` publishes the result with `Release`, and `READY -> TAKEN` reads it
//! with `Acquire`. The waiter machine claims with `Acquire`, publishes with
//! `Release`, and uses `AcqRel` when the producer both observes registration and
//! requests a wake. These edges cover separate cells; neither role waits for the
//! other to advance.
//!
//! # The waiter payload
//!
//! [`Waiter`] abstracts the parked handle: `thread::Thread` for the blocking
//! side, `Waker` for the async one. The two differ in exactly one behaviour, and
//! it is a trait constant rather than a branch —
//! [`REPLACE_ON_REPEAT`](Waiter::REPLACE_ON_REPEAT) is `false` for a thread,
//! which parks once per wait and must not be overwritten, and `true` for a
//! waker, which a re-poll may legitimately replace. A mixed waiter type may opt
//! into replacement so a blocking join can replace an earlier async waker.
//!
//! # Layout
//!
//! The cell is packed by default: the state word sits beside the two cells with
//! no padding, which is what a per-task allocation wants. A caller that wants the
//! state in an interference sector of its own supplies that as the
//! [`StateWord`] — `moirai-core`'s `TaskResultSlot` passes
//! `CacheAligned<AtomicU8>` — and the machine runs identically either way, since
//! the alignment never reaches the protocol.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(not(loom))]
use core::sync::atomic::{AtomicBool as WaiterAtomicBool, AtomicU8 as WaiterAtomicU8};
#[cfg(loom)]
use loom::sync::atomic::{AtomicBool as WaiterAtomicBool, AtomicU8 as WaiterAtomicU8};

mod state_word;

pub use state_word::StateWord;

const RESULT_PENDING: u8 = 0;
const RESULT_WRITING: u8 = 3;
const RESULT_READY: u8 = 4;
const RESULT_TAKEN: u8 = 5;

const WAITER_IDLE: u8 = 0;
const WAITER_REGISTERING: u8 = 1;
const WAITER_WAKING: u8 = 2;
const WAITER_REGISTERING_AND_WAKING: u8 = WAITER_REGISTERING | WAITER_WAKING;

/// A handle that can be parked on and woken once.
///
/// See the [module docs](self#the-waiter-payload) for the one behaviour the two
/// implementations differ in.
pub trait Waiter: Send + Clone + Sized {
    /// Whether registering again while one is already parked replaces it.
    ///
    /// `false` for a thread handle: the blocking side registers once per wait, so
    /// a second registration is a no-op rather than an overwrite. `true` for a
    /// waker: a re-poll may carry a different waker and the newest must win.
    const REPLACE_ON_REPEAT: bool;

    /// Wake the parked waiter.
    fn wake(self);
}

/// A thread parks once per wait, so a repeat registration is a no-op.
#[cfg(feature = "std")]
impl Waiter for std::thread::Thread {
    const REPLACE_ON_REPEAT: bool = false;

    fn wake(self) {
        std::thread::Thread::unpark(&self);
    }
}

/// A waker may be replaced between polls, so the newest registration wins.
impl Waiter for core::task::Waker {
    const REPLACE_ON_REPEAT: bool = true;

    fn wake(self) {
        core::task::Waker::wake(self);
    }
}

/// Single-producer, single-consumer completion cell for one unit of work.
///
/// The state word's storage is a parameter ([`StateWord`]), so each caller keeps
/// the layout it needs; see [the layout note](self#layout).
pub struct ResultCell<T, W: Waiter, S: StateWord = AtomicU8> {
    result: UnsafeCell<MaybeUninit<T>>,
    state: S,
    waiter_state: WaiterAtomicU8,
    waiter_present: WaiterAtomicBool,
    waiter: UnsafeCell<Option<W>>,
}

// Safety: the cell has one producer and one consumer. Atomic states serialize
// result publication, result consumption, and waiter updates, so concurrent
// `&self` use is sound exactly when the payload and the waiter may each move
// between threads.
unsafe impl<T: Send, W: Waiter, S: StateWord> Send for ResultCell<T, W, S> {}

// Safety: shared access is mediated by the state machine; the result and waiter
// cells are touched only after the corresponding atomic transition succeeds.
unsafe impl<T: Send, W: Waiter, S: StateWord> Sync for ResultCell<T, W, S> {}

impl<T, W: Waiter, S: StateWord> ResultCell<T, W, S> {
    /// Create an empty cell.
    #[must_use]
    pub fn new() -> Self {
        Self {
            result: UnsafeCell::new(MaybeUninit::uninit()),
            state: S::pending(),
            waiter_state: WaiterAtomicU8::new(WAITER_IDLE),
            waiter_present: WaiterAtomicBool::new(false),
            waiter: UnsafeCell::new(None),
        }
    }

    /// Publish the result, waking the parked waiter if one is registered.
    ///
    /// Called exactly once by the producer; a second call is a no-op.
    pub fn complete(&self, result: T) {
        if !self.begin_completion() {
            return;
        }

        // Safety: WRITING is reachable only after `begin_completion` wins the
        // producer transition, so no consumer can read this cell yet.
        unsafe {
            (*self.result.get()).write(result);
        }

        self.state.store(RESULT_READY, Ordering::Release);
        self.wake_waiter();
    }

    /// Take the result if the producer has published it.
    pub fn try_take_ready(&self) -> Option<T> {
        if self
            .state
            .compare_exchange(
                RESULT_READY,
                RESULT_TAKEN,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // Safety: READY is published only after the producer initializes the
            // result cell; READY -> TAKEN is a unique consumer transition.
            Some(unsafe { (*self.result.get()).assume_init_read() })
        } else {
            None
        }
    }

    /// Take the result only if a relaxed load already observed it, for a spin
    /// loop that re-checks: the load is the hint, [`try_take_ready`](ResultCell::try_take_ready)
    /// is the decision.
    pub fn try_take_observed_ready(&self) -> Option<T> {
        if self.state.load(Ordering::Relaxed) == RESULT_READY {
            self.try_take_ready()
        } else {
            None
        }
    }

    /// Whether the producer has published a result.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        self.state.load(Ordering::Acquire) == RESULT_READY
    }

    /// Whether a waiter is parked and no result has been published yet.
    ///
    /// The complement of [`is_completed`](ResultCell::is_completed) would also
    /// answer true before any registration, so this reports the waiting state
    /// itself: it is what distinguishes a registration that took effect from
    /// one that never ran.
    #[must_use]
    pub fn has_registered_waiter(&self) -> bool {
        !matches!(
            self.state.load(Ordering::Acquire),
            RESULT_READY | RESULT_TAKEN
        ) && self.waiter_present.load(Ordering::Acquire)
    }

    /// Park a clone of `waiter`, or replace the parked one when
    /// [`REPLACE_ON_REPEAT`](Waiter::REPLACE_ON_REPEAT) is set.
    ///
    /// Registration makes one non-blocking attempt. It clones before claiming
    /// the waiter slot, and it drops that clone if the producer already owns the
    /// wake path. The caller must re-check
    /// [`try_take_ready`](ResultCell::try_take_ready) after this returns: a
    /// producer that wins the race publishes the result before it makes the
    /// waiter unavailable. See the [module docs](self#cell-access-invariants).
    ///
    /// # Safety
    ///
    /// No other call to `register` may run on this cell concurrently. The cell
    /// has one consumer by construction (a blocking join takes the handle by
    /// value; an async poll takes `Pin<&mut Self>`), and the caller is that
    /// consumer. `waiter.clone()` may run arbitrary code but must not recursively
    /// register another consumer on this cell.
    pub unsafe fn register(&self, waiter: &W) {
        if matches!(
            self.state.load(Ordering::Acquire),
            RESULT_READY | RESULT_TAKEN
        ) {
            return;
        }

        // Clone before claiming the waiter slot. Clone may panic or run
        // user-defined code, neither of which may strand REGISTERING.
        let replacement = waiter.clone();

        if matches!(
            self.state.load(Ordering::Acquire),
            RESULT_READY | RESULT_TAKEN
        ) {
            drop(replacement);
            return;
        }

        if self
            .waiter_state
            .compare_exchange(
                WAITER_IDLE,
                WAITER_REGISTERING,
                Ordering::Acquire,
                Ordering::Acquire,
            )
            .is_err()
        {
            // A producer that owns WAKING has already published READY. The
            // caller's required post-registration result check observes it.
            drop(replacement);
            return;
        }

        // Safety: IDLE -> REGISTERING gives this consumer exclusive access to
        // the waiter slot until it publishes IDLE again.
        let mut replaced = unsafe { (*self.waiter.get()).take() };
        let unused_replacement = if W::REPLACE_ON_REPEAT || replaced.is_none() {
            // Safety: this consumer still owns REGISTERING and the slot is empty.
            unsafe {
                *self.waiter.get() = Some(replacement);
            }
            None
        } else {
            // Safety: this consumer still owns REGISTERING and restores the
            // parked non-replaceable waiter before publishing.
            unsafe {
                *self.waiter.get() = replaced.take();
            }
            Some(replacement)
        };
        self.waiter_present.store(true, Ordering::Release);

        let wake = if self
            .waiter_state
            .compare_exchange(
                WAITER_REGISTERING,
                WAITER_IDLE,
                Ordering::Release,
                Ordering::Acquire,
            )
            .is_ok()
        {
            None
        } else {
            debug_assert_eq!(
                self.waiter_state.load(Ordering::Relaxed),
                WAITER_REGISTERING_AND_WAKING
            );
            // Safety: the producer only set WAKING and left the waiter slot to
            // this consumer, which still owns REGISTERING.
            let wake = unsafe { (*self.waiter.get()).take() };
            self.waiter_present.store(false, Ordering::Release);
            self.waiter_state.store(WAITER_IDLE, Ordering::Release);
            wake
        };

        // Waiter drop and wake may run arbitrary code, including completion.
        // The waiter state is IDLE before any of it runs.
        drop(replaced);
        drop(unused_replacement);
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    /// Claim the producer side of the hand-off.
    ///
    /// Returns whether this producer claimed the pending result.
    fn begin_completion(&self) -> bool {
        self.state
            .compare_exchange(
                RESULT_PENDING,
                RESULT_WRITING,
                Ordering::Relaxed,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Wake the registered consumer without waiting for registration to finish.
    fn wake_waiter(&self) {
        let previous = self.waiter_state.fetch_or(WAITER_WAKING, Ordering::AcqRel);

        if previous != WAITER_IDLE {
            debug_assert_eq!(previous, WAITER_REGISTERING);
            return;
        }

        // Safety: IDLE -> WAKING gives the producer exclusive access to the
        // waiter slot. A consumer cannot claim it until IDLE is restored.
        let waiter = unsafe { (*self.waiter.get()).take() };
        self.waiter_present.store(false, Ordering::Release);
        self.waiter_state.store(WAITER_IDLE, Ordering::Release);
        if let Some(waiter) = waiter {
            waiter.wake();
        }
    }
}

impl<T, W: Waiter, S: StateWord> Default for ResultCell<T, W, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, W: Waiter, S: StateWord> Drop for ResultCell<T, W, S> {
    fn drop(&mut self) {
        if self.state.exclusive_state() == RESULT_READY {
            // Safety: READY means the result cell is initialized and no
            // consumer took it, because drop has exclusive access.
            unsafe {
                self.result.get_mut().assume_init_drop();
            }
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
