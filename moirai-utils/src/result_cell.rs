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
//! no lock: the producer runs once; the consumer is serialized with itself
//! (`poll` takes `Pin<&mut Self>` on the async side, and the blocking side
//! registers once per wait); and `Drop` runs only after the last `Arc`, so it
//! has exclusive access and races neither side.
//!
//! # State machine
//!
//! `state: AtomicU8` is the sole synchronization variable; the `result` and
//! `waiter` cells are touched only while a transition grants exclusive access to
//! them (C = consumer, P = producer):
//!
//! ```text
//!   PENDING ──C register──▶ WAITING ──C re-register──▶ UPDATING_WAITER ──C──▶ WAITING
//!      │                       │
//!      │ P complete            │ P complete
//!      ▼                       ▼
//!   WRITING ────────────────▶ WRITING ──P──▶ READY ──C take──▶ TAKEN
//! ```
//!
//! `WRITING` is the producer's exclusive claim on `result`, and `READY`
//! publishes it; `UPDATING_WAITER` is the consumer's exclusive claim on `waiter`
//! while it swaps a stale one, past which the producer spins. A waiter that is
//! never replaced (see [`Waiter::REPLACE_ON_REPEAT`]) never enters
//! `UPDATING_WAITER`, and monomorphization removes that arm entirely.
//!
//! # Cell-access invariants (what the per-site `// Safety:` comments rely on)
//!
//! 1. **`result`: written once, read once.** Only the producer writes it, only
//!    under `WRITING` (entered by winning the producer transition, so no
//!    consumer can see it yet). It is read exactly once — by the unique
//!    `READY -> TAKEN` consumer transition, or by `Drop` at `READY` when the
//!    consumer never took it.
//! 2. **`waiter`: written by the consumer, read once by the producer.** The
//!    consumer writes it under `PENDING` (published by the `PENDING -> WAITING`
//!    release transition) or under the `UPDATING_WAITER` claim. The producer
//!    reads it exactly once, on `WAITING -> WRITING`; otherwise `Drop` at
//!    `WAITING` drops it. A *failed* publish transition proves no producer
//!    observed the write, so the consumer drops its own value.
//! 3. **No lost wakeup.** [`complete`](ResultCell::complete) on the `PENDING ->
//!    WRITING` path (it beat registration) deliberately does not wake: no waiter
//!    is registered yet. Liveness therefore requires the consumer to check →
//!    [`register`](ResultCell::register) → **re-check**; should `complete` land in that
//!    window, the re-check observes `READY`. That re-check is load-bearing, not
//!    defensive — dropping it reintroduces a hang.
//!
//! # Ordering
//!
//! Each cell access is ordered by a release/acquire pair on `state`: the writer
//! releases on the publishing transition, the reader acquires on the transition
//! that reads the cell. Thus `PENDING -> WAITING` and the `UPDATING_WAITER ->
//! WAITING` store are `Release` (they publish `waiter`), while `WAITING ->
//! WRITING` and `READY -> TAKEN` are `Acquire` (they read a cell). The `PENDING
//! -> WRITING` success is `Relaxed`: that path reads neither cell before its own
//! `store(READY, Release)` publishes `result`, so it carries no incoming edge to
//! establish.
//!
//! # The waiter payload
//!
//! [`Waiter`] abstracts the parked handle: `thread::Thread` for the blocking
//! side, `Waker` for the async one. The two differ in exactly one behaviour, and
//! it is a trait constant rather than a branch —
//! [`REPLACE_ON_REPEAT`](Waiter::REPLACE_ON_REPEAT) is `false` for a thread,
//! which parks once per wait and must not be overwritten, and `true` for a
//! waker, which a re-poll may legitimately replace. Each instantiation therefore
//! compiles to its own machine, and both share this one copy of the protocol,
//! its invariants and its ordering argument.
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

const RESULT_PENDING: u8 = 0;
const RESULT_WAITING: u8 = 1;
const RESULT_UPDATING_WAITER: u8 = 2;
const RESULT_WRITING: u8 = 3;
const RESULT_READY: u8 = 4;
const RESULT_TAKEN: u8 = 5;

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

/// The one word the state machine synchronizes through.
///
/// The machine only ever loads, stores, and transitions a `u8`, which is all it
/// needs; naming that as a trait keeps *where the word lives* a decision of the
/// caller, not of the protocol. The async handle keeps it packed beside the
/// result, because it allocates one cell per spawned task and refuses to pad
/// each of them; the blocking handle puts it in an interference sector of its
/// own, so the producer's publish does not invalidate the result's line. Both
/// run the identical machine — see [the layout note](self#layout).
pub trait StateWord: Send + Sync {
    /// The word every cell starts at.
    fn pending() -> Self;

    /// Load the current state.
    fn load(&self, order: Ordering) -> u8;

    /// Store a new state.
    fn store(&self, value: u8, order: Ordering);

    /// Transition the state, returning the observed value on failure.
    fn compare_exchange(
        &self,
        current: u8,
        new: u8,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u8, u8>;

    /// Exclusive access for `Drop`, which needs no atomicity.
    fn get_mut(&mut self) -> &mut u8;
}

impl StateWord for AtomicU8 {
    #[inline]
    fn pending() -> Self {
        Self::new(RESULT_PENDING)
    }

    #[inline]
    fn load(&self, order: Ordering) -> u8 {
        Self::load(self, order)
    }

    #[inline]
    fn store(&self, value: u8, order: Ordering) {
        Self::store(self, value, order);
    }

    #[inline]
    fn compare_exchange(
        &self,
        current: u8,
        new: u8,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u8, u8> {
        Self::compare_exchange(self, current, new, success, failure)
    }

    #[inline]
    fn get_mut(&mut self) -> &mut u8 {
        Self::get_mut(self)
    }
}

impl StateWord for crate::cache::CacheAligned<AtomicU8> {
    #[inline]
    fn pending() -> Self {
        Self::new(AtomicU8::new(RESULT_PENDING))
    }

    #[inline]
    fn load(&self, order: Ordering) -> u8 {
        self.0.load(order)
    }

    #[inline]
    fn store(&self, value: u8, order: Ordering) {
        self.0.store(value, order);
    }

    #[inline]
    fn compare_exchange(
        &self,
        current: u8,
        new: u8,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u8, u8> {
        self.0.compare_exchange(current, new, success, failure)
    }

    #[inline]
    fn get_mut(&mut self) -> &mut u8 {
        self.0.get_mut()
    }
}

/// Single-producer, single-consumer completion cell for one unit of work.
///
/// The state word's storage is a parameter ([`StateWord`]), so each caller keeps
/// the layout it needs; see [the layout note](self#layout).
pub struct ResultCell<T, W: Waiter, S: StateWord = AtomicU8> {
    result: UnsafeCell<MaybeUninit<T>>,
    state: S,
    waiter: UnsafeCell<MaybeUninit<W>>,
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
            waiter: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Publish the result, waking the parked waiter if one is registered.
    ///
    /// Called exactly once by the producer; a second call is a no-op.
    pub fn complete(&self, result: T) {
        let Some(waiting) = self.begin_completion() else {
            return;
        };

        // Safety: WRITING is reachable only after `begin_completion` wins the
        // producer transition, so no consumer can read this cell yet.
        unsafe {
            (*self.result.get()).write(result);
        }

        self.state.store(RESULT_READY, Ordering::Release);

        if waiting {
            // Safety: WAITING is reachable only after `register` writes the
            // waiter and publishes it with a release transition.
            let waiter = unsafe { (*self.waiter.get()).assume_init_read() };
            waiter.wake();
        }
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
        self.state.load(Ordering::Acquire) == RESULT_WAITING
    }

    /// Park a clone of `waiter`, or replace the parked one when
    /// [`REPLACE_ON_REPEAT`](Waiter::REPLACE_ON_REPEAT) is set.
    ///
    /// Registration retries until the state is settled, so the waiter is stored
    /// from a clone and a failed publish transition costs only that clone. The
    /// caller must re-check [`try_take_ready`](ResultCell::try_take_ready) after this
    /// returns: a `complete` that raced in before the registration does not wake,
    /// because it saw no waiter. See the
    /// [module docs](self#cell-access-invariants).
    pub fn register(&self, waiter: &W) {
        loop {
            match self.state.load(Ordering::Acquire) {
                RESULT_PENDING => {
                    // Safety: there is one consumer. If the publish transition
                    // fails, this clone is dropped before retry.
                    unsafe {
                        (*self.waiter.get()).write(waiter.clone());
                    }

                    if self
                        .state
                        .compare_exchange(
                            RESULT_PENDING,
                            RESULT_WAITING,
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return;
                    }

                    // Safety: the transition failed, so no producer can observe
                    // this waiter cell as initialized through the WAITING state.
                    unsafe {
                        (*self.waiter.get()).assume_init_drop();
                    }
                }
                RESULT_WAITING => {
                    if !W::REPLACE_ON_REPEAT {
                        return;
                    }

                    if self
                        .state
                        .compare_exchange(
                            RESULT_WAITING,
                            RESULT_UPDATING_WAITER,
                            Ordering::Acquire,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        // Safety: UPDATING_WAITER excludes the producer from
                        // reading the waiter cell while the consumer replaces it.
                        unsafe {
                            (*self.waiter.get()).assume_init_drop();
                            (*self.waiter.get()).write(waiter.clone());
                        }
                        self.state.store(RESULT_WAITING, Ordering::Release);
                        return;
                    }
                }
                RESULT_UPDATING_WAITER | RESULT_WRITING => core::hint::spin_loop(),
                _ => return,
            }
        }
    }

    /// Claim the producer side of the hand-off.
    ///
    /// Returns `Some(waiting)` with the result cell claimed for writing, or
    /// `None` when the result was already published (or taken) and the caller
    /// must abandon its value.
    fn begin_completion(&self) -> Option<bool> {
        loop {
            match self.state.load(Ordering::Acquire) {
                RESULT_PENDING => {
                    if self
                        .state
                        .compare_exchange(
                            RESULT_PENDING,
                            RESULT_WRITING,
                            Ordering::Relaxed,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Some(false);
                    }
                }
                RESULT_WAITING => {
                    if self
                        .state
                        .compare_exchange(
                            RESULT_WAITING,
                            RESULT_WRITING,
                            Ordering::Acquire,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Some(true);
                    }
                }
                RESULT_UPDATING_WAITER => core::hint::spin_loop(),
                _ => return None,
            }
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
        match *self.state.get_mut() {
            RESULT_READY => {
                // Safety: READY means the result cell is initialized and no
                // consumer took it, because drop has exclusive access.
                unsafe {
                    self.result.get_mut().assume_init_drop();
                }
            }
            RESULT_WAITING => {
                // Safety: WAITING means the waiter cell is initialized and no
                // producer read it, because drop has exclusive access.
                unsafe {
                    self.waiter.get_mut().assume_init_drop();
                }
            }
            _ => {}
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
