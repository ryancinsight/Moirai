//! The single atomic word the completion state machine synchronizes through.

use core::sync::atomic::{AtomicU8, Ordering};

use super::RESULT_PENDING;

/// The one word the state machine synchronizes through.
///
/// The machine only ever loads, stores, and transitions a `u8`, which is all it
/// needs; naming that as a trait keeps *where the word lives* a decision of the
/// caller, not of the protocol. The async handle keeps it packed beside the
/// result, because it allocates one cell per spawned task and refuses to pad
/// each of them; the blocking handle puts it in an interference sector of its
/// own, so the producer's publish does not invalidate the result's line. Both
/// run the identical machine — see [the layout note](super#layout).
///
/// # Safety
///
/// An implementation must behave as one atomic `u8` cell:
///
/// - `load`, `store`, and `compare_exchange` are atomic and honor the ordering
///   they are given (at least as strong as the requested one), because the
///   cell publishes and consumes its result and waiter through those edges;
/// - `load` returns only values a `store` or successful `compare_exchange`
///   wrote, or [`pending`](Self::pending)'s initial value, and `get_mut`
///   exposes that same cell.
///
/// A state word that reports `READY` before the result was written, or that
/// drops the release/acquire edge, makes the cell read uninitialized memory.
///
/// Implementing it takes `unsafe impl`, so safe code cannot install a word that
/// lies:
///
/// ```compile_fail,E0200
/// use core::sync::atomic::Ordering;
/// use moirai_utils::result_cell::StateWord;
///
/// struct Liar(u8);
///
/// impl StateWord for Liar {
///     fn pending() -> Self { Liar(0) }
///     fn load(&self, _: Ordering) -> u8 { 4 }
///     fn store(&self, _: u8, _: Ordering) {}
///     fn compare_exchange(&self, c: u8, _: u8, _: Ordering, _: Ordering) -> Result<u8, u8> { Ok(c) }
///     fn get_mut(&mut self) -> &mut u8 { &mut self.0 }
/// }
/// ```
pub unsafe trait StateWord: Send + Sync {
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

// SAFETY: `AtomicU8` is the atomic `u8` cell the contract describes, and each
// method forwards the caller's ordering unchanged.
unsafe impl StateWord for AtomicU8 {
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

// SAFETY: the wrapper only pads the `AtomicU8` it forwards every call to, with
// the caller's ordering unchanged.
unsafe impl StateWord for crate::cache::CacheAligned<AtomicU8> {
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
