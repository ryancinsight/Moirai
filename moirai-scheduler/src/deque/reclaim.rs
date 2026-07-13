//! Deque backing-array reclamation policies.
//!
//! Defines the sealed [`DequeReclaimPolicy`] / [`DequeReclaimState`] trait pair
//! and the two concrete policy types:
//! - [`DeferredReclaim`] — defer reclamation until the final endpoint drops (ZST)
//! - [`SharedEpochReclaim`] — shared epoch-counter reclamation (ZST)

use std::sync::atomic::{AtomicUsize, Ordering};

// ── Sealed trait ─────────────────────────────────────────────────────────────

mod reclaim_policy {
    pub trait Sealed {}
}

// ── Public traits ─────────────────────────────────────────────────────────────

/// Sealed policy interface for deque backing-array reclamation.
pub trait DequeReclaimPolicy: reclaim_policy::Sealed + Copy + Default {
    /// Concrete state carried by the deque for this reclamation policy.
    type State: DequeReclaimState;
}

/// State contract for monomorphized deque reclamation policies.
pub trait DequeReclaimState: Default + Send + Sync {
    /// Guard held while an operation may dereference the current backing array.
    type Guard<'a>
    where
        Self: 'a;

    /// Enter an array-access section.
    fn enter(&self) -> Self::Guard<'_>;

    /// Return true when retired arrays can be reclaimed from shared access.
    fn can_reclaim_shared(&self) -> bool;
}

// ── DeferredReclaim ───────────────────────────────────────────────────────────

/// Zero-sized state for final-drop reclamation.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeferredState;

/// Zero-sized access guard for deferred reclamation.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeferredAccessGuard;

impl DequeReclaimState for DeferredState {
    type Guard<'a> = DeferredAccessGuard;

    #[inline]
    fn enter(&self) -> Self::Guard<'_> {
        DeferredAccessGuard
    }

    #[inline]
    fn can_reclaim_shared(&self) -> bool {
        false
    }
}

/// Zero-sized policy retaining retired arrays until the final owner or stealer
/// endpoint drops. This adds no operation-path synchronization.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeferredReclaim;

impl reclaim_policy::Sealed for DeferredReclaim {}
impl DequeReclaimPolicy for DeferredReclaim {
    type State = DeferredState;
}

// ── SharedEpochReclaim ────────────────────────────────────────────────────────

/// Zero-sized policy enabling shared retired-array reclamation through an
/// active-access epoch counter.
#[derive(Clone, Copy, Debug, Default)]
pub struct SharedEpochReclaim;

impl reclaim_policy::Sealed for SharedEpochReclaim {}
impl DequeReclaimPolicy for SharedEpochReclaim {
    type State = SharedEpochState;
}

/// Shared reclamation state. This field exists only for deques instantiated
/// with `SharedEpochReclaim`.
#[derive(Debug, Default)]
pub struct SharedEpochState {
    active_accesses: AtomicUsize,
}

impl SharedEpochState {
    /// Get the current number of active accesses.
    #[inline]
    pub fn active_accesses(&self) -> usize {
        self.active_accesses.load(Ordering::Acquire)
    }
}

/// Guard for a shared array-access section.
#[derive(Debug)]
pub struct SharedEpochAccessGuard<'a> {
    active_accesses: &'a AtomicUsize,
}

impl DequeReclaimState for SharedEpochState {
    type Guard<'a> = SharedEpochAccessGuard<'a>;

    #[inline]
    fn enter(&self) -> Self::Guard<'_> {
        self.active_accesses.fetch_add(1, Ordering::AcqRel);
        SharedEpochAccessGuard {
            active_accesses: &self.active_accesses,
        }
    }

    #[inline]
    fn can_reclaim_shared(&self) -> bool {
        self.active_accesses.load(Ordering::Acquire) == 0
    }
}

impl Drop for SharedEpochAccessGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.active_accesses.fetch_sub(1, Ordering::AcqRel);
    }
}
