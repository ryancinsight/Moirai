//! Deque backing-array reclamation policies.
//!
//! Defines the sealed [`DequeReclaimPolicy`] / [`DequeReclaimState`] trait pair
//! and the two concrete policy types:
//! - [`QuiescentReclaim`] — exclusive quiescent-point reclamation (ZST)
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

// ── QuiescentReclaim ──────────────────────────────────────────────────────────

/// Zero-sized state for exclusive quiescent reclamation.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuiescentState;

/// Zero-sized access guard for exclusive quiescent reclamation.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuiescentAccessGuard;

impl DequeReclaimState for QuiescentState {
    type Guard<'a> = QuiescentAccessGuard;

    #[inline]
    fn enter(&self) -> Self::Guard<'_> {
        QuiescentAccessGuard
    }

    #[inline]
    fn can_reclaim_shared(&self) -> bool {
        false
    }
}

/// Zero-sized policy proving retired deque arrays are reclaimed only from an
/// exclusive quiescent access path.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuiescentReclaim;

impl reclaim_policy::Sealed for QuiescentReclaim {}
impl DequeReclaimPolicy for QuiescentReclaim {
    type State = QuiescentState;
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
