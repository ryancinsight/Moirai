//! Shared waiter-queue state machine for the async sync primitives.
//!
//! `Notify`, `Semaphore`, and `RwLock` (read and write queues) all require the
//! same mechanics: FIFO-ordered waiter registration keyed by a monotonic id,
//! grant hand-off to the oldest pending waiter (or a batch), a three-phase
//! poll (granted-check → primitive-specific admission → register), and
//! grant restoration when a granted-but-unconsumed future is cancelled.
//! [`WaitQueue`] owns those mechanics once; each primitive keeps only its
//! admission predicate (permit counter, reader/writer state, notify flag) and
//! its grant-restoration policy in its own `poll`/`Drop` bodies.
//!
//! The queue is keyed by a monotonic `u64` id (a `BTreeMap` rather than a
//! linear `VecDeque`) so per-waiter `poll`/drop lookups and removals are
//! O(log n) instead of O(n) — the primitive's state lock is held for less
//! time per operation under contention. Monotonic ids keep in-order iteration
//! FIFO, so grants go to the oldest waiter first.
//!
//! Grant semantics (verified by the pre-consolidation sync-primitive tests,
//! which pass unmodified against this implementation):
//! - a grant marks the entry with its payload and leaves it in the queue; the
//!   entry is removed when the waiter consumes it ([`Self::poll_waiter`]
//!   returning [`WaiterPoll::Granted`]) or is cancelled
//!   ([`Self::deregister`]);
//! - [`Self::deregister`] returns the unconsumed grant payload so the caller
//!   can restore it (re-grant a permit, release a lock) instead of losing it;
//! - [`Self::grant_all`] marks every pending waiter, returning their wakers,
//!   for batch admission (reader batches, `notify_waiters`).

use std::collections::BTreeMap;
use std::task::Waker;

/// FIFO waiter queue generic over the grant payload `G`.
///
/// `G` carries whatever the granting side must hand to the waiter (e.g. a
/// notification kind); use `()` when the grant itself is the only signal.
pub(crate) struct WaitQueue<G> {
    waiters: BTreeMap<u64, WaitEntry<G>>,
    next_id: u64,
}

struct WaitEntry<G> {
    waker: Waker,
    /// `Some(payload)` once the entry has been granted; `None` while pending.
    granted: Option<G>,
}

/// Outcome of polling a registered waiter.
pub(crate) enum WaiterPoll<G> {
    /// The waiter had been granted; its entry has been removed and the grant
    /// payload is returned for consumption.
    Granted(G),
    /// Still pending; the stored waker has been refreshed.
    Pending,
    /// The registration is gone (id absent); the caller retries its admission
    /// predicate and re-registers.
    NotRegistered,
}

impl<G> WaitQueue<G> {
    pub(crate) const fn new() -> Self {
        Self {
            waiters: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// `true` when no waiter (pending or granted) is registered.
    pub(crate) fn is_empty(&self) -> bool {
        self.waiters.is_empty()
    }

    /// Register a fresh pending waiter, returning its id.
    pub(crate) fn register(&mut self, waker: Waker) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.waiters.insert(
            id,
            WaitEntry {
                waker,
                granted: None,
            },
        );
        id
    }

    /// Grant `payload` to the oldest pending waiter and return its waker to
    /// wake. Returns `None` (dropping `payload`) if no waiter is pending, in
    /// which case the caller applies its no-waiter policy (store a permit,
    /// bump an availability counter, ...).
    pub(crate) fn grant_oldest(&mut self, payload: G) -> Option<Waker> {
        let waiter = self.waiters.values_mut().find(|w| w.granted.is_none())?;
        waiter.granted = Some(payload);
        Some(waiter.waker.clone())
    }

    /// Grant a clone of `payload` to every pending waiter, returning their
    /// wakers. The returned length is the number of waiters granted.
    pub(crate) fn grant_all(&mut self, payload: G) -> Vec<Waker>
    where
        G: Clone,
    {
        let mut wakers = Vec::new();
        for waiter in self.waiters.values_mut() {
            if waiter.granted.is_none() {
                waiter.granted = Some(payload.clone());
                wakers.push(waiter.waker.clone());
            }
        }
        wakers
    }

    /// Poll a registered waiter: consume its grant if present (removing the
    /// entry), otherwise refresh its waker.
    pub(crate) fn poll_waiter(&mut self, id: u64, waker: &Waker) -> WaiterPoll<G> {
        let Some(entry) = self.waiters.get_mut(&id) else {
            return WaiterPoll::NotRegistered;
        };
        if entry.granted.is_none() {
            entry.waker = waker.clone();
            return WaiterPoll::Pending;
        }
        let entry = self
            .waiters
            .remove(&id)
            .expect("invariant: entry present, checked above");
        WaiterPoll::Granted(
            entry
                .granted
                .expect("invariant: grant present, checked above"),
        )
    }

    /// Remove a waiter (cancellation or post-acquire cleanup). Returns the
    /// unconsumed grant payload if the entry had been granted, so the caller
    /// can restore it; `None` if the entry was pending or absent.
    pub(crate) fn deregister(&mut self, id: u64) -> Option<G> {
        self.waiters.remove(&id).and_then(|entry| entry.granted)
    }
}
