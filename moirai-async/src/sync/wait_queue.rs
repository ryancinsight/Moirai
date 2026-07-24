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

use std::collections::{BTreeMap, VecDeque};
use std::task::Waker;

/// FIFO waiter queue generic over the grant payload `G`.
///
/// `G` carries whatever the granting side must hand to the waiter (e.g. a
/// notification kind); use `()` when the grant itself is the only signal.
///
/// # Implementation notes
///
/// Waiters are stored in a `BTreeMap` keyed by a monotonic id so that
/// per-waiter lookup, poll, and cancellation are O(log n).  A separate
/// `VecDeque<u64>` keeps the ids of *pending* waiters in FIFO order, which
/// lets `grant_oldest` run in amortized O(1) instead of scanning every waiter.
/// Ids of cancelled waiters are left in the pending queue and skipped lazily
/// when they reach the front; this avoids an O(n) removal cost on the
/// cancellation hot path.
pub(crate) struct WaitQueue<G> {
    waiters: BTreeMap<u64, WaitEntry<G>>,
    /// Ids of waiters that are still pending, in registration order.
    pending: VecDeque<u64>,
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
            pending: VecDeque::new(),
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
        self.pending.push_back(id);
        id
    }

    /// Grant `payload` to the oldest pending waiter and return its waker to
    /// wake. Returns `None` (dropping `payload`) if no waiter is pending, in
    /// which case the caller applies its no-waiter policy (store a permit,
    /// bump an availability counter, ...).
    pub(crate) fn grant_oldest(&mut self, payload: G) -> Option<Waker> {
        while let Some(&id) = self.pending.front() {
            match self.waiters.get_mut(&id) {
                Some(entry) if entry.granted.is_none() => {
                    entry.granted = Some(payload);
                    self.pending.pop_front();
                    return Some(entry.waker.clone());
                }
                _ => {
                    self.pending.pop_front();
                }
            }
        }
        None
    }

    /// Grant a clone of `payload` to every pending waiter, returning their
    /// wakers. The returned length is the number of waiters granted.
    pub(crate) fn grant_all(&mut self, payload: G) -> Vec<Waker>
    where
        G: Clone,
    {
        let mut wakers = Vec::new();
        let pending_ids: Vec<u64> = self.pending.drain(..).collect();
        for id in pending_ids {
            if let Some(entry) = self.waiters.get_mut(&id) {
                if entry.granted.is_none() {
                    entry.granted = Some(payload.clone());
                    wakers.push(entry.waker.clone());
                }
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

#[cfg(test)]
mod tests {
    use super::{WaitQueue, WaiterPoll};
    use std::task::Waker;

    fn noop_waker() -> Waker {
        Waker::noop().clone()
    }

    #[test]
    fn cancelled_head_does_not_block_fifo_grant() {
        let mut queue = WaitQueue::new();
        let cancelled = queue.register(noop_waker());
        let active = queue.register(noop_waker());

        assert_eq!(queue.deregister(cancelled), None);
        assert!(queue.grant_oldest(17_u8).is_some());
        assert!(matches!(
            queue.poll_waiter(active, &noop_waker()),
            WaiterPoll::Granted(17)
        ));
        assert!(queue.is_empty());
    }

    #[test]
    fn grant_all_preserves_every_pending_payload() {
        let mut queue = WaitQueue::new();
        let first = queue.register(noop_waker());
        let second = queue.register(noop_waker());

        assert_eq!(queue.grant_all(29_u8).len(), 2);
        assert!(matches!(
            queue.poll_waiter(first, &noop_waker()),
            WaiterPoll::Granted(29)
        ));
        assert!(matches!(
            queue.poll_waiter(second, &noop_waker()),
            WaiterPoll::Granted(29)
        ));
        assert!(queue.is_empty());
    }
}
