//! Shared subscriber registry for the fan-out async channels.
//!
//! `Broadcast` and `Watch` both keep one slot per receiver, keyed by a monotonic
//! id, so that a per-poll and per-drop lookup of a receiver's slot is O(log n)
//! instead of O(n) — the lock is held for less time per operation when many
//! receivers subscribe to one channel. Both also hand a fresh subscriber its own
//! cursor at the publisher's current position, drop the slot with the receiver,
//! and fan a wake out to every registered waker on publication.
//!
//! Those mechanics live here once, the same consolidation
//! [`WaitQueue`](super::wait_queue::WaitQueue) performed for the waiter-ordered
//! primitives: each channel keeps only its cursor meaning (a watch version, a
//! broadcast position) and its own publish policy.

use std::collections::BTreeMap;
use std::task::Waker;

/// One subscriber's slot: its cursor and the waker registered while it waits.
pub(crate) struct Subscriber<C> {
    /// The channel's own per-subscriber payload — the last value this
    /// subscriber observed, in whatever unit the channel counts in.
    pub(crate) cursor: C,
    /// Set by the subscriber's `poll` while it waits, and cleared when it is
    /// polled to completion, cancelled, or woken.
    pub(crate) waker: Option<Waker>,
}

/// Monotonic-id registry of subscriber slots.
pub(crate) struct SubscriberRegistry<C> {
    entries: BTreeMap<u64, Subscriber<C>>,
    next_id: u64,
}

impl<C> SubscriberRegistry<C> {
    /// A registry holding the initial subscriber, id `0`, at `cursor`.
    pub(crate) fn with_initial(cursor: C) -> Self {
        let mut entries = BTreeMap::new();
        entries.insert(
            0,
            Subscriber {
                cursor,
                waker: None,
            },
        );
        Self {
            entries,
            next_id: 1,
        }
    }

    /// Register a new subscriber at `cursor`, returning its id.
    pub(crate) fn register(&mut self, cursor: C) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(
            id,
            Subscriber {
                cursor,
                waker: None,
            },
        );
        id
    }

    /// Remove `id`'s slot, if it is still registered.
    pub(crate) fn remove(&mut self, id: u64) {
        self.entries.remove(&id);
    }

    /// Mutable access to `id`'s slot.
    pub(crate) fn get_mut(&mut self, id: u64) -> Option<&mut Subscriber<C>> {
        self.entries.get_mut(&id)
    }

    /// Number of registered subscribers.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Every subscriber's cursor, for the channel's own fan-out decisions.
    pub(crate) fn cursors(&self) -> impl Iterator<Item = &C> {
        self.entries.values().map(|subscriber| &subscriber.cursor)
    }

    /// Take every registered waker, leaving each slot's waker clear.
    ///
    /// The wakers are returned rather than woken here because the caller holds
    /// the channel state lock and must release it first: `Waker::wake` may poll
    /// the task inline on this thread, and that poll re-locks the same state —
    /// waking under the lock would self-deadlock. `hybrid::notify` documents the
    /// same discipline for its registries.
    pub(crate) fn drain_wakers(&mut self) -> Vec<Waker> {
        self.entries
            .values_mut()
            .filter_map(|subscriber| subscriber.waker.take())
            .collect()
    }
}

/// Wake wakers previously taken by [`SubscriberRegistry::drain_wakers`].
///
/// A caller must release the channel state lock before calling this — see the
/// re-entrancy note on `drain_wakers`.
pub(crate) fn wake_drained(wakers: Vec<Waker>) {
    for waker in wakers {
        waker.wake();
    }
}
