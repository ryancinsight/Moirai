//! Timer wheel data structure for explicit timer management.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::task::Waker;
use std::time::Instant;

/// Timer entry for the timer wheel.
#[derive(Debug)]
struct TimerEntry {
    id: u64,
    deadline: Instant,
    waker: Option<Waker>,
}

impl PartialEq for TimerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.deadline == other.deadline
    }
}

impl Eq for TimerEntry {}

impl PartialOrd for TimerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimerEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .deadline
            .cmp(&self.deadline)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// Timer wheel for efficient timer management.
pub struct TimerWheel {
    timers: BinaryHeap<TimerEntry>,
    /// Ids currently present in `timers`. Inserted on `schedule`, removed when an
    /// entry is popped (fired or drained). Used so `cancel` can tell whether a
    /// timer is still live without an O(n) heap scan.
    active: HashSet<u64>,
    /// Cancelled-but-not-yet-popped ids; entries are skipped when they reach the
    /// heap head. Invariant: `cancelled ⊆ active`, so every tombstone is reclaimed
    /// when its entry is popped — the set cannot grow without bound.
    cancelled: HashSet<u64>,
    next_id: u64,
}

/// Commands for timer management.
pub enum TimerCommand {
    /// Register a timer firing at `deadline`.
    Schedule {
        /// Identifier the wheel tracks the timer under.
        timer_id: u64,
        /// Instant the timer fires.
        deadline: Instant,
        /// Waker invoked at expiry.
        waker: Waker,
    },
    /// Remove a scheduled timer.
    Cancel {
        /// Identifier of the timer to remove.
        timer_id: u64,
    },
    /// Move a scheduled timer to a new deadline.
    Reschedule {
        /// Identifier of the timer to move.
        timer_id: u64,
        /// Replacement expiry instant.
        new_deadline: Instant,
    },
}

impl TimerWheel {
    /// Create a new timer wheel.
    pub fn new() -> Self {
        Self {
            timers: BinaryHeap::new(),
            active: HashSet::new(),
            cancelled: HashSet::new(),
            next_id: 1,
        }
    }

    /// Schedule a new timer.
    pub fn schedule(&mut self, deadline: Instant, waker: Waker) -> u64 {
        let timer_id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        self.active.insert(timer_id);
        self.timers.push(TimerEntry {
            id: timer_id,
            deadline,
            waker: Some(waker),
        });

        timer_id
    }

    /// Cancel a timer by ID. Returns `true` if a live timer was cancelled.
    ///
    /// Cancelling an id that was never scheduled or has already fired/drained is
    /// a no-op: such an id has no entry left in the heap, so recording a tombstone
    /// for it would never be reclaimed and would leak unboundedly.
    pub fn cancel(&mut self, timer_id: u64) -> bool {
        if !self.active.contains(&timer_id) {
            return false;
        }

        self.cancelled.insert(timer_id)
    }

    /// Pop the heap head, keeping the `active` membership index in sync.
    fn pop_head(&mut self) -> TimerEntry {
        let entry = self.timers.pop().expect("entry existed after peek");
        self.active.remove(&entry.id);
        entry
    }

    /// Drain cancelled entries sitting at the heap head, reclaiming their
    /// tombstones.
    fn drain_cancelled_head(&mut self) {
        while self
            .timers
            .peek()
            .is_some_and(|entry| self.cancelled.contains(&entry.id))
        {
            let entry = self.pop_head();
            self.cancelled.remove(&entry.id);
        }
    }

    /// Poll for expired timers and wake them.
    pub fn poll_expired(&mut self) -> usize {
        let now = Instant::now();
        let mut expired_count = 0;

        self.drain_cancelled_head();

        while self
            .timers
            .peek()
            .is_some_and(|entry| entry.deadline <= now)
        {
            let mut expired = self.pop_head();
            if self.cancelled.remove(&expired.id) {
                continue;
            }

            if let Some(waker) = expired.waker.take() {
                waker.wake();
                expired_count += 1;
            }
        }

        expired_count
    }

    /// Get the next expiration time.
    pub fn next_expiration(&mut self) -> Option<Instant> {
        self.drain_cancelled_head();
        self.timers.peek().map(|entry| entry.deadline)
    }

    /// Get the number of live (scheduled, not cancelled) timers.
    pub fn timer_count(&self) -> usize {
        // `cancelled ⊆ active`, so this never underflows.
        self.active.len() - self.cancelled.len()
    }

    /// Number of outstanding cancellation tombstones (test-only invariant probe).
    #[cfg(test)]
    fn tombstone_count(&self) -> usize {
        self.cancelled.len()
    }
}

impl Default for TimerWheel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::TimerWheel;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::task::{Wake, Waker};
    use std::time::{Duration, Instant};

    struct CountingWake {
        count: Arc<AtomicUsize>,
    }

    impl Wake for CountingWake {
        fn wake(self: Arc<Self>) {
            self.count.fetch_add(1, Ordering::Release);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.count.fetch_add(1, Ordering::Release);
        }
    }

    fn counting_waker(count: Arc<AtomicUsize>) -> Waker {
        Waker::from(Arc::new(CountingWake { count }))
    }

    #[test]
    fn timer_wheel_cancelled_timer_does_not_wake() {
        let mut wheel = TimerWheel::new();
        let wake_count = Arc::new(AtomicUsize::new(0));
        let timer_id = wheel.schedule(
            Instant::now() - Duration::from_millis(1),
            counting_waker(Arc::clone(&wake_count)),
        );

        assert!(wheel.cancel(timer_id));
        assert_eq!(wheel.poll_expired(), 0);
        assert_eq!(wake_count.load(Ordering::Acquire), 0);
        assert_eq!(wheel.timer_count(), 0);
    }

    #[test]
    fn timer_wheel_poll_wakes_only_uncancelled_expired_timers() {
        let mut wheel = TimerWheel::new();
        let wake_count = Arc::new(AtomicUsize::new(0));
        let deadline = Instant::now() - Duration::from_millis(1);
        let cancelled = wheel.schedule(deadline, counting_waker(Arc::clone(&wake_count)));
        let active = wheel.schedule(deadline, counting_waker(Arc::clone(&wake_count)));

        assert!(wheel.cancel(cancelled));
        assert_ne!(cancelled, active);
        assert_eq!(wheel.poll_expired(), 1);
        assert_eq!(wake_count.load(Ordering::Acquire), 1);
        assert_eq!(wheel.timer_count(), 0);
    }

    #[test]
    fn timer_wheel_cancel_after_expiry_does_not_leak_tombstones() {
        // Regression: cancelling an already-fired timer used to insert a tombstone
        // into `cancelled` that was never reclaimed (the entry was gone from the
        // heap), growing the set without bound across a long-running wheel.
        let mut wheel = TimerWheel::new();
        let wake_count = Arc::new(AtomicUsize::new(0));
        let id = wheel.schedule(
            Instant::now() - Duration::from_millis(1),
            counting_waker(Arc::clone(&wake_count)),
        );

        assert_eq!(wheel.poll_expired(), 1);
        assert_eq!(wheel.tombstone_count(), 0);

        // Cancelling the fired timer, or any never-scheduled id, is a no-op and
        // leaves no tombstone behind.
        assert!(!wheel.cancel(id));
        assert!(!wheel.cancel(9_999));
        assert_eq!(wheel.tombstone_count(), 0);
        assert_eq!(wheel.timer_count(), 0);
    }

    #[test]
    fn timer_wheel_cancelled_then_fired_reclaims_tombstone() {
        // A cancelled-but-still-queued timer leaves exactly one tombstone, which
        // is reclaimed when the entry is drained on the next poll.
        let mut wheel = TimerWheel::new();
        let wake_count = Arc::new(AtomicUsize::new(0));
        let id = wheel.schedule(
            Instant::now() - Duration::from_millis(1),
            counting_waker(Arc::clone(&wake_count)),
        );

        assert!(wheel.cancel(id));
        assert_eq!(wheel.tombstone_count(), 1);

        assert_eq!(wheel.poll_expired(), 0);
        assert_eq!(wake_count.load(Ordering::Acquire), 0);
        assert_eq!(wheel.tombstone_count(), 0, "tombstone must be reclaimed");
        assert_eq!(wheel.timer_count(), 0);
    }
}
