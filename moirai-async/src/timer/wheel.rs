//! Timer wheel data structure for explicit timer management.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, binary_heap::PeekMut};
use std::task::Waker;
use std::time::Instant;

/// Timer entry for the timer wheel.
#[derive(Debug)]
struct TimerEntry {
    id: u64,
    deadline: Instant,
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
    /// Owns each live waker. A heap entry without a waker is stale.
    active: HashMap<u64, Waker>,
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
            active: HashMap::new(),
            next_id: 1,
        }
    }

    /// Schedule a new timer.
    pub fn schedule(&mut self, deadline: Instant, waker: Waker) -> u64 {
        let timer_id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        self.active.insert(timer_id, waker);
        self.timers.push(TimerEntry {
            id: timer_id,
            deadline,
        });

        timer_id
    }

    /// Cancel a timer by ID. Returns `true` if a live timer was cancelled.
    ///
    /// Releases the waker before returning. A stale deadline may remain in the
    /// heap until it reaches the head or stale entries outnumber live timers.
    pub fn cancel(&mut self, timer_id: u64) -> bool {
        let Some(waker) = self.active.remove(&timer_id) else {
            return false;
        };

        self.compact_deadlines();
        drop(waker);
        true
    }

    fn compact_deadlines(&mut self) {
        // Charge the linear scan to the stale entries it removes. They account
        // for more than half of the heap whenever a rebuild is needed.
        if self.active.len() < self.timers.len().div_ceil(2) {
            self.timers
                .retain(|entry| self.active.contains_key(&entry.id));
        }
    }

    /// Discard stale deadlines before reporting the next live deadline.
    fn drain_cancelled_head(&mut self) {
        while self
            .timers
            .peek()
            .is_some_and(|entry| !self.active.contains_key(&entry.id))
        {
            self.timers.pop();
        }
    }

    /// Poll for expired timers and wake them.
    pub fn poll_expired(&mut self) -> usize {
        let now = Instant::now();
        let mut expired_count = 0;

        while let Some(expired) = self
            .timers
            .peek_mut()
            .filter(|entry| entry.deadline <= now)
            .map(PeekMut::pop)
        {
            if let Some(waker) = self.active.remove(&expired.id) {
                // Reduce stale deadlines before calling a waker that can unwind.
                self.compact_deadlines();
                waker.wake();
                expired_count += 1;
            }
        }
        self.drain_cancelled_head();

        expired_count
    }

    /// Get the next expiration time.
    pub fn next_expiration(&mut self) -> Option<Instant> {
        self.drain_cancelled_head();
        self.timers.peek().map(|entry| entry.deadline)
    }

    /// Get the number of live (scheduled, not cancelled) timers.
    pub fn timer_count(&self) -> usize {
        self.active.len()
    }

    /// Number of outstanding cancellation tombstones (test-only invariant probe).
    #[cfg(test)]
    fn tombstone_count(&self) -> usize {
        self.timers.len() - self.active.len()
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
        Arc,
        atomic::{AtomicUsize, Ordering},
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
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("invariant: process uptime exceeds 1ms"),
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
        let deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("invariant: process uptime exceeds 1ms");
        let cancelled = wheel.schedule(deadline, counting_waker(Arc::clone(&wake_count)));
        let active = wheel.schedule(deadline, counting_waker(Arc::clone(&wake_count)));

        assert!(wheel.cancel(cancelled));
        assert_ne!(cancelled, active);
        assert_eq!(wheel.poll_expired(), 1);
        assert_eq!(wake_count.load(Ordering::Acquire), 1);
        assert_eq!(wheel.timer_count(), 0);
    }

    #[test]
    fn timer_wheel_cancel_releases_wakers_behind_a_live_deadline() {
        let mut wheel = TimerWheel::new();
        let deadline = Instant::now() + Duration::from_secs(60);
        let live = wheel.schedule(deadline, Waker::noop().clone());
        let later = deadline + Duration::from_secs(60);
        let wake = Arc::new(CountingWake {
            count: Arc::new(AtomicUsize::new(0)),
        });

        for _ in 0..128 {
            let id = wheel.schedule(later, Waker::from(Arc::clone(&wake)));
            assert!(wheel.cancel(id));
            assert!(!wheel.cancel(id));
            assert_eq!(Arc::strong_count(&wake), 1);
            assert!(wheel.timers.len() <= 2);
        }

        assert_eq!(wheel.next_expiration(), Some(deadline));
        assert_eq!(wheel.timer_count(), 1);
        assert!(wheel.cancel(live));
        assert_eq!(wheel.next_expiration(), None);
        assert_eq!(wheel.timer_count(), 0);
        assert_eq!(wheel.tombstone_count(), 0);
        assert_eq!(wake.count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn timer_wheel_expiry_compacts_stale_deadlines_behind_a_live_deadline() {
        let mut wheel = TimerWheel::new();
        let now = Instant::now();
        let live_deadline = now + Duration::from_secs(60);
        let stale_deadline = live_deadline + Duration::from_secs(60);
        let wakes = Arc::new(AtomicUsize::new(0));
        wheel.schedule(live_deadline, Waker::noop().clone());
        for _ in 0..8 {
            wheel.schedule(now, counting_waker(Arc::clone(&wakes)));
            let stale = wheel.schedule(stale_deadline, Waker::noop().clone());
            assert!(wheel.cancel(stale));
        }

        assert_eq!(wheel.tombstone_count(), 8);
        assert_eq!(wheel.poll_expired(), 8);
        assert_eq!(wakes.load(Ordering::Acquire), 8);
        assert_eq!(wheel.tombstone_count(), 0);
        assert_eq!(wheel.next_expiration(), Some(live_deadline));
        assert_eq!(wheel.timer_count(), 1);
    }

    #[test]
    fn timer_wheel_retains_its_bound_when_a_waker_panics() {
        struct PanickingWake;

        impl Wake for PanickingWake {
            fn wake(self: Arc<Self>) {
                panic!("timer wake panicked");
            }
        }

        let mut wheel = TimerWheel::new();
        let now = Instant::now();
        let future = now + Duration::from_secs(60);
        wheel.schedule(now, Waker::from(Arc::new(PanickingWake)));
        wheel.schedule(future, Waker::noop().clone());
        for _ in 0..2 {
            let stale = wheel.schedule(future + Duration::from_secs(60), Waker::noop().clone());
            assert!(wheel.cancel(stale));
        }

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| wheel.poll_expired()))
            .expect_err("the due waker should panic");
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"timer wake panicked"));
        assert_eq!(wheel.next_expiration(), Some(future));
        assert_eq!(wheel.timer_count(), 1);
        assert_eq!(wheel.tombstone_count(), 0);
    }

    #[test]
    fn timer_wheel_cancel_after_expiry_does_not_leak_tombstones() {
        // Regression: cancelling an already-fired timer used to insert a tombstone
        // into `cancelled` that was never reclaimed (the entry was gone from the
        // heap), growing the set without bound across a long-running wheel.
        let mut wheel = TimerWheel::new();
        let wake_count = Arc::new(AtomicUsize::new(0));
        let id = wheel.schedule(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("invariant: process uptime exceeds 1ms"),
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
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("invariant: process uptime exceeds 1ms"),
            counting_waker(Arc::clone(&wake_count)),
        );

        let later = Instant::now() + Duration::from_secs(60);
        let live = wheel.schedule(later, Waker::noop().clone());
        assert!(wheel.cancel(id));
        assert_eq!(wheel.tombstone_count(), 1);

        assert_eq!(wheel.poll_expired(), 0);
        assert_eq!(wake_count.load(Ordering::Acquire), 0);
        assert_eq!(wheel.tombstone_count(), 0, "tombstone must be reclaimed");
        assert_eq!(wheel.next_expiration(), Some(later));
        assert!(wheel.cancel(live));
        assert_eq!(wheel.timer_count(), 0);
    }
}
