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
    cancelled: HashSet<u64>,
    next_id: u64,
}

/// Commands for timer management.
pub enum TimerCommand {
    Schedule {
        timer_id: u64,
        deadline: Instant,
        waker: Waker,
    },
    Cancel {
        timer_id: u64,
    },
    Reschedule {
        timer_id: u64,
        new_deadline: Instant,
    },
}

impl TimerWheel {
    /// Create a new timer wheel.
    pub fn new() -> Self {
        Self {
            timers: BinaryHeap::new(),
            cancelled: HashSet::new(),
            next_id: 1,
        }
    }

    /// Schedule a new timer.
    pub fn schedule(&mut self, deadline: Instant, waker: Waker) -> u64 {
        let timer_id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        self.timers.push(TimerEntry {
            id: timer_id,
            deadline,
            waker: Some(waker),
        });

        timer_id
    }

    /// Cancel a timer by ID.
    pub fn cancel(&mut self, timer_id: u64) -> bool {
        if timer_id == 0 || timer_id >= self.next_id {
            return false;
        }

        self.cancelled.insert(timer_id)
    }

    /// Poll for expired timers and wake them.
    pub fn poll_expired(&mut self) -> usize {
        let now = Instant::now();
        let mut expired_count = 0;

        while self
            .timers
            .peek()
            .is_some_and(|entry| self.cancelled.contains(&entry.id))
        {
            let entry = self
                .timers
                .pop()
                .expect("cancelled timer existed after peek");
            self.cancelled.remove(&entry.id);
        }

        while self
            .timers
            .peek()
            .is_some_and(|entry| entry.deadline <= now)
        {
            let mut expired = self.timers.pop().expect("expired timer existed after peek");
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
        while self
            .timers
            .peek()
            .is_some_and(|entry| self.cancelled.contains(&entry.id))
        {
            let entry = self
                .timers
                .pop()
                .expect("cancelled timer existed after peek");
            self.cancelled.remove(&entry.id);
        }

        self.timers.peek().map(|entry| entry.deadline)
    }

    /// Get the number of active timers.
    pub fn timer_count(&self) -> usize {
        self.timers
            .iter()
            .filter(|entry| !self.cancelled.contains(&entry.id))
            .count()
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
}
