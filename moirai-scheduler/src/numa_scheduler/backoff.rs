//! Adaptive backoff strategy for work stealing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

/// Adaptive backoff strategy for work stealing.
#[derive(Debug)]
pub struct AdaptiveBackoff {
    pub(super) base_delay_ns: u64,
    pub(super) max_delay_ns: u64,
    pub(super) current_delay_ns: AtomicUsize,
    pub(super) consecutive_failures: AtomicUsize,
}

impl AdaptiveBackoff {
    /// Create a new adaptive backoff strategy.
    pub fn new(base_delay_ns: u64, max_delay_ns: u64) -> Self {
        Self {
            base_delay_ns,
            max_delay_ns,
            current_delay_ns: AtomicUsize::new(base_delay_ns as usize),
            consecutive_failures: AtomicUsize::new(0),
        }
    }

    /// Record a successful steal operation.
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.current_delay_ns
            .store(self.base_delay_ns as usize, Ordering::Relaxed);
    }

    /// Record a failed steal operation and increase backoff.
    pub fn record_failure(&self) {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        let new_delay = (self.base_delay_ns * (1 << failures.min(10))).min(self.max_delay_ns);
        self.current_delay_ns
            .store(new_delay as usize, Ordering::Relaxed);
    }

    /// Get the current backoff delay.
    pub fn current_delay(&self) -> Duration {
        Duration::from_nanos(self.current_delay_ns.load(Ordering::Relaxed) as u64)
    }

    /// Perform backoff delay.
    pub fn backoff(&self) {
        let delay = self.current_delay();
        if delay.as_nanos() < 1000 {
            // For very short delays, use spin loop
            for _ in 0..(delay.as_nanos() / 10) {
                std::hint::spin_loop();
            }
        } else {
            // For longer delays, yield or sleep
            if delay.as_millis() < 1 {
                thread::yield_now();
            } else {
                thread::sleep(delay);
            }
        }
    }
}

impl Default for AdaptiveBackoff {
    fn default() -> Self {
        Self::new(100, 1_000_000) // 100ns to 1ms
    }
}
