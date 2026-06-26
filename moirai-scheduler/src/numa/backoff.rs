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
        // saturating_mul: the exponential factor (up to 1<<10) can overflow the
        // product under `overflow-checks` for a pathologically large base delay;
        // saturate to the configured cap instead of panicking.
        let new_delay = self
            .base_delay_ns
            .saturating_mul(1u64 << failures.min(10))
            .min(self.max_delay_ns);
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

#[cfg(test)]
mod tests {
    use super::AdaptiveBackoff;
    use std::time::Duration;

    #[test]
    fn delay_grows_geometrically_saturates_at_cap_and_resets() {
        // base = 100ns, cap = 1000ns. `record_failure` uses the failure count
        // BEFORE the increment, so after the k-th failure the delay is
        // base * 2^(k-1), clamped to the cap.
        let backoff = AdaptiveBackoff::new(100, 1000);
        assert_eq!(backoff.current_delay(), Duration::from_nanos(100)); // initial == base

        backoff.record_failure(); // 100 * 2^0
        assert_eq!(backoff.current_delay(), Duration::from_nanos(100));
        backoff.record_failure(); // 100 * 2^1
        assert_eq!(backoff.current_delay(), Duration::from_nanos(200));
        backoff.record_failure(); // 100 * 2^2
        assert_eq!(backoff.current_delay(), Duration::from_nanos(400));
        backoff.record_failure(); // 100 * 2^3
        assert_eq!(backoff.current_delay(), Duration::from_nanos(800));
        backoff.record_failure(); // 100 * 2^4 = 1600 -> clamped to 1000
        assert_eq!(backoff.current_delay(), Duration::from_nanos(1000));

        backoff.record_success(); // resets to base
        assert_eq!(backoff.current_delay(), Duration::from_nanos(100));
    }

    #[test]
    fn pathological_base_saturates_without_overflow() {
        // base near u64::MAX * the 2^failures factor would overflow; the
        // saturating multiply must clamp to the cap rather than panic under
        // overflow-checks.
        let backoff = AdaptiveBackoff::new(u64::MAX, u64::MAX);
        for _ in 0..20 {
            backoff.record_failure();
        }
        assert_eq!(backoff.current_delay(), Duration::from_nanos(u64::MAX));
    }
}
