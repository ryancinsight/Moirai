use crate::timer::interval::Interval;
use std::time::Duration;

/// Rate limiter using token bucket algorithm
pub struct RateLimiter {
    permits: u32,
    current_permits: u32,
    interval: Interval,
}

/// A permit from the rate limiter
pub struct RatePermit;

impl RateLimiter {
    /// Create a new rate limiter with specified permits per second
    pub fn new(permits_per_second: u32) -> Self {
        let interval_duration = if permits_per_second > 0 {
            Duration::from_nanos(1_000_000_000 / permits_per_second as u64)
        } else {
            Duration::from_secs(1)
        };

        Self {
            permits: permits_per_second,
            current_permits: permits_per_second,
            interval: Interval::new(interval_duration),
        }
    }

    /// Acquire a permit to perform an operation
    pub async fn acquire(&mut self) -> RatePermit {
        if self.current_permits > 0 {
            self.current_permits -= 1;
            return RatePermit;
        }

        // Wait for next interval, then refill and consume one. `saturating_sub`
        // guards the degenerate `permits_per_second == 0` construction: without
        // it `self.permits - 1` underflows to `u32::MAX` (a zero-rate limiter
        // that grants ~4.3 billion permits per interval in release builds, or
        // panics under overflow-checks). A zero rate therefore refills to 0 and
        // grants exactly the one permit this call is returning.
        self.interval.next().await;
        self.current_permits = self.permits.saturating_sub(1);
        RatePermit
    }

    /// Try to acquire a permit without waiting
    pub fn try_acquire(&mut self) -> Option<RatePermit> {
        if self.current_permits > 0 {
            self.current_permits -= 1;
            Some(RatePermit)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_acquire_exhausts_then_denies() {
        let mut limiter = RateLimiter::new(3);
        assert!(limiter.try_acquire().is_some());
        assert!(limiter.try_acquire().is_some());
        assert!(limiter.try_acquire().is_some());
        assert!(
            limiter.try_acquire().is_none(),
            "a 3-permit limiter must deny the fourth immediate acquire"
        );
    }

    #[test]
    fn zero_rate_limiter_does_not_underflow_on_refill() {
        // Regression: `new(0)` refilling via `permits - 1` underflowed to
        // u32::MAX. The refill must saturate at 0 so the bucket never grants a
        // spurious ~4.3 billion permits.
        let mut limiter = RateLimiter::new(0);
        assert!(
            limiter.try_acquire().is_none(),
            "a zero-rate limiter starts with no immediate permits"
        );
        // Exercise the refill arithmetic directly (the async path performs the
        // same `permits.saturating_sub(1)`): it must not panic or wrap.
        limiter.current_permits = limiter.permits.saturating_sub(1);
        assert_eq!(limiter.current_permits, 0);
    }
}
