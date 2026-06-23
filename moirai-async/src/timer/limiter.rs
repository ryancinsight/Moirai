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

        // Wait for next interval
        self.interval.next().await;
        self.current_permits = self.permits - 1;
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
