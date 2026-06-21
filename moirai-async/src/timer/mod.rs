//! Async timer primitives for Moirai concurrency library.
//!
//! Following SLAP principle with focused responsibility on time-based async operations.

pub mod delay;
pub(super) mod registration;
pub(super) mod driver;
pub mod timeout;
pub mod interval;
pub mod limiter;
pub mod wheel;

pub use delay::Delay;
pub use timeout::{Timeout, TimeoutError};
pub use interval::Interval;
pub use limiter::{RateLimiter, RatePermit};
pub use wheel::{TimerCommand, TimerWheel};

use std::future::Future;
use std::time::{Duration, Instant};

/// Create a delay future that completes after the specified duration
pub fn sleep(duration: Duration) -> Delay {
    Delay::new(duration)
}

/// Timeout wrapper for futures with comprehensive cancellation
pub fn timeout<F>(duration: Duration, future: F) -> Timeout<F>
where
    F: Future,
{
    Timeout::new(future, duration)
}

/// Create a new interval timer
pub fn interval(period: Duration) -> Interval {
    Interval::new(period)
}

/// Create an interval timer that starts at a specific time
pub fn interval_at(start: Instant, period: Duration) -> Interval {
    Interval::new_at(start, period)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_delay_basic() {
        let delay = Delay::new(Duration::from_millis(10));
        assert!(delay.deadline() > Instant::now());
    }

    #[test]
    fn test_sleep_function() {
        let timer = sleep(Duration::from_millis(10));
        assert!(timer.deadline() > Instant::now());
    }
}
