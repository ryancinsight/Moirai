//! Async timer primitives for Moirai concurrency library.
//!
//! Following SLAP principle with focused responsibility on time-based async operations.

/// One-shot delay futures.
pub mod delay;
pub(super) mod driver;
/// Repeating interval ticks.
pub mod interval;
/// Time-based rate limiting.
pub mod limiter;
pub(super) mod registration;
/// Deadline wrappers over futures.
pub mod timeout;
/// Hierarchical timing-wheel implementation.
pub mod wheel;

pub use delay::Delay;
pub use interval::Interval;
pub use limiter::{RateLimiter, RatePermit};
pub use timeout::{Timeout, TimeoutError};
pub use wheel::{TimerCommand, TimerWheel};

use std::future::Future;
use std::time::{Duration, Instant};

/// Compute `base + duration` without panicking on absurd durations.
///
/// `Instant + Duration` panics on overflow, so a near-`Duration::MAX` input
/// (e.g. a caller using `Duration::MAX` as "never") would abort. Clamp the
/// duration to ~100 years — effectively "never" — which `checked_add` then
/// resolves without overflowing `Instant`. Mirrors the round-16 hardening of
/// `moirai_pal::timer::Timer::new`. `unwrap_or(base)` is a safe
/// (non-panicking) degenerate fallback; it is unreachable on any real
/// platform, where `Instant` has decades of headroom.
pub(crate) fn clamped_deadline(base: Instant, duration: Duration) -> Instant {
    const MAX_TIMER: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);
    base.checked_add(duration.min(MAX_TIMER)).unwrap_or(base)
}

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

    /// A deadline one year out — far below the ~100-year clamp, so a clamped
    /// extreme duration must land beyond it.
    fn one_year_from_now() -> Instant {
        Instant::now() + Duration::from_secs(365 * 24 * 60 * 60)
    }

    #[test]
    fn delay_extreme_duration_does_not_panic() {
        // Regression: `Instant::now() + Duration::MAX` panics on overflow. The
        // deadline computation must clamp/`checked_add` instead, yielding a
        // far-future deadline rather than aborting.
        let delay = Delay::new(Duration::MAX);
        assert!(delay.deadline() > one_year_from_now());
    }

    #[test]
    fn delay_reset_extreme_duration_does_not_panic() {
        let mut delay = Delay::new(Duration::from_millis(1));
        delay.reset(Duration::MAX);
        assert!(delay.deadline() > one_year_from_now());
    }

    #[test]
    fn interval_extreme_period_does_not_panic() {
        let timer = interval(Duration::MAX);
        assert!(timer.next_tick() > one_year_from_now());

        let mut timer = interval(Duration::from_millis(1));
        timer.set_period(Duration::MAX);
        assert!(timer.next_tick() > one_year_from_now());
    }
}
