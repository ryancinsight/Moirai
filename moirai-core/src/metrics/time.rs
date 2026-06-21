//! Time-related metrics primitives (Instant, TimeDuration).

use std::time::Duration;

/// High-precision timestamp for performance measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant(u64);

impl Instant {
    /// Create a new instant representing the current time.
    #[must_use]
    pub fn now() -> Self {
        Self(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX), // Handle potential truncation gracefully
        )
    }

    /// Calculate the duration since an earlier instant.
    #[must_use]
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }

    /// Get the elapsed time since this instant.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        Self::now().duration_since(*self)
    }
}

/// A duration type optimized for performance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimeDuration(u64);

impl TimeDuration {
    /// Create a duration from nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Create a duration from microseconds.
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }

    /// Create a duration from milliseconds.
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }

    /// Create a duration from seconds.
    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }

    /// Get the duration in nanoseconds.
    #[must_use]
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Get the duration in microseconds.
    #[must_use]
    pub const fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }

    /// Get the duration in milliseconds.
    #[must_use]
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }

    /// Get the duration in seconds.
    #[must_use]
    pub const fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Get the duration as seconds with fractional precision.
    #[must_use]
    pub fn as_secs_f64(&self) -> f64 {
        // Use explicit conversion to handle precision loss intentionally
        #[allow(clippy::cast_precision_loss)]
        {
            self.0 as f64 / 1_000_000_000.0
        }
    }
}

impl std::fmt::Display for TimeDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 < 1_000 {
            write!(f, "{}ns", self.0)
        } else if self.0 < 1_000_000 {
            #[allow(clippy::cast_precision_loss)]
            {
                write!(f, "{:.1}μs", self.0 as f64 / 1_000.0)
            }
        } else if self.0 < 1_000_000_000 {
            #[allow(clippy::cast_precision_loss)]
            {
                write!(f, "{:.1}ms", self.0 as f64 / 1_000_000.0)
            }
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                write!(f, "{:.1}s", self.0 as f64 / 1_000_000_000.0)
            }
        }
    }
}
