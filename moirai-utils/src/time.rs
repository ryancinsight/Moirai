//! High-resolution timing utilities for performance measurement.
//!
//! This module provides utilities for accurate time measurement and
//! timestamp generation, particularly useful for benchmarking and
//! performance monitoring.

#[cfg(feature = "std")]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// High-resolution timer for performance measurements.
///
/// This timer provides nanosecond-precision timing for performance
/// measurements and benchmarking. It uses the most accurate timer
/// available on the current platform.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct HighResTimer {
    start: Instant,
}

#[cfg(feature = "std")]
impl HighResTimer {
    /// Create a new timer and start measuring.
    ///
    /// The timer begins counting immediately upon creation.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get the elapsed time since the timer was created.
    ///
    /// # Returns
    /// A `Duration` representing the elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get the elapsed time in nanoseconds.
    ///
    /// # Returns
    /// The elapsed time as a u64 in nanoseconds
    pub fn elapsed_nanos(&self) -> u64 {
        self.elapsed().as_nanos() as u64
    }

    /// Get the elapsed time in microseconds.
    ///
    /// # Returns
    /// The elapsed time as a u64 in microseconds
    pub fn elapsed_micros(&self) -> u64 {
        self.elapsed().as_micros() as u64
    }

    /// Get the elapsed time in milliseconds.
    ///
    /// # Returns
    /// The elapsed time as a u64 in milliseconds
    pub fn elapsed_millis(&self) -> u64 {
        self.elapsed().as_millis() as u64
    }

    /// Get the elapsed time in seconds.
    ///
    /// # Returns
    /// The elapsed time as a f64 in seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    /// Reset the timer.
    ///
    /// This resets the timer's start point to the current time.
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }

    /// Restart the timer and return the elapsed time.
    ///
    /// This is equivalent to calling `elapsed()` followed by `reset()`,
    /// but is more convenient for timing successive operations.
    ///
    /// # Returns
    /// The elapsed time since the last reset
    pub fn restart(&mut self) -> Duration {
        let elapsed = self.elapsed();
        self.reset();
        elapsed
    }
}

#[cfg(feature = "std")]
impl Default for HighResTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the current Unix timestamp in nanoseconds.
///
/// # Returns
/// The current Unix timestamp as nanoseconds since the Unix epoch,
/// or 0 if the system time is before the Unix epoch.
#[cfg(feature = "std")]
pub fn unix_timestamp_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Get the current Unix timestamp in microseconds.
///
/// # Returns
/// The current Unix timestamp as microseconds since the Unix epoch,
/// or 0 if the system time is before the Unix epoch.
#[cfg(feature = "std")]
pub fn unix_timestamp_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Get the current Unix timestamp in milliseconds.
///
/// # Returns
/// The current Unix timestamp as milliseconds since the Unix epoch,
/// or 0 if the system time is before the Unix epoch.
#[cfg(feature = "std")]
pub fn unix_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Get the current Unix timestamp in seconds.
///
/// # Returns
/// The current Unix timestamp as seconds since the Unix epoch,
/// or 0 if the system time is before the Unix epoch.
#[cfg(feature = "std")]
pub fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Measure the time taken to execute a closure.
///
/// This function executes the given closure and returns both its result
/// and the time taken to execute it.
///
/// # Arguments
/// * `f` - The closure to time
///
/// # Returns
/// A tuple containing the closure's result and the execution duration
#[cfg(feature = "std")]
pub fn time_execution<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Measure the average time taken to execute a closure multiple times.
///
/// This function executes the given closure the specified number of times
/// and returns the average execution time.
///
/// # Arguments
/// * `iterations` - The number of times to execute the closure
/// * `f` - The closure to time
///
/// # Returns
/// The average execution duration across all iterations
#[cfg(feature = "std")]
pub fn time_average<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    if iterations == 0 {
        return Duration::ZERO;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let total_duration = start.elapsed();

    total_duration / iterations as u32
}

#[cfg(all(feature = "std", test))]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_high_res_timer() {
        let mut timer = HighResTimer::new();

        // Sleep for a small amount
        thread::sleep(Duration::from_millis(10));

        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(100)); // Should be much less

        // Test restart
        let restart_elapsed = timer.restart();
        assert!(restart_elapsed >= Duration::from_millis(10));

        // Timer should be reset now
        thread::sleep(Duration::from_millis(1));
        let new_elapsed = timer.elapsed();
        assert!(new_elapsed < restart_elapsed);
    }

    #[test]
    fn test_timestamp_functions() {
        let nanos = unix_timestamp_nanos();
        let micros = unix_timestamp_micros();
        let millis = unix_timestamp_millis();
        let secs = unix_timestamp_secs();

        // Basic sanity checks
        assert!(nanos > 0);
        assert!(micros > 0);
        assert!(millis > 0);
        assert!(secs > 0);

        // Check relationships: each call uses a fresh SystemTime::now(), so later
        // calls may be slightly higher.  Allow up to 1 second of drift to keep
        // the test robust on slow or heavily-loaded machines.
        let nanos_as_secs = nanos / 1_000_000_000;
        let micros_as_secs = micros / 1_000_000;
        let millis_as_secs = millis / 1_000;
        assert!(nanos_as_secs.abs_diff(secs) <= 1, "nanos and secs diverged");
        assert!(
            micros_as_secs.abs_diff(secs) <= 1,
            "micros and secs diverged"
        );
        assert!(
            millis_as_secs.abs_diff(secs) <= 1,
            "millis and secs diverged"
        );
    }

    #[test]
    fn test_time_execution() {
        let (result, duration) = time_execution(|| {
            thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(5));
        assert!(duration < Duration::from_millis(50));
    }

    #[test]
    fn test_time_average() {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let avg_duration = time_average(3, || {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            thread::sleep(Duration::from_millis(1));
        });

        // Sleeping establishes a lower bound only; OS scheduling can add
        // unbounded delay on a loaded test host.
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert!(avg_duration >= Duration::from_millis(1));
    }

    #[test]
    fn test_time_average_zero_iterations() {
        let avg_duration = time_average(0, || {
            // This should never be called
            panic!("Should not be executed");
        });

        assert_eq!(avg_duration, Duration::ZERO);
    }
}
