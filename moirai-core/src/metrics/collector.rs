//! Metric collector types (Counter, Gauge, Histogram).

use core::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe counter for performance metrics.
#[derive(Debug)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Create a new counter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Increment the counter by 1.
    pub fn increment(&self) {
        self.add(1);
    }

    /// Add a value to the counter.
    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe gauge for current values.
#[derive(Debug)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Create a new gauge.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Set the gauge value.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Increment the gauge by 1.
    pub fn increment(&self) {
        self.add(1);
    }

    /// Add to the gauge value.
    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    /// Subtract from the gauge value.
    pub fn subtract(&self, value: u64) {
        self.value.fetch_sub(value, Ordering::Relaxed);
    }
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe histogram for value distributions.
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; 16],
    sum: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram.
    #[must_use]
    pub const fn new() -> Self {
        // Use const fn to avoid interior mutable const warning
        const fn new_atomic() -> AtomicU64 {
            AtomicU64::new(0)
        }

        Self {
            buckets: [
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
                new_atomic(),
            ],
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a value in the histogram.
    pub fn record(&self, value: u64) {
        // Simple bucket assignment - in a real implementation this would be more sophisticated
        let bucket_index = if value == 0 {
            0
        } else {
            // Safe calculation to avoid overflow
            let leading_zeros = value.leading_zeros();
            if leading_zeros >= 15 {
                0
            } else {
                (15 - leading_zeros as usize).min(15)
            }
        };

        self.buckets[bucket_index].fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the total count of recorded values.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the sum of all recorded values.
    pub fn sum(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
    }

    /// Calculate the average of recorded values.
    pub fn average(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            0.0
        } else {
            // Intentional precision loss for averaging - use explicit allow
            #[allow(clippy::cast_precision_loss)]
            {
                self.sum() as f64 / count as f64
            }
        }
    }

    /// Get the count for a specific bucket.
    pub fn bucket_count(&self, bucket: usize) -> u64 {
        if bucket < self.buckets.len() {
            self.buckets[bucket].load(Ordering::Relaxed)
        } else {
            0
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}
