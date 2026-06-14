//! Atomic gauge metric handle.

use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc,
};

/// A signed gauge metric that may increase or decrease.
#[derive(Clone, Debug)]
pub struct Gauge {
    value: Arc<AtomicI64>,
}

impl Gauge {
    /// Create an independent gauge initialized to zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicI64::new(0)),
        }
    }

    /// Set the gauge value.
    pub fn set(&self, value: i64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment the gauge by one.
    pub fn increment(&self) {
        self.add(1);
    }

    /// Decrement the gauge by one.
    pub fn decrement(&self) {
        self.add(-1);
    }

    /// Add `value` to the gauge.
    pub fn add(&self, value: i64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    /// Read the current gauge value.
    #[must_use]
    pub fn get(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
    }
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}
