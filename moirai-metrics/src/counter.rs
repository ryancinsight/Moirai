//! Atomic counter metric handle.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

/// A monotonically increasing counter metric.
#[derive(Clone, Debug)]
pub struct Counter {
    value: Arc<AtomicU64>,
}

impl Counter {
    /// Create an independent counter initialized to zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Increment the counter by one.
    pub fn increment(&self) {
        self.add(1);
    }

    /// Add `value` to the counter.
    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    /// Read the current counter value.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}
