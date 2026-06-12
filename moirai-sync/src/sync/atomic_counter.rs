use std::sync::atomic::{AtomicU64, Ordering};

/// An atomic counter with convenience methods.
pub struct AtomicCounter {
    inner: AtomicU64,
}

impl AtomicCounter {
    /// Create a new atomic counter.
    pub const fn new(value: u64) -> Self {
        Self {
            inner: AtomicU64::new(value),
        }
    }

    /// Increment the counter and return the new value.
    pub fn inc(&self) -> u64 {
        self.inner.fetch_add(1, Ordering::Relaxed).wrapping_add(1)
    }

    /// Decrement the counter and return the new value.
    pub fn dec(&self) -> u64 {
        self.inner.fetch_sub(1, Ordering::Relaxed).wrapping_sub(1)
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.inner.load(Ordering::Relaxed)
    }

    /// Set the value.
    pub fn set(&self, value: u64) {
        self.inner.store(value, Ordering::Relaxed);
    }
}
