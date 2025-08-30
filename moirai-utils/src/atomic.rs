//! Atomic operations and counters for lock-free programming.
//!
//! This module provides atomic utilities for building lock-free data structures
//! and implementing efficient counters and statistics tracking.

use core::sync::atomic::{AtomicUsize, Ordering};

/// A thread-safe atomic counter with increment and decrement operations.
///
/// This counter uses relaxed ordering for maximum performance while
/// maintaining atomicity across threads.
#[derive(Debug)]
pub struct AtomicCounter {
    value: AtomicUsize,
}

impl AtomicCounter {
    /// Create a new atomic counter with initial value 0.
    pub const fn new() -> Self {
        Self {
            value: AtomicUsize::new(0),
        }
    }

    /// Create a new atomic counter with the given initial value.
    pub const fn with_initial(initial: usize) -> Self {
        Self {
            value: AtomicUsize::new(initial),
        }
    }

    /// Get the current value of the counter.
    pub fn get(&self) -> usize {
        self.value.load(Ordering::Relaxed)
    }

    /// Set the counter to a specific value.
    pub fn set(&self, value: usize) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment the counter by 1 and return the previous value.
    pub fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::Relaxed)
    }

    /// Increment the counter by a specific amount and return the previous value.
    pub fn add(&self, amount: usize) -> usize {
        self.value.fetch_add(amount, Ordering::Relaxed)
    }

    /// Decrement the counter by 1 and return the previous value.
    /// Note: This will wrap around on underflow.
    pub fn decrement(&self) -> usize {
        self.value.fetch_sub(1, Ordering::Relaxed)
    }

    /// Decrement the counter by a specific amount and return the previous value.
    /// Note: This will wrap around on underflow.
    pub fn subtract(&self, amount: usize) -> usize {
        self.value.fetch_sub(amount, Ordering::Relaxed)
    }

    /// Atomically set the counter to the maximum of its current value and the given value.
    pub fn max(&self, value: usize) -> usize {
        self.value.fetch_max(value, Ordering::Relaxed)
    }

    /// Atomically set the counter to the minimum of its current value and the given value.
    pub fn min(&self, value: usize) -> usize {
        self.value.fetch_min(value, Ordering::Relaxed)
    }

    /// Reset the counter to 0 and return the previous value.
    pub fn reset(&self) -> usize {
        self.value.swap(0, Ordering::Relaxed)
    }

    /// Compare and swap the counter value.
    /// Returns the previous value and whether the swap was successful.
    pub fn compare_and_swap(&self, current: usize, new: usize) -> (usize, bool) {
        match self
            .value
            .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(prev) => (prev, true),
            Err(prev) => (prev, false),
        }
    }
}

impl Default for AtomicCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AtomicCounter {
    fn clone(&self) -> Self {
        Self::with_initial(self.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_counter_basic() {
        let counter = AtomicCounter::new();
        assert_eq!(counter.get(), 0);

        counter.increment();
        assert_eq!(counter.get(), 1);

        counter.decrement();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_atomic_counter_add_subtract() {
        let counter = AtomicCounter::new();
        
        counter.add(10);
        assert_eq!(counter.get(), 10);

        counter.subtract(5);
        assert_eq!(counter.get(), 5);
    }

    #[test]
    fn test_atomic_counter_max_min() {
        let counter = AtomicCounter::with_initial(5);
        
        counter.max(3); // Should remain 5
        assert_eq!(counter.get(), 5);

        counter.max(10); // Should become 10
        assert_eq!(counter.get(), 10);

        counter.min(15); // Should remain 10
        assert_eq!(counter.get(), 10);

        counter.min(7); // Should become 7
        assert_eq!(counter.get(), 7);
    }

    #[test]
    fn test_atomic_counter_reset() {
        let counter = AtomicCounter::with_initial(42);
        let prev = counter.reset();
        assert_eq!(prev, 42);
        assert_eq!(counter.get(), 0);
    }
}