//! Utility functions and data structures for Moirai concurrency library.
//!
//! This crate provides modular utility components organized by domain:
//!
//! - [`cache`] - Cache alignment utilities for performance optimization
//! - [`atomic`] - Atomic operations and counters for lock-free programming
//! - [`queue`] - Lock-free queues for high-performance data structures
//! - [`memory`] - Memory utilities for cache prefetching

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

// Modular organization following SOC and domain-oriented design
pub mod atomic;
pub mod cache;
pub mod memory;
pub mod queue;

// SIMD optimizations for high-performance computing
#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod simd;

// Re-export commonly used types for convenience
pub use atomic::AtomicCounter;
pub use cache::{align_to_cache_line, CacheAligned, CachePadded, CACHE_LINE_SIZE};
pub use memory::{prefetch_read, prefetch_write};
pub use queue::LockFreeQueue;

// SIMD optimization counter and scalar contracts for performance tracking.
#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use simd::{has_native_vector_path, SimdReal, SimdScalar};

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
use std::sync::OnceLock;

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
static GLOBAL_SIMD_COUNTER: OnceLock<SimdCounter> = OnceLock::new();

/// Get the global SIMD performance counter instance.
///
/// This provides a singleton counter for tracking SIMD vs scalar operation usage
/// across the entire application.
#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn global_simd_counter() -> &'static SimdCounter {
    GLOBAL_SIMD_COUNTER.get_or_init(SimdCounter::new)
}

/// Performance counter for tracking SIMD optimization usage.
///
/// This counter tracks the ratio of vectorized vs scalar operations
/// to help optimize performance-critical code paths.
#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[derive(Debug)]
pub struct SimdCounter {
    vectorized_ops: AtomicCounter,
    scalar_ops: AtomicCounter,
    vectorized_elements: AtomicCounter,
    scalar_elements: AtomicCounter,
}

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
impl SimdCounter {
    /// Create a new SIMD performance counter.
    pub fn new() -> Self {
        Self {
            vectorized_ops: AtomicCounter::new(),
            scalar_ops: AtomicCounter::new(),
            vectorized_elements: AtomicCounter::new(),
            scalar_elements: AtomicCounter::new(),
        }
    }

    /// Record a vectorized operation with the number of elements processed.
    pub fn record_vectorized_op(&self, elements: usize) {
        self.vectorized_ops.increment();
        self.vectorized_elements.add(elements);
    }

    /// Record a scalar operation with the number of elements processed.
    pub fn record_scalar_op(&self, elements: usize) {
        self.scalar_ops.increment();
        self.scalar_elements.add(elements);
    }

    /// Get the total number of vectorized operations performed.
    pub fn vectorized_ops(&self) -> usize {
        self.vectorized_ops.get()
    }

    /// Get the total number of scalar operations performed.
    pub fn scalar_ops(&self) -> usize {
        self.scalar_ops.get()
    }

    /// Calculate the vectorization rate as a fraction of total operations.
    pub fn vectorization_rate(&self) -> f64 {
        let total = self.vectorized_ops() + self.scalar_ops();
        if total == 0 {
            0.0
        } else {
            self.vectorized_ops() as f64 / total as f64
        }
    }

    /// Get basic statistics about SIMD usage.
    pub fn get_stats(&self) -> (usize, usize, usize, usize) {
        (
            self.vectorized_ops(),
            self.scalar_ops(),
            self.vectorized_elements.get(),
            self.scalar_elements.get(),
        )
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.vectorized_ops.reset();
        self.scalar_ops.reset();
        self.vectorized_elements.reset();
        self.scalar_elements.reset();
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
impl Default for SimdCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_modular_integration() {
        // Test that all modules work together
        let aligned_data = CacheAligned::new(42);
        assert_eq!(*aligned_data, 42);

        let counter = AtomicCounter::new();
        counter.increment();
        assert_eq!(counter.get(), 1);

        let queue = LockFreeQueue::<i32>::with_capacity(4);
        queue.enqueue(1);
        assert_eq!(queue.try_dequeue(), Some(1));
    }

    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(align_to_cache_line(1), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE), CACHE_LINE_SIZE);
        assert_eq!(
            align_to_cache_line(CACHE_LINE_SIZE + 1),
            CACHE_LINE_SIZE * 2
        );
    }
}
