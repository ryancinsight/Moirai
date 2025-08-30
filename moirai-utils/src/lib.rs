//! Utility functions and data structures for Moirai concurrency library.
//!
//! This crate provides modular utility components organized by domain:
//! 
//! - [`cache`] - Cache alignment utilities for performance optimization
//! - [`atomic`] - Atomic operations and counters for lock-free programming  
//! - [`queue`] - Lock-free queues and ring buffers for high-performance data structures
//! - [`backoff`] - Exponential backoff strategies for retry operations
//! - [`random`] - Fast pseudo-random number generation for performance-critical scenarios
//! - [`bits`] - Bit manipulation utilities for high-performance computing
//! - [`memory`] - Memory utilities for cache optimization and prefetching
//! - [`time`] - High-resolution timing utilities for performance measurement (std only)

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

// Modular organization following SOC and domain-oriented design
pub mod cache;
pub mod atomic;
pub mod queue;
pub mod backoff;
pub mod random;
pub mod bits;
pub mod memory;

#[cfg(feature = "std")]
pub mod time;

// Re-export commonly used types for convenience
pub use cache::{CacheAligned, CACHE_LINE_SIZE, align_to_cache_line};
pub use atomic::AtomicCounter;
pub use queue::{RingBuffer, LockFreeQueue};
pub use backoff::Backoff;
pub use random::XorshiftRng;
pub use memory::{prefetch_read, prefetch_write, aligned_vec};

#[cfg(feature = "std")]
pub use time::{HighResTimer, unix_timestamp_nanos, unix_timestamp_micros, unix_timestamp_millis};

// Legacy re-exports for backward compatibility - these maintain the old flat structure
// while the new modular structure is the preferred approach
pub use bits::*;

/// Type alias for boxed errors.
#[cfg(feature = "std")]
pub type BoxError = std::boxed::Box<dyn std::error::Error + Send + Sync>;

/// Type alias for results with boxed errors.
#[cfg(feature = "std")]
pub type Result<T> = std::result::Result<T, BoxError>;

#[cfg(not(feature = "std"))]
pub type BoxError = alloc::boxed::Box<dyn core::error::Error + Send + Sync>;

#[cfg(not(feature = "std"))]
pub type Result<T> = core::result::Result<T, BoxError>;

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

        let buffer = RingBuffer::<i32>::new(4);
        buffer.try_push(1).unwrap();
        assert_eq!(buffer.try_pop(), Some(1));

        let mut rng = XorshiftRng::new(12345);
        let random_val = rng.next_u64();
        assert_ne!(random_val, 0);

        let backoff = Backoff::new();
        assert_eq!(backoff.current_step(), 0);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(align_to_cache_line(1), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE), CACHE_LINE_SIZE);
        assert_eq!(align_to_cache_line(CACHE_LINE_SIZE + 1), CACHE_LINE_SIZE * 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_time_integration() {
        let timer = HighResTimer::new();
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(timer.elapsed_millis() >= 1);
        
        let timestamp = unix_timestamp_millis();
        assert!(timestamp > 0);
    }
}