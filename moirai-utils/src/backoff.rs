//! Exponential backoff strategy for retry operations.
//!
//! This module provides adaptive backoff mechanisms for handling contention
//! and implementing retry strategies in concurrent systems.

use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "std")]
use std::thread;

/// Exponential backoff strategy for handling contention.
///
/// This implementation provides an adaptive backoff mechanism that starts
/// with a small delay and exponentially increases with each failure,
/// helping to reduce contention in high-load scenarios.
#[derive(Debug)]
pub struct Backoff {
    /// Current step in the backoff sequence
    step: AtomicUsize,
    /// Maximum number of steps before capping
    max_steps: usize,
}

impl Backoff {
    /// Create a new backoff instance with default parameters.
    pub const fn new() -> Self {
        Self {
            step: AtomicUsize::new(0),
            max_steps: 10,
        }
    }

    /// Create a new backoff instance with a custom maximum step count.
    pub const fn with_max_steps(max_steps: usize) -> Self {
        Self {
            step: AtomicUsize::new(0),
            max_steps,
        }
    }

    /// Perform a backoff operation, incrementing the internal step counter.
    /// This will either spin or yield depending on the current step.
    pub fn backoff(&self) {
        let step = self.step.fetch_add(1, Ordering::Relaxed);
        let effective_step = step.min(self.max_steps);

        if effective_step <= 6 {
            // For small steps, use CPU spin waiting
            for _ in 0..(1 << effective_step) {
                core::hint::spin_loop();
            }
        } else {
            // For larger steps, yield to the scheduler
            #[cfg(feature = "std")]
            thread::yield_now();
            
            #[cfg(not(feature = "std"))]
            core::hint::spin_loop();
        }
    }

    /// Reset the backoff to its initial state.
    pub fn reset(&self) {
        self.step.store(0, Ordering::Relaxed);
    }

    /// Get the current step number.
    pub fn current_step(&self) -> usize {
        self.step.load(Ordering::Relaxed)
    }

    /// Check if we've reached the maximum backoff level.
    pub fn is_max(&self) -> bool {
        self.current_step() >= self.max_steps
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Backoff {
    fn clone(&self) -> Self {
        Self::with_max_steps(self.max_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff_creation() {
        let backoff = Backoff::new();
        assert_eq!(backoff.current_step(), 0);
        assert!(!backoff.is_max());
    }

    #[test]
    fn test_backoff_progression() {
        let backoff = Backoff::with_max_steps(3);
        
        assert_eq!(backoff.current_step(), 0);
        
        backoff.backoff();
        assert_eq!(backoff.current_step(), 1);
        
        backoff.backoff();
        assert_eq!(backoff.current_step(), 2);
        
        backoff.backoff();
        assert_eq!(backoff.current_step(), 3);
        assert!(backoff.is_max());
    }

    #[test]
    fn test_backoff_reset() {
        let backoff = Backoff::new();
        
        backoff.backoff();
        backoff.backoff();
        assert_ne!(backoff.current_step(), 0);
        
        backoff.reset();
        assert_eq!(backoff.current_step(), 0);
        assert!(!backoff.is_max());
    }
}