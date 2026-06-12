use std::hint;
use std::sync::atomic::{AtomicU64, Ordering};

/// A wait group for synchronizing multiple threads (Go-inspired).
/// This provides value beyond standard library primitives.
pub struct WaitGroup {
    counter: AtomicU64,
    generation: AtomicU64,
}

impl WaitGroup {
    /// Create a new wait group.
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
            generation: AtomicU64::new(0),
        }
    }

    /// Add to the wait group counter.
    pub fn add(&self, delta: u64) {
        self.counter.fetch_add(delta, Ordering::Release);
    }

    /// Decrement the wait group counter.
    pub fn done(&self) {
        let old = self.counter.fetch_sub(1, Ordering::Release);
        if old == 1 {
            // Last one out, increment generation to wake waiters
            self.generation.fetch_add(1, Ordering::Release);
            std::thread::yield_now(); // Give waiters a chance to wake
        }
    }

    /// Wait for the counter to reach zero.
    pub fn wait(&self) {
        let gen = self.generation.load(Ordering::Acquire);
        while self.counter.load(Ordering::Acquire) > 0 {
            hint::spin_loop();
            if self.generation.load(Ordering::Acquire) != gen {
                break;
            }
        }
    }
}
