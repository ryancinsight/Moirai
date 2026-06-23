use std::fmt;
use std::hint;
use std::sync::atomic::{AtomicU64, Ordering};

/// A wait group for synchronizing multiple threads (Go-inspired).
/// This provides value beyond standard library primitives.
pub struct WaitGroup {
    counter: AtomicU64,
    generation: AtomicU64,
}

impl fmt::Debug for WaitGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let counter = self.counter.load(Ordering::Relaxed);
        let generation = self.generation.load(Ordering::Relaxed);
        f.debug_struct("WaitGroup")
            .field("counter", &counter)
            .field("generation", &generation)
            .finish()
    }
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
        let old = self.counter.fetch_sub(1, Ordering::AcqRel);
        if old == 1 {
            // Last one out, increment generation to wake waiters
            self.generation.fetch_add(1, Ordering::Release);
            std::thread::yield_now(); // Give waiters a chance to wake
        }
    }

    /// Wait for the counter to reach zero.
    pub fn wait(&self) {
        let gen = self.generation.load(Ordering::Acquire);
        let mut backoff: usize = 1;
        while self.counter.load(Ordering::Acquire) > 0 {
            for _ in 0..backoff {
                hint::spin_loop();
            }
            if backoff < 64 {
                backoff = backoff.saturating_mul(2);
            } else {
                std::thread::yield_now();
                backoff = 1;
            }
            if self.generation.load(Ordering::Acquire) != gen {
                break;
            }
        }
    }
}
