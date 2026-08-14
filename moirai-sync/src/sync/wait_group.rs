#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::fmt;
use std::sync::{Condvar, Mutex};

/// A wait group for synchronizing multiple threads (Go-inspired).
/// This provides value beyond standard library primitives.
pub struct WaitGroup {
    state: Mutex<WaitGroupState>,
    cond: Condvar,
}

struct WaitGroupState {
    counter: u64,
}

impl fmt::Debug for WaitGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("WaitGroup")
            .field("counter", &state.counter)
            .finish()
    }
}

impl Default for WaitGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl WaitGroup {
    /// Create a new wait group.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(WaitGroupState { counter: 0 }),
            cond: Condvar::new(),
        }
    }

    /// Add to the wait group counter.
    pub fn add(&self, delta: u64) {
        if delta == 0 {
            return;
        }
        let mut state = self.state.lock().unwrap();
        state.counter = state
            .counter
            .checked_add(delta)
            .expect("WaitGroup counter overflow");
    }

    /// Decrement the wait group counter.
    pub fn done(&self) {
        let mut state = self.state.lock().unwrap();
        if state.counter == 0 {
            panic!("WaitGroup counter decremented below zero");
        }
        state.counter -= 1;
        if state.counter == 0 {
            self.cond.notify_all();
        }
    }

    /// Wait for the counter to reach zero.
    pub fn wait(&self) {
        let mut state = self.state.lock().unwrap();
        while state.counter > 0 {
            state = self.cond.wait(state).unwrap();
        }
    }
}
