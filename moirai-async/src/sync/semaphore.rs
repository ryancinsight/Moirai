//! Async-aware semaphore for resource limiting
//!
//! Provides semaphore synchronization primitive that integrates with Moirai's
//! async runtime, following SLAP principle with focused responsibility.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Async-aware semaphore for resource limiting
pub struct Semaphore {
    permits: Arc<Mutex<SemaphoreState>>,
}

struct SemaphoreState {
    available: usize,
    waiters: VecDeque<Waker>,
}

impl Semaphore {
    /// Create a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Self {
        Self {
            permits: Arc::new(Mutex::new(SemaphoreState {
                available: permits,
                waiters: VecDeque::new(),
            })),
        }
    }

    /// Acquire a permit asynchronously
    pub fn acquire(&self) -> SemaphoreAcquire<'_> {
        SemaphoreAcquire {
            semaphore: self,
            registered: false,
        }
    }

    /// Try to acquire a permit immediately
    pub fn try_acquire(&self) -> Option<SemaphorePermit<'_>> {
        let mut state = self.permits.lock().unwrap();
        if state.available > 0 {
            state.available -= 1;
            Some(SemaphorePermit { semaphore: self })
        } else {
            None
        }
    }

    /// Get the number of available permits
    pub fn available_permits(&self) -> usize {
        self.permits.lock().unwrap().available
    }

    fn release(&self) {
        let mut state = self.permits.lock().unwrap();
        state.available += 1;
        if let Some(waker) = state.waiters.pop_front() {
            drop(state);
            waker.wake();
        }
    }
}

/// Future for acquiring a semaphore permit
pub struct SemaphoreAcquire<'a> {
    semaphore: &'a Semaphore,
    registered: bool,
}

impl<'a> Future for SemaphoreAcquire<'a> {
    type Output = SemaphorePermit<'a>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.semaphore.permits.lock().unwrap();
        
        if state.available > 0 {
            state.available -= 1;
            Poll::Ready(SemaphorePermit {
                semaphore: self.semaphore,
            })
        } else {
            if !self.registered {
                state.waiters.push_back(cx.waker().clone());
                self.registered = true;
            }
            Poll::Pending
        }
    }
}

/// RAII guard for semaphore permit
pub struct SemaphorePermit<'a> {
    semaphore: &'a Semaphore,
}

impl<'a> Drop for SemaphorePermit<'a> {
    fn drop(&mut self) {
        self.semaphore.release();
    }
}