//! Async-aware semaphore for resource limiting
//!
//! Provides semaphore synchronization primitive that integrates with Moirai's
//! async runtime, following SLAP principle with focused responsibility.
//! Waiter-queue mechanics live in `WaitQueue`; this module keeps only the
//! permit-counter admission predicate and the permit-restoration policy for
//! cancelled acquire futures.

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use crate::sync::wait_queue::{WaitQueue, WaiterPoll};

/// Async-aware semaphore for resource limiting
pub struct Semaphore {
    state: Mutex<SemaphoreState>,
}

struct SemaphoreState {
    available: usize,
    /// A grant hands a released permit directly to a waiter; the `()` payload
    /// carries no data because the grant itself is the permit.
    waiters: WaitQueue<()>,
}

impl Semaphore {
    /// Create a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Self {
        Self {
            state: Mutex::new(SemaphoreState {
                available: permits,
                waiters: WaitQueue::new(),
            }),
        }
    }

    /// Acquire a permit asynchronously
    pub fn acquire(&self) -> SemaphoreAcquire<'_> {
        SemaphoreAcquire {
            semaphore: self,
            id: None,
        }
    }

    /// Try to acquire a permit immediately
    pub fn try_acquire(&self) -> Option<SemaphorePermit<'_>> {
        let mut state = self.state.lock().unwrap();
        if state.available > 0 {
            state.available -= 1;
            Some(SemaphorePermit { semaphore: self })
        } else {
            None
        }
    }

    /// Get the number of available permits
    pub fn available_permits(&self) -> usize {
        self.state.lock().unwrap().available
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        match state.waiters.grant_oldest(()) {
            Some(waker) => waker.wake(),
            None => state.available += 1,
        }
    }
}

/// Future for acquiring a semaphore permit
pub struct SemaphoreAcquire<'a> {
    semaphore: &'a Semaphore,
    id: Option<u64>,
}

impl<'a> Future for SemaphoreAcquire<'a> {
    type Output = SemaphorePermit<'a>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.semaphore.state.lock().unwrap();

        // 1. Check if we were already registered and have been granted a permit
        if let Some(id) = self.id {
            match state.waiters.poll_waiter(id, cx.waker()) {
                WaiterPoll::Granted(()) => {
                    self.id = None;
                    return Poll::Ready(SemaphorePermit {
                        semaphore: self.semaphore,
                    });
                }
                WaiterPoll::Pending => return Poll::Pending,
                // registration lost; fall through to re-acquire/register
                WaiterPoll::NotRegistered => {}
            }
        }

        // 2. Try to acquire an available permit
        if state.available > 0 {
            state.available -= 1;
            if let Some(id) = self.id.take() {
                let _removed_grant = state.waiters.deregister(id);
            }
            return Poll::Ready(SemaphorePermit {
                semaphore: self.semaphore,
            });
        }

        // 3. Register as a waiter
        if self.id.is_none() {
            self.id = Some(state.waiters.register(cx.waker().clone()));
        }

        Poll::Pending
    }
}

impl<'a> Drop for SemaphoreAcquire<'a> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.semaphore.state.lock() {
                // If we were granted a permit we never consumed, hand it back so
                // it reaches another waiter (or the available count).
                if state.waiters.deregister(id).is_some() {
                    drop(state);
                    self.semaphore.release();
                }
            }
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
