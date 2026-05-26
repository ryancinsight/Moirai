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
    waiters: VecDeque<(u64, Waker, bool)>,
    next_id: u64,
}

impl Semaphore {
    /// Create a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Self {
        Self {
            permits: Arc::new(Mutex::new(SemaphoreState {
                available: permits,
                waiters: VecDeque::new(),
                next_id: 0,
            })),
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
        if let Some(waiter) = state.waiters.iter_mut().find(|(_, _, woken)| !*woken) {
            waiter.2 = true;
            waiter.1.wake_by_ref();
        } else {
            state.available += 1;
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
        let mut state = self.semaphore.permits.lock().unwrap();

        // 1. Check if we were already registered and have been granted a permit
        if let Some(id) = self.id {
            if let Some(pos) = state.waiters.iter().position(|(w_id, _, _)| *w_id == id) {
                if state.waiters[pos].2 {
                    state.waiters.remove(pos);
                    self.id = None;
                    return Poll::Ready(SemaphorePermit {
                        semaphore: self.semaphore,
                    });
                } else {
                    state.waiters[pos].1 = cx.waker().clone();
                    return Poll::Pending;
                }
            }
        }

        // 2. Try to acquire an available permit
        if state.available > 0 {
            state.available -= 1;
            if let Some(id) = self.id.take() {
                state.waiters.retain(|(w_id, _, _)| *w_id != id);
            }
            return Poll::Ready(SemaphorePermit {
                semaphore: self.semaphore,
            });
        }

        // 3. Register as a waiter
        if self.id.is_none() {
            let id = state.next_id;
            state.next_id += 1;
            state.waiters.push_back((id, cx.waker().clone(), false));
            self.id = Some(id);
        }

        Poll::Pending
    }
}

impl<'a> Drop for SemaphoreAcquire<'a> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.semaphore.permits.lock() {
                if let Some(pos) = state.waiters.iter().position(|(w_id, _, _)| *w_id == id) {
                    let was_granted = state.waiters[pos].2;
                    state.waiters.remove(pos);
                    if was_granted {
                        drop(state);
                        self.semaphore.release();
                    }
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
