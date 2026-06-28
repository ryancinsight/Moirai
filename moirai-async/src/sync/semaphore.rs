//! Async-aware semaphore for resource limiting
//!
//! Provides semaphore synchronization primitive that integrates with Moirai's
//! async runtime, following SLAP principle with focused responsibility.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Async-aware semaphore for resource limiting
pub struct Semaphore {
    permits: Arc<Mutex<SemaphoreState>>,
}

struct SemWaiter {
    waker: Waker,
    /// Set when a released permit has been handed to this waiter.
    granted: bool,
}

struct SemaphoreState {
    available: usize,
    /// Waiters keyed by a monotonic id. Keyed (rather than a linear `VecDeque`)
    /// so per-waiter `poll`/drop lookups and removals are O(log n) instead of
    /// O(n), shortening the time the single state lock is held when many tasks
    /// contend for permits. Monotonic ids keep in-order iteration FIFO, so a
    /// released permit goes to the oldest waiter first.
    waiters: BTreeMap<u64, SemWaiter>,
    next_id: u64,
}

impl SemaphoreState {
    /// Grant a released permit to the oldest ungranted waiter and return its
    /// waker to wake. Returns `None` if no waiter is ungranted.
    fn grant_oldest_waiter(&mut self) -> Option<Waker> {
        let waiter = self.waiters.values_mut().find(|w| !w.granted)?;
        waiter.granted = true;
        Some(waiter.waker.clone())
    }
}

impl Semaphore {
    /// Create a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Self {
        Self {
            permits: Arc::new(Mutex::new(SemaphoreState {
                available: permits,
                waiters: BTreeMap::new(),
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
        match state.grant_oldest_waiter() {
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
        let mut state = self.semaphore.permits.lock().unwrap();

        // 1. Check if we were already registered and have been granted a permit
        if let Some(id) = self.id {
            match state.waiters.get(&id).map(|w| w.granted) {
                Some(true) => {
                    state.waiters.remove(&id);
                    self.id = None;
                    return Poll::Ready(SemaphorePermit {
                        semaphore: self.semaphore,
                    });
                }
                Some(false) => {
                    state.waiters.get_mut(&id).unwrap().waker = cx.waker().clone();
                    return Poll::Pending;
                }
                None => {} // registration lost; fall through to re-acquire/register
            }
        }

        // 2. Try to acquire an available permit
        if state.available > 0 {
            state.available -= 1;
            if let Some(id) = self.id.take() {
                state.waiters.remove(&id);
            }
            return Poll::Ready(SemaphorePermit {
                semaphore: self.semaphore,
            });
        }

        // 3. Register as a waiter
        if self.id.is_none() {
            let id = state.next_id;
            state.next_id += 1;
            state.waiters.insert(
                id,
                SemWaiter {
                    waker: cx.waker().clone(),
                    granted: false,
                },
            );
            self.id = Some(id);
        }

        Poll::Pending
    }
}

impl<'a> Drop for SemaphoreAcquire<'a> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.semaphore.permits.lock() {
                // If we were granted a permit we never consumed, hand it back so
                // it reaches another waiter (or the available count).
                if state.waiters.remove(&id).is_some_and(|w| w.granted) {
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
