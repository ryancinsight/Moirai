use crate::channel::error::Result;
use crate::communication::RingBuffer;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;

/// Receiver half of hybrid channel
pub struct HybridReceiver<T> {
    pub(super) ring: Arc<RingBuffer<T>>,
    pub(super) parker: Arc<Mutex<Vec<std::thread::Thread>>>,
    pub(super) async_wakers: Arc<Mutex<Vec<(u64, Waker)>>>,
    pub(super) parked_count: Arc<AtomicUsize>,
    pub(super) waker_count: Arc<AtomicUsize>,
    pub(super) closed: Arc<AtomicBool>,
    pub(super) next_id: Arc<AtomicU64>,
    pub(super) _marker: PhantomData<std::cell::Cell<()>>,
}

impl<T: Send> HybridReceiver<T> {
    /// Receive value with zero-copy
    pub fn recv(&self) -> Result<T> {
        // Fast path: try to receive without blocking
        if let Some(value) = self.ring.try_consume() {
            return Ok(value);
        }

        // Slow path: park the thread and wait for notification
        let current_thread = std::thread::current();

        loop {
            if self.closed.load(Ordering::Acquire) && self.ring.is_empty() {
                return Err(crate::channel::error::ChannelError::Closed);
            }

            // Register this thread for unparking if not already present
            if let Ok(mut parked) = self.parker.lock() {
                if !parked.iter().any(|t| t.id() == current_thread.id()) {
                    parked.push(current_thread.clone());
                    self.parked_count.fetch_add(1, Ordering::Release);
                }
            }

            // Check again after registering (to avoid race)
            if let Some(value) = self.ring.try_consume() {
                // Remove ourselves from the parker list
                if let Ok(mut parked) = self.parker.lock() {
                    let old_len = parked.len();
                    parked.retain(|t| !t.id().eq(&current_thread.id()));
                    let removed = old_len - parked.len();
                    if removed > 0 {
                        self.parked_count.fetch_sub(removed, Ordering::Release);
                    }
                }
                return Ok(value);
            }

            if self.closed.load(Ordering::Acquire) && self.ring.is_empty() {
                if let Ok(mut parked) = self.parker.lock() {
                    let old_len = parked.len();
                    parked.retain(|t| !t.id().eq(&current_thread.id()));
                    let removed = old_len - parked.len();
                    if removed > 0 {
                        self.parked_count.fetch_sub(removed, Ordering::Release);
                    }
                }
                return Err(crate::channel::error::ChannelError::Closed);
            }

            // Park until unparked by sender
            std::thread::park();
        }
    }

    /// Try to receive without blocking
    pub fn try_recv(&self) -> Result<T> {
        if let Some(value) = self.ring.try_consume() {
            Ok(value)
        } else if self.closed.load(Ordering::Acquire) {
            Err(crate::channel::error::ChannelError::Closed)
        } else {
            Err(crate::channel::error::ChannelError::Empty)
        }
    }

    /// Receive with timeout
    pub fn recv_timeout(&self, timeout: std::time::Duration) -> Result<T> {
        let start = std::time::Instant::now();

        loop {
            match self.try_recv() {
                Ok(value) => return Ok(value),
                Err(crate::channel::error::ChannelError::Empty) => {
                    if start.elapsed() >= timeout {
                        return Err(crate::channel::error::ChannelError::Empty);
                    }

                    // Register for wake-up before checking again
                    let current_thread = std::thread::current();
                    if let Ok(mut parked) = self.parker.lock() {
                        if !parked.iter().any(|t| t.id() == current_thread.id()) {
                            parked.push(current_thread.clone());
                            self.parked_count.fetch_add(1, Ordering::Release);
                        }
                    }

                    // Park for the remaining timeout budget.
                    if let Some(remaining) = timeout.checked_sub(start.elapsed()) {
                        std::thread::park_timeout(remaining);
                    }

                    // Remove from parker list
                    if let Ok(mut parked) = self.parker.lock() {
                        let old_len = parked.len();
                        parked.retain(|t| !t.id().eq(&current_thread.id()));
                        let removed = old_len - parked.len();
                        if removed > 0 {
                            self.parked_count.fetch_sub(removed, Ordering::Release);
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Check if there are messages available
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Get the number of messages available
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Drain all available messages
    pub fn drain(&self) -> Vec<T> {
        let mut messages = Vec::new();
        while let Ok(msg) = self.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// Async receive for use in async contexts (zero-cost waker-based Future)
    #[cfg(feature = "std")]
    pub fn recv_async(&self) -> super::future::RecvFuture<'_, T> {
        super::future::RecvFuture {
            receiver: self,
            id: None,
        }
    }
}

impl<T> Drop for HybridReceiver<T> {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        // Unpark any waiting threads (just in case) if there are any
        if self.parked_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut parked) = self.parker.lock() {
                for thread in parked.drain(..) {
                    thread.unpark();
                }
                self.parked_count.store(0, Ordering::Release);
            }
        }
        // Wake any waiting async tasks (just in case) if there are any
        if self.waker_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut wakers) = self.async_wakers.lock() {
                for (_, waker) in wakers.drain(..) {
                    waker.wake();
                }
                self.waker_count.store(0, Ordering::Release);
            }
        }
    }
}
