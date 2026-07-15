use crate::channel::error::Result;
use crate::communication::RingBuffer;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;

/// Sender half of hybrid channel
pub struct HybridSender<T> {
    pub(super) ring: Arc<RingBuffer<T>>,
    pub(super) parker: Arc<Mutex<Vec<std::thread::Thread>>>,
    pub(super) async_wakers: Arc<Mutex<Vec<(u64, Waker)>>>,
    pub(super) parked_count: Arc<AtomicUsize>,
    pub(super) waker_count: Arc<AtomicUsize>,
    pub(super) closed: Arc<AtomicBool>,
    pub(super) _marker: PhantomData<std::cell::Cell<()>>,
}

impl<T: Send> HybridSender<T> {
    /// Send value with zero-copy when possible
    pub fn send(&self, value: T) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(crate::channel::error::ChannelError::Closed);
        }

        self.ring
            .try_produce(value)
            .map_err(|_| crate::channel::error::ChannelError::Full)?;

        // Unpark any waiting threads if there are any
        if self.parked_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut parked) = self.parker.lock() {
                for thread in parked.drain(..) {
                    thread.unpark();
                }
                self.parked_count.store(0, Ordering::Release);
            }
        }

        // Wake any waiting async tasks if there are any
        if self.waker_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut wakers) = self.async_wakers.lock() {
                for (_, waker) in wakers.drain(..) {
                    waker.wake();
                }
                self.waker_count.store(0, Ordering::Release);
            }
        }

        Ok(())
    }

    /// Try to send without blocking
    pub fn try_send(&self, value: T) -> Result<()> {
        self.send(value)
    }

    /// Send with timeout
    pub fn send_timeout(&self, mut value: T, timeout: std::time::Duration) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(crate::channel::error::ChannelError::Closed);
            }

            match self.ring.try_produce(value) {
                Ok(()) => {
                    // Unpark any waiting threads if there are any
                    if self.parked_count.load(Ordering::Relaxed) > 0 {
                        if let Ok(mut parked) = self.parker.lock() {
                            for thread in parked.drain(..) {
                                thread.unpark();
                            }
                            self.parked_count.store(0, Ordering::Release);
                        }
                    }

                    // Wake any waiting async tasks if there are any
                    if self.waker_count.load(Ordering::Relaxed) > 0 {
                        if let Ok(mut wakers) = self.async_wakers.lock() {
                            for (_, waker) in wakers.drain(..) {
                                waker.wake();
                            }
                            self.waker_count.store(0, Ordering::Release);
                        }
                    }

                    return Ok(());
                }
                Err(v) => {
                    value = v;
                    if start.elapsed() >= timeout {
                        return Err(crate::channel::error::ChannelError::Full);
                    }
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Check if the sender can send without blocking
    pub fn can_send(&self) -> bool {
        !self.ring.is_full() && !self.closed.load(Ordering::Acquire)
    }

    /// Get the number of items that can be sent without blocking
    pub fn available_capacity(&self) -> usize {
        if self.closed.load(Ordering::Acquire) {
            0
        } else {
            self.ring.capacity() - self.ring.len()
        }
    }
}

impl<T> Drop for HybridSender<T> {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        // Unpark any waiting threads if there are any
        if self.parked_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut parked) = self.parker.lock() {
                for thread in parked.drain(..) {
                    thread.unpark();
                }
                self.parked_count.store(0, Ordering::Release);
            }
        }
        // Wake any waiting async tasks if there are any
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
