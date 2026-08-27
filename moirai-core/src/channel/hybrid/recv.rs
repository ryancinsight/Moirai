use crate::channel::error::Result;
use crate::communication::RingBuffer;
use std::marker::PhantomData;
use std::sync::atomic::{fence, AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;

use super::notify::notify_consumers;

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
    /// Register the calling thread for sender unparks.
    ///
    /// The `SeqCst` increment is half of the Dekker pair documented on
    /// [`notify_consumers`](super::notify::notify_consumers); the caller must
    /// execute `fence(SeqCst)` and re-check the ring (and `closed`) before
    /// parking, or a concurrent send can miss this registration while the
    /// re-check misses its message — the last-message hang.
    fn register_parked(&self, current_thread: &std::thread::Thread) {
        let mut parked = self
            .parker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !parked.iter().any(|t| t.id() == current_thread.id()) {
            parked.push(current_thread.clone());
            self.parked_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Remove the calling thread from the unpark registry.
    fn deregister_parked(&self, current_thread: &std::thread::Thread) {
        let mut parked = self
            .parker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let old_len = parked.len();
        parked.retain(|t| !t.id().eq(&current_thread.id()));
        let removed = old_len - parked.len();
        if removed > 0 {
            self.parked_count.fetch_sub(removed, Ordering::SeqCst);
        }
    }

    /// Receive value with zero-copy
    ///
    /// # Errors
    /// Returns [`ChannelError::Closed`](crate::channel::error::ChannelError::Closed)
    /// when the sender is gone and the ring is drained.
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

            self.register_parked(&current_thread);

            // Dekker fence between the registration above and the re-checks
            // below; pairs with the fence in `notify_consumers` between the
            // sender's publication and its counter gate loads. Without both
            // fences a StoreLoad reorder lets the sender read a zero count
            // while this re-check reads an empty ring, and the thread parks
            // against a delivered message.
            fence(Ordering::SeqCst);

            if let Some(value) = self.ring.try_consume() {
                self.deregister_parked(&current_thread);
                return Ok(value);
            }

            if self.closed.load(Ordering::Acquire) && self.ring.is_empty() {
                self.deregister_parked(&current_thread);
                return Err(crate::channel::error::ChannelError::Closed);
            }

            // Park until unparked by sender
            std::thread::park();
        }
    }

    /// Try to receive without blocking
    ///
    /// # Errors
    /// Returns [`ChannelError::Empty`](crate::channel::error::ChannelError::Empty)
    /// when no message is ready and
    /// [`ChannelError::Closed`](crate::channel::error::ChannelError::Closed) when
    /// the sender is gone.
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
    ///
    /// # Errors
    /// Returns [`ChannelError::Empty`](crate::channel::error::ChannelError::Empty)
    /// when `timeout` elapses without a message and
    /// [`ChannelError::Closed`](crate::channel::error::ChannelError::Closed) when
    /// the sender is gone.
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
                    self.register_parked(&current_thread);

                    // Same Dekker fence-and-re-check as `recv`: without it a
                    // send racing this registration is missed on both sides
                    // and the thread pays the full remaining timeout for a
                    // message that is already in the ring.
                    fence(Ordering::SeqCst);
                    if let Some(value) = self.ring.try_consume() {
                        self.deregister_parked(&current_thread);
                        return Ok(value);
                    }

                    // Park for the remaining timeout budget.
                    if let Some(remaining) = timeout.checked_sub(start.elapsed()) {
                        std::thread::park_timeout(remaining);
                    }

                    self.deregister_parked(&current_thread);
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
        // Fenced Dekker gate between the close above and the counter loads
        // (see `notify_consumers`), mirroring the sender drop.
        notify_consumers(
            &self.parker,
            &self.parked_count,
            &self.async_wakers,
            &self.waker_count,
        );
    }
}
