use crate::channel::error::Result;
use crate::communication::RingBuffer;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;

use super::notify::notify_consumers;

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
    ///
    /// # Errors
    /// Returns [`ChannelError::Closed`](crate::channel::error::ChannelError::Closed)
    /// when the receiver is gone and
    /// [`ChannelError::Full`](crate::channel::error::ChannelError::Full) when the
    /// ring has no free slot.
    pub fn send(&self, value: T) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(crate::channel::error::ChannelError::Closed);
        }

        self.ring
            .try_produce(value)
            .map_err(|_| crate::channel::error::ChannelError::Full)?;

        // Fenced Dekker gate between the produce above and the counter loads
        // (see `notify_consumers`): skipping it lets this thread miss a
        // concurrent registration while the registrant misses the produce.
        self.notify_consumers();
        Ok(())
    }

    /// Try to send without blocking
    ///
    /// # Errors
    /// Propagates [`Self::send`], which never blocks.
    pub fn try_send(&self, value: T) -> Result<()> {
        self.send(value)
    }

    /// Send with timeout
    ///
    /// # Errors
    /// Returns [`ChannelError::Closed`](crate::channel::error::ChannelError::Closed)
    /// when the receiver is gone and
    /// [`ChannelError::Full`](crate::channel::error::ChannelError::Full) when no
    /// slot freed within `timeout`.
    pub fn send_timeout(&self, mut value: T, timeout: std::time::Duration) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(crate::channel::error::ChannelError::Closed);
            }

            match self.ring.try_produce(value) {
                Ok(()) => {
                    // Same fenced gate as `send`.
                    self.notify_consumers();
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

    fn notify_consumers(&self) {
        notify_consumers(
            &self.parker,
            &self.parked_count,
            &self.async_wakers,
            &self.waker_count,
        );
    }
}

impl<T> Drop for HybridSender<T> {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        // Fenced Dekker gate between the close above and the counter loads
        // (see `notify_consumers`): a receiver registering concurrently with
        // this drop must not park against a channel that will never send.
        notify_consumers(
            &self.parker,
            &self.parked_count,
            &self.async_wakers,
            &self.waker_count,
        );
    }
}
