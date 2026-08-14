#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use crate::channel::error::{Channel, Result};
use crate::channel::mpmc::MpmcChannel;
use std::sync::Arc;

/// Sender half of MPMC channel
pub struct MpmcSender<T> {
    pub(super) channel: Arc<MpmcChannel<T>>,
}

impl<T: Send> MpmcSender<T> {
    /// Send a value through the channel, blocking if necessary
    pub fn send(&self, value: T) -> Result<()> {
        self.channel.send(value)
    }

    /// Try to send a value without blocking
    pub fn try_send(&self, value: T) -> Result<()> {
        self.channel.try_send(value)
    }
}

impl<T: Send> crate::channel::roles::Producer<T> for MpmcSender<T> {
    #[inline]
    fn send(&self, value: T) -> Result<()> {
        MpmcSender::send(self, value)
    }

    #[inline]
    fn try_send(&self, value: T) -> Result<()> {
        MpmcSender::try_send(self, value)
    }

    #[inline]
    fn is_full(&self) -> bool {
        Channel::is_full(&*self.channel)
    }

    #[inline]
    fn capacity(&self) -> Option<usize> {
        Channel::capacity(&*self.channel)
    }
}

impl<T> MpmcSender<T> {
    /// Returns `true` once the channel is closed (every receiver — or every
    /// sender — has been dropped); subsequent sends fail with
    /// [`crate::channel::error::ChannelError::Closed`].
    pub fn is_closed(&self) -> bool {
        self.channel
            .closed
            .load(std::sync::atomic::Ordering::Acquire)
    }
}

impl<T> Clone for MpmcSender<T> {
    fn clone(&self) -> Self {
        let (mutex, _, _) = &*self.channel.state;
        let mut guard = mutex.lock().unwrap();
        guard.sender_count += 1;
        Self {
            channel: self.channel.clone(),
        }
    }
}

impl<T> Drop for MpmcSender<T> {
    fn drop(&mut self) {
        let (mutex, _, not_empty) = &*self.channel.state;
        let mut guard = mutex.lock().unwrap();
        guard.sender_count -= 1;
        if guard.sender_count == 0 {
            guard.closed = true;
            self.channel
                .closed
                .store(true, std::sync::atomic::Ordering::Release);
            not_empty.notify_all();
        }
    }
}
