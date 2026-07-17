//! Receiver half of a unified channel.

use std::marker::PhantomData;
use std::sync::Arc;

use super::core::UnifiedChannel;
use crate::channel::error::ChannelError;
use crate::channel::stats::ChannelStatistics;

/// Receiver half of a unified channel
pub struct UnifiedReceiver<T> {
    pub(crate) channel: Arc<UnifiedChannel<T>>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T> UnifiedReceiver<T> {
    /// Receive a message (non-blocking; returns `Err(Empty)` when none is available).
    pub fn recv(&self) -> Result<T, ChannelError> {
        self.channel.recv()
    }

    /// Try to receive without blocking.
    ///
    /// Identical to [`Self::recv`] — this channel has no blocking receive
    /// path. Retained for consumers written against the `try_recv` name.
    pub fn try_recv(&self) -> Result<T, ChannelError> {
        self.channel.recv()
    }

    /// Receive batch of messages
    pub fn recv_batch(&self, max_count: usize) -> Vec<T> {
        self.channel.recv_batch(max_count)
    }

    /// Check if channel is closed
    pub fn is_closed(&self) -> bool {
        self.channel.is_closed()
    }

    /// Get channel statistics
    pub fn stats(&self) -> ChannelStatistics {
        self.channel.stats()
    }
}

impl<T> Clone for UnifiedReceiver<T> {
    fn clone(&self) -> Self {
        Self {
            channel: self.channel.clone(),
            _phantom: PhantomData,
        }
    }
}

// Safety: UnifiedReceiver is safe to send and share between threads
unsafe impl<T: Send> Send for UnifiedReceiver<T> {}
unsafe impl<T: Send> Sync for UnifiedReceiver<T> {}
