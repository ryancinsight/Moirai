//! Receiver half of a unified channel.

use std::marker::PhantomData;
use std::sync::Arc;

use super::core::UnifiedChannel;
use super::error::UnifiedChannelError;
use super::stats::ChannelStatistics;

/// Receiver half of a unified channel
pub struct UnifiedReceiver<T> {
    pub(crate) channel: Arc<UnifiedChannel<T>>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T> UnifiedReceiver<T> {
    /// Receive a message
    pub fn recv(&self) -> Result<T, UnifiedChannelError> {
        self.channel.recv()
    }

    /// Try to receive without blocking
    pub fn try_recv(&self) -> Result<T, UnifiedChannelError> {
        self.channel.try_recv()
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
