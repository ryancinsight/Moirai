//! Sender half of a unified channel.

use std::marker::PhantomData;
use std::sync::Arc;

use super::core::UnifiedChannel;
use super::error::UnifiedChannelError;

/// Sender half of a unified channel
pub struct UnifiedSender<T> {
    pub(crate) channel: Arc<UnifiedChannel<T>>,
    pub(crate) _phantom: PhantomData<T>,
}

impl<T> UnifiedSender<T> {
    /// Send a message
    pub fn send(&self, message: T) -> Result<(), UnifiedChannelError> {
        self.channel.send(message)
    }

    /// Try to send without blocking
    pub fn try_send(&self, message: T) -> Result<(), (T, UnifiedChannelError)> {
        self.channel.try_send(message)
    }

    /// Send batch of messages
    pub fn send_batch(&self, messages: Vec<T>) -> Result<usize, UnifiedChannelError> {
        self.channel.send_batch(messages)
    }

    /// Check if channel is closed
    pub fn is_closed(&self) -> bool {
        self.channel.is_closed()
    }
}

impl<T> Clone for UnifiedSender<T> {
    fn clone(&self) -> Self {
        Self {
            channel: self.channel.clone(),
            _phantom: PhantomData,
        }
    }
}

// Safety: UnifiedSender is safe to send and share between threads
unsafe impl<T: Send> Send for UnifiedSender<T> {}
unsafe impl<T: Send> Sync for UnifiedSender<T> {}
