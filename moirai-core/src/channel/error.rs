//! Error types, cache-alignment helper, unified `Channel` trait, and `Result` alias.

use std::fmt;

pub(super) use moirai_utils::cache::CacheAligned;

use super::stats::ChannelStatistics;

/// Error types for channel operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelError {
    /// Channel is full and cannot accept more messages
    Full,
    /// Channel is empty and has no messages
    Empty,
    /// Channel has been closed
    Closed,
    /// Operation would block but non-blocking was requested
    WouldBlock,
    /// Invalid channel configuration
    InvalidConfig,
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "channel is full"),
            Self::Empty => write!(f, "channel is empty"),
            Self::Closed => write!(f, "channel is closed"),
            Self::WouldBlock => write!(f, "operation would block"),
            Self::InvalidConfig => write!(f, "invalid channel configuration"),
        }
    }
}

impl std::error::Error for ChannelError {}

/// Result type for channel operations
pub type Result<T> = std::result::Result<T, ChannelError>;

/// Trait for unified channel behavior following Interface Segregation Principle
pub trait Channel<T>: Send + Sync {
    /// Send a value, blocking if necessary
    fn send(&self, value: T) -> Result<()>;

    /// Try to send without blocking
    fn try_send(&self, value: T) -> Result<()>;

    /// Receive a value, blocking if necessary
    fn recv(&self) -> Result<T>;

    /// Try to receive without blocking
    fn try_recv(&self) -> Result<T>;

    /// Check if channel is empty
    fn is_empty(&self) -> bool;

    /// Check if channel is full
    fn is_full(&self) -> bool;

    /// Get the capacity of the channel
    fn capacity(&self) -> Option<usize>;

    /// Send multiple values in batch. Default sends each individually.
    fn send_batch(&self, values: Vec<T>) -> Result<usize> {
        let mut count = 0;
        for value in values {
            self.send(value)?;
            count += 1;
        }
        Ok(count)
    }

    /// Receive up to `max_count` values in batch. Default receives individually.
    fn recv_batch(&self, max_count: usize) -> Vec<T> {
        let mut results = Vec::with_capacity(max_count);
        for _ in 0..max_count {
            match self.recv() {
                Ok(value) => results.push(value),
                Err(_) => break,
            }
        }
        results
    }

    /// Close the channel. Default: no-op (channels that support close override).
    fn close(&self) {}

    /// Check if channel is closed. Default: false.
    fn is_closed(&self) -> bool {
        false
    }

    /// Current number of buffered items. Default: 0.
    fn len(&self) -> usize {
        0
    }

    /// Return statistics if the channel tracks them.
    fn stats(&self) -> Option<ChannelStatistics> {
        None
    }
}
