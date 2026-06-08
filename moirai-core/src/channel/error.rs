//! Error types, cache-padding helper, unified `Channel` trait, and `Result` alias.

use std::fmt;

/// Padding to prevent false sharing between CPU cores
#[repr(align(64))]
pub(super) struct CachePadded<T> {
    pub(super) value: T,
}

impl<T> CachePadded<T> {
    pub(super) const fn new(value: T) -> Self {
        Self { value }
    }
}

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
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "channel is full"),
            Self::Empty => write!(f, "channel is empty"),
            Self::Closed => write!(f, "channel is closed"),
            Self::WouldBlock => write!(f, "operation would block"),
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
}
