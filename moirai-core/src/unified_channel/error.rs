//! Unified channel error types.

/// Unified channel error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifiedChannelError {
    /// Channel buffer is full
    Full,
    /// Channel buffer is empty  
    Empty,
    /// Channel has been closed
    Closed,
    /// Operation would block in non-blocking mode
    WouldBlock,
    /// Invalid channel configuration
    InvalidConfig,
}

impl std::fmt::Display for UnifiedChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "channel buffer is full"),
            Self::Empty => write!(f, "channel buffer is empty"),
            Self::Closed => write!(f, "channel has been closed"),
            Self::WouldBlock => write!(f, "operation would block"),
            Self::InvalidConfig => write!(f, "invalid channel configuration"),
        }
    }
}

impl std::error::Error for UnifiedChannelError {}
