//! Error types for zero-copy operations.

/// Error types for zero-copy operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZeroCopyError {
    /// Channel is full
    Full,
    /// Channel is empty
    Empty,
    /// Channel is closed
    Closed,
    /// Operation would block
    WouldBlock,
    /// Memory mapping failed
    MemoryMapFailed,
    /// Invalid buffer size
    InvalidBufferSize,
    /// Alignment error
    AlignmentError,
    /// No route found for domain
    NoRoute,
}

impl std::fmt::Display for ZeroCopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "Zero-copy channel is full"),
            Self::Empty => write!(f, "Zero-copy channel is empty"),
            Self::Closed => write!(f, "Zero-copy channel is closed"),
            Self::WouldBlock => write!(f, "Zero-copy operation would block"),
            Self::MemoryMapFailed => write!(f, "Memory mapping failed"),
            Self::InvalidBufferSize => write!(f, "Invalid buffer size"),
            Self::AlignmentError => write!(f, "Memory alignment error"),
            Self::NoRoute => write!(f, "No route found for domain"),
        }
    }
}

impl std::error::Error for ZeroCopyError {}

/// Result type for zero-copy operations.
pub type ZeroCopyResult<T> = Result<T, ZeroCopyError>;
