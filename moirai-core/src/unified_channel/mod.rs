//! Unified channel architecture with memory-efficient design.

use std::sync::Arc;

/// Config types.
pub mod config;
/// Core logic.
pub mod core;
/// Error types.
pub mod error;
/// Receiver half.
pub mod receiver;
/// Sender half.
pub mod sender;
/// Stats tracking.
pub mod stats;

pub use config::ChannelConfig;
pub use core::UnifiedChannel;
pub use error::UnifiedChannelError;
pub use receiver::UnifiedReceiver;
pub use sender::UnifiedSender;
pub use stats::ChannelStatistics;

/// Create a unified channel pair with default configuration
pub fn unified_channel<T>(
    capacity: usize,
) -> Result<(UnifiedSender<T>, UnifiedReceiver<T>), UnifiedChannelError> {
    let channel = Arc::new(UnifiedChannel::with_capacity(capacity)?);

    let sender = UnifiedSender {
        channel: channel.clone(),
        _phantom: std::marker::PhantomData,
    };

    let receiver = UnifiedReceiver {
        channel,
        _phantom: std::marker::PhantomData,
    };

    Ok((sender, receiver))
}

/// Create a unified channel with custom configuration
pub fn unified_channel_with_config<T>(
    config: ChannelConfig,
) -> Result<(UnifiedSender<T>, UnifiedReceiver<T>), UnifiedChannelError> {
    let channel = Arc::new(UnifiedChannel::new(config)?);

    let sender = UnifiedSender {
        channel: channel.clone(),
        _phantom: std::marker::PhantomData,
    };

    let receiver = UnifiedReceiver {
        channel,
        _phantom: std::marker::PhantomData,
    };

    Ok((sender, receiver))
}

#[cfg(test)]
mod tests;
