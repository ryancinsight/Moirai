//! Unified channel architecture with memory-efficient design.

use std::sync::Arc;

use crate::channel::error::ChannelError;

/// Core logic.
pub mod core;
/// Receiver half.
pub mod receiver;
/// Sender half.
pub mod sender;

pub use core::UnifiedChannel;
pub use receiver::UnifiedReceiver;
pub use sender::UnifiedSender;

/// Create a unified channel pair with default configuration
pub fn unified_channel<T>(
    capacity: usize,
) -> Result<(UnifiedSender<T>, UnifiedReceiver<T>), ChannelError> {
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
    config: crate::channel::config::ChannelConfig,
) -> Result<(UnifiedSender<T>, UnifiedReceiver<T>), ChannelError> {
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
