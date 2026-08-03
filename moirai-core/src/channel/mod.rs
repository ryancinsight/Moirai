//! Unified high-performance channel implementations for Moirai.
//!
//! This module provides zero-cost channel abstractions that work seamlessly
//! across different execution contexts following DRY and SOLID principles.
//!
//! # Design Principles
//! - **Zero-copy**: Minimize data copies for maximum performance
//! - **Cache-friendly**: Align data structures to cache lines
//! - **Lock-free**: Use atomic operations where possible
//! - **Unified API**: Single interface for different channel types
//!
//! # Safety
//! All channel implementations maintain memory safety through:
//! - Sequence number validation before reading uninitialized memory
//! - Proper memory ordering with acquire-release semantics
//! - Safe cleanup on drop with reference counting

pub mod config;
pub mod error;
pub mod hybrid;
pub mod mpmc;
pub mod roles;
pub mod select;
pub mod spsc;
pub mod stats;
pub mod unified;

pub use config::ChannelConfig;
pub use error::{Channel, ChannelError, Result};
pub use hybrid::{HybridChannel, HybridReceiver, HybridSender};
pub use mpmc::{MpmcChannel, MpmcReceiver, MpmcSender};
pub use roles::{Consumer, Producer};
pub use select::{mpmc, spsc, unbounded, Select};
pub use spsc::{SpscConsumer, SpscProducer, SpscReceiver, SpscRing, SpscSender};
pub use stats::ChannelStatistics;
pub use unified::{
    unified_channel, unified_channel_with_config, UnifiedChannel, UnifiedReceiver, UnifiedSender,
};

#[cfg(test)]
mod tests;
