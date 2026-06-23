//! High-performance communication patterns for concurrent systems.

pub mod zero_copy;
pub use zero_copy::{
    AdaptiveBatchChannel, AdaptiveBatchReceiver, AdaptiveBatchSender, AdaptiveThreshold,
    BatchStats, DomainId, MemoryMappedRing, ThroughputMonitor, ZeroCopyChannel, ZeroCopyError,
    ZeroCopyReceiver, ZeroCopyResult, ZeroCopyRouter, ZeroCopySender,
};

/// Broadcast channel patterns for one-to-many communication.
pub mod broadcast;
/// Collective operations for group communication.
pub mod collective;
/// Message patterns for shared ownership communication.
pub mod message;
/// Publish-subscribe patterns for topic-based communication.
pub mod pubsub;
/// Ring buffer patterns for high-throughput streaming.
pub mod ring_buffer;
/// Message routing for key-based communication.
pub mod router;

pub use broadcast::{BroadcastChannel, BroadcastReceiver};
pub use collective::CollectiveOps;
pub use message::Message;
pub use pubsub::PubSub;
pub use ring_buffer::RingBuffer;
pub use router::MessageRouter;

#[cfg(test)]
mod tests;
