//! High-performance communication patterns for concurrent systems.

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
pub use collective::{ChunkedVec, CollectiveOps};
pub use message::Message;
pub use pubsub::PubSub;
pub use ring_buffer::RingBuffer;
pub use router::MessageRouter;

#[cfg(test)]
mod tests;
