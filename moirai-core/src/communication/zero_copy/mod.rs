//! Zero-copy communication primitives (consolidated sub-modules).

mod adaptive;
mod channel;
mod error;
mod ring;
mod router;

pub use adaptive::{
    AdaptiveBatchChannel, AdaptiveBatchReceiver, AdaptiveBatchSender, AdaptiveThreshold,
    BatchStats, ThroughputMonitor,
};
pub use channel::{ZeroCopyChannel, ZeroCopyReceiver, ZeroCopySender};
pub use error::{ZeroCopyError, ZeroCopyResult};
pub use ring::MemoryMappedRing;
pub use router::{DomainId, ZeroCopyRouter};
