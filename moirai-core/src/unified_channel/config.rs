//! Configuration for unified channels.

use crate::constants::DEFAULT_RING_BUFFER_CAPACITY;

/// Channel configuration for unified memory management
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Buffer capacity (will be rounded to power of 2)
    pub capacity: usize,
    /// Whether to use memory pooling for overflow
    pub enable_pooling: bool,
    /// Maximum pool size for overflow handling
    pub max_pool_size: usize,
    /// Whether to enable batch operations
    pub enable_batching: bool,
    /// Batch size for bulk operations
    pub batch_size: usize,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_RING_BUFFER_CAPACITY,
            enable_pooling: true,
            max_pool_size: DEFAULT_RING_BUFFER_CAPACITY * 2,
            enable_batching: false,
            batch_size: 64,
        }
    }
}
