//! Configuration for unified channels.

/// Capacity used by every channel constructor that does not take one.
///
/// Channels are bounded by default: an unbounded queue lets a producer that
/// outruns its consumer convert backpressure into unbounded memory growth,
/// which is a liveness failure under adversarial or merely bursty load. The
/// default therefore has to be a number, and this is it.
///
/// Derivation: 1024 slots is large enough that a producer burst spanning a
/// scheduler quantum never blocks on an otherwise-keeping-up consumer (the
/// runtime's own queue capacity is 256 jobs per worker), and small enough that
/// the resident bound is one pointer-sized slot array — 8 KiB for a
/// word-sized payload — per channel rather than a function of producer speed.
/// Power of two so the bounded MPMC ring uses it without rounding up.
///
/// Callers whose producer/consumer rates are known should pass an explicit
/// capacity instead of inheriting this one.
pub const DEFAULT_CHANNEL_CAPACITY: usize = 1024;

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
            capacity: DEFAULT_CHANNEL_CAPACITY,
            enable_pooling: true,
            max_pool_size: DEFAULT_CHANNEL_CAPACITY * 2,
            enable_batching: false,
            batch_size: 64,
        }
    }
}
