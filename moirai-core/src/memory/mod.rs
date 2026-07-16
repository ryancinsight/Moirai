//! Advanced memory management for Moirai concurrency library.

mod allocator;
mod buffer;
mod pool;

#[cfg(test)]
mod tests;

/// Cache line size for alignment optimizations
pub const CACHE_LINE_SIZE: usize = 64;

pub use allocator::CacheAlignedAllocator;
pub use buffer::UnifiedRingBuffer;
pub use pool::MemoryPool;
