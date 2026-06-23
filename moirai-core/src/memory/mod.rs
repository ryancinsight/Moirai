//! Advanced memory management for Moirai concurrency library.

mod allocator;
mod buffer;
mod pool;

#[cfg(test)]
mod tests;

pub use allocator::CacheAlignedAllocator;
pub use buffer::UnifiedRingBuffer;
pub use pool::{GlobalMemoryManager, MemoryPool};
