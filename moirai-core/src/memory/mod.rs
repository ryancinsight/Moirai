//! Advanced memory management for Moirai concurrency library.

mod allocator;
mod pool;

#[cfg(test)]
mod tests;

/// Cache line size for alignment optimizations.
///
/// Re-exported from `moirai-utils`, which owns the per-target table. See
/// [`moirai_utils::DESTRUCTIVE_INTERFERENCE_SIZE`] for the (larger) separation
/// that concurrently written data needs.
pub use moirai_utils::CACHE_LINE_SIZE;

pub use allocator::CacheAlignedAllocator;
pub use pool::MemoryPool;
