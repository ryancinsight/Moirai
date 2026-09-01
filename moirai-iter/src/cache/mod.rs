//! Cache-aware borrowed-slice iteration.
//!
//! Sequential windows and chunks preserve cache-sized traversal, while
//! [ZeroCopyParallelIter] uses joined scheduler fan-out without copying the
//! source slice.

mod parallel;
mod prefetch;
#[cfg(test)]
mod tests;
mod windows;

pub use parallel::ZeroCopyParallelIter;
pub use prefetch::{prefetch_read_data, prefetch_write_data};
pub use windows::{CacheAlignedChunks, WindowIterator};

/// Cache line size used to derive chunk widths.
///
/// Re-exported from moirai-utils, the single source for the per-target table.
/// Chunking uses transfer granularity rather than destructive-interference
/// width.
pub use moirai_utils::CACHE_LINE_SIZE;

/// Chunk size for cache-friendly iteration.
pub const CACHE_CHUNK_SIZE: usize = 16_384;

/// Extension methods for cache-aware borrowed-slice iteration.
pub trait CacheIterExt<T> {
    /// Iterate windows of window_size elements.
    fn cache_windows(&self, window_size: usize) -> WindowIterator<'_, T>;

    /// Iterate cache-sized chunks of this slice.
    fn cache_chunks(&self) -> CacheAlignedChunks<'_, T>;

    /// Create a zero-copy parallel iterator over this slice.
    fn zero_copy_par_iter(&self) -> ZeroCopyParallelIter<'_, T>;
}

impl<T: Send + Sync> CacheIterExt<T> for [T] {
    fn cache_windows(&self, window_size: usize) -> WindowIterator<'_, T> {
        WindowIterator::new(self, window_size, window_size)
    }

    fn cache_chunks(&self) -> CacheAlignedChunks<'_, T> {
        CacheAlignedChunks::new(self)
    }

    fn zero_copy_par_iter(&self) -> ZeroCopyParallelIter<'_, T> {
        ZeroCopyParallelIter::new(self)
    }
}
