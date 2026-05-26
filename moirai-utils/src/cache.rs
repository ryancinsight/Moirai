//! Cache alignment utilities for performance optimization.
//!
//! This module provides utilities for aligning data structures to cache line
//! boundaries to avoid false sharing and improve cache performance.

/// Cache line size for alignment optimizations.
pub const CACHE_LINE_SIZE: usize = 64;

/// Align a value to the cache line boundary.
#[must_use]
pub const fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// A cache-aligned wrapper for data structures.
#[repr(align(64))]
pub struct CacheAligned<T>(pub T);

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Get a reference to the inner value.
    pub const fn get(&self) -> &T {
        &self.0
    }

    /// Get a mutable reference to the inner value.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> core::ops::Deref for CacheAligned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for CacheAligned<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CacheAligned")
            .field("value", &self.0)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(align_to_cache_line(1), 64);
        assert_eq!(align_to_cache_line(64), 64);
        assert_eq!(align_to_cache_line(65), 128);
    }

    #[test]
    fn test_cache_aligned_wrapper() {
        let aligned = CacheAligned::new(42);
        assert_eq!(*aligned, 42);
        assert_eq!(aligned.get(), &42);
    }
}
