//! Sequential cache-sized window and chunk iterators.

use std::mem;

use super::{prefetch_read_data, CACHE_CHUNK_SIZE};

/// Iterator over borrowed windows.
pub struct WindowIterator<'a, T> {
    data: &'a [T],
    window_size: usize,
    stride: usize,
    position: usize,
}

impl<'a, T> WindowIterator<'a, T> {
    /// Create windows with the specified size and stride.
    #[track_caller]
    pub fn new(data: &'a [T], window_size: usize, stride: usize) -> Self {
        assert!(window_size > 0, "window size must be positive");
        assert!(stride > 0, "window stride must be positive");
        Self {
            data,
            window_size,
            stride,
            position: 0,
        }
    }

    /// Create non-overlapping cache-sized windows.
    pub fn for_cache(data: &'a [T]) -> Self {
        let window = CACHE_CHUNK_SIZE / mem::size_of::<T>().max(1);
        Self::new(data, window, window)
    }
}

impl<'a, T> Iterator for WindowIterator<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }
        let end = self
            .position
            .saturating_add(self.window_size)
            .min(self.data.len());
        let window = &self.data[self.position..end];
        self.position = self.position.saturating_add(self.stride);
        Some(window)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.position >= self.data.len() {
            return (0, Some(0));
        }
        let windows = (self.data.len() - self.position).div_ceil(self.stride);
        (windows, Some(windows))
    }
}

/// Iterator over cache-sized borrowed chunks.
pub struct CacheAlignedChunks<'a, T> {
    data: &'a [T],
    chunk_size: usize,
    position: usize,
}

impl<'a, T> CacheAlignedChunks<'a, T> {
    /// Create cache-sized chunks over data.
    pub fn new(data: &'a [T]) -> Self {
        let chunk_size = (CACHE_CHUNK_SIZE / mem::size_of::<T>().max(1)).max(1);
        Self {
            data,
            chunk_size,
            position: 0,
        }
    }
}

impl<'a, T> Iterator for CacheAlignedChunks<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }
        let end = self
            .position
            .saturating_add(self.chunk_size)
            .min(self.data.len());
        let chunk = &self.data[self.position..end];
        if end < self.data.len() {
            // SAFETY: end is below data.len(), so this points at a live element.
            unsafe {
                prefetch_read_data(self.data.as_ptr().add(end).cast(), 3);
            }
        }
        self.position = end;
        Some(chunk)
    }
}
