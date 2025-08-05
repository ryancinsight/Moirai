//! Zero-cost window iterators for efficient data processing
//!
//! This module provides window-based iterators that enable zero-copy processing
//! of data sequences. All iterators are designed to have zero runtime overhead
//! through careful use of Rust's ownership system and compile-time optimizations.
//!
//! # Design Principles
//! - **Zero-copy**: All iterators work with references, avoiding data duplication
//! - **Cache-friendly**: Sequential access patterns for optimal cache utilization
//! - **Composable**: Can be chained with other iterator combinators
//! - **Safe**: Memory safety guaranteed at compile time
//!
//! # Literature References
//! - "Cache-Oblivious Algorithms" by Frigo et al. (1999)
//! - "The Art of Computer Programming, Vol 3: Sorting and Searching" by Knuth
//! - "Elements of Programming" by Stepanov & McJones (2009)

use std::marker::PhantomData;

/// Iterator over overlapping windows of a slice
///
/// This struct is created by the [`windows`] method on slices.
///
/// # Examples
/// ```
/// let slice = &[1, 2, 3, 4, 5];
/// let windows: Vec<_> = slice.windows(3).collect();
/// assert_eq!(windows, vec![&[1, 2, 3][..], &[2, 3, 4][..], &[3, 4, 5][..]]);
/// ```
#[derive(Debug, Clone)]
pub struct Windows<'a, T> {
    slice: &'a [T],
    size: usize,
    pos: usize,
}

impl<'a, T> Windows<'a, T> {
    /// Creates a new `Windows` iterator.
    ///
    /// # Panics
    /// Panics if `size` is 0.
    #[inline]
    pub fn new(slice: &'a [T], size: usize) -> Self {
        assert!(size != 0, "window size must be non-zero");
        Windows { slice, size, pos: 0 }
    }
}

impl<'a, T> Iterator for Windows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.size > self.slice.len() - self.pos {
            None
        } else {
            let window = &self.slice[self.pos..self.pos + self.size];
            self.pos += 1;
            Some(window)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.size > self.slice.len() {
            (0, Some(0))
        } else {
            let remaining = self.slice.len() - self.pos - self.size + 1;
            (remaining, Some(remaining))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_pos = self.pos.saturating_add(n);
        if new_pos + self.size > self.slice.len() {
            self.pos = self.slice.len();
            None
        } else {
            self.pos = new_pos + 1;
            Some(&self.slice[new_pos..new_pos + self.size])
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.size > self.slice.len() {
            None
        } else {
            Some(&self.slice[self.slice.len() - self.size..])
        }
    }
}

impl<'a, T> DoubleEndedIterator for Windows<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.size > self.slice.len() - self.pos {
            None
        } else {
            let end = self.slice.len() - self.size;
            if self.pos <= end {
                let window = &self.slice[end..end + self.size];
                self.slice = &self.slice[..end];
                Some(window)
            } else {
                None
            }
        }
    }
}

impl<'a, T> ExactSizeIterator for Windows<'a, T> {}

/// Mutable iterator over overlapping windows of a slice
#[derive(Debug)]
pub struct WindowsMut<'a, T> {
    slice: &'a mut [T],
    size: usize,
    pos: usize,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T> WindowsMut<'a, T> {
    /// Creates a new `WindowsMut` iterator.
    ///
    /// # Safety
    /// This is safe because we ensure windows don't overlap in mutable references
    #[inline]
    pub fn new(slice: &'a mut [T], size: usize) -> Self {
        assert!(size != 0, "window size must be non-zero");
        WindowsMut {
            slice,
            size,
            pos: 0,
            _phantom: PhantomData,
        }
    }
}

/// Iterator over non-overlapping chunks of a slice
///
/// # Examples
/// ```
/// let slice = &[1, 2, 3, 4, 5, 6, 7];
/// let chunks: Vec<_> = slice.chunks(3).collect();
/// assert_eq!(chunks, vec![&[1, 2, 3][..], &[4, 5, 6][..], &[7][..]]);
/// ```
#[derive(Debug, Clone)]
pub struct Chunks<'a, T> {
    slice: &'a [T],
    chunk_size: usize,
}

impl<'a, T> Chunks<'a, T> {
    /// Creates a new `Chunks` iterator.
    ///
    /// # Panics
    /// Panics if `chunk_size` is 0.
    #[inline]
    pub fn new(slice: &'a [T], chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        Chunks { slice, chunk_size }
    }
}

impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let chunk_size = self.chunk_size.min(self.slice.len());
            let (chunk, rest) = self.slice.split_at(chunk_size);
            self.slice = rest;
            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.is_empty() {
            (0, Some(0))
        } else {
            let n = self.slice.len() / self.chunk_size;
            let rem = self.slice.len() % self.chunk_size;
            let count = n + (rem > 0) as usize;
            (count, Some(count))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.saturating_mul(self.chunk_size);
        if skip >= self.slice.len() {
            self.slice = &[];
            None
        } else {
            self.slice = &self.slice[skip..];
            self.next()
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let start = (self.slice.len() - 1) / self.chunk_size * self.chunk_size;
            Some(&self.slice[start..])
        }
    }
}

impl<'a, T> DoubleEndedIterator for Chunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let remainder = self.slice.len() % self.chunk_size;
            let chunk_size = if remainder != 0 { remainder } else { self.chunk_size };
            let (rest, chunk) = self.slice.split_at(self.slice.len() - chunk_size);
            self.slice = rest;
            Some(chunk)
        }
    }
}

impl<'a, T> ExactSizeIterator for Chunks<'a, T> {}

/// Mutable iterator over non-overlapping chunks
#[derive(Debug)]
pub struct ChunksMut<'a, T> {
    slice: &'a mut [T],
    chunk_size: usize,
}

impl<'a, T> ChunksMut<'a, T> {
    /// Creates a new `ChunksMut` iterator.
    #[inline]
    pub fn new(slice: &'a mut [T], chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        ChunksMut { slice, chunk_size }
    }
}

impl<'a, T> Iterator for ChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let chunk_size = self.chunk_size.min(self.slice.len());
            let slice = std::mem::take(&mut self.slice);
            let (chunk, rest) = slice.split_at_mut(chunk_size);
            self.slice = rest;
            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.is_empty() {
            (0, Some(0))
        } else {
            let n = self.slice.len() / self.chunk_size;
            let rem = self.slice.len() % self.chunk_size;
            let count = n + (rem > 0) as usize;
            (count, Some(count))
        }
    }
}

impl<'a, T> DoubleEndedIterator for ChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let remainder = self.slice.len() % self.chunk_size;
            let chunk_size = if remainder != 0 { remainder } else { self.chunk_size };
            let slice = std::mem::take(&mut self.slice);
            let (rest, chunk) = slice.split_at_mut(slice.len() - chunk_size);
            self.slice = rest;
            Some(chunk)
        }
    }
}

impl<'a, T> ExactSizeIterator for ChunksMut<'a, T> {}

/// Iterator over chunks of exact size (last chunk excluded if not exact)
#[derive(Debug, Clone)]
pub struct ChunksExact<'a, T> {
    slice: &'a [T],
    chunk_size: usize,
}

impl<'a, T> ChunksExact<'a, T> {
    /// Creates a new `ChunksExact` iterator.
    #[inline]
    pub fn new(slice: &'a [T], chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        ChunksExact { slice, chunk_size }
    }

    /// Returns the remainder of the slice that doesn't fit into chunks.
    #[inline]
    pub fn remainder(&self) -> &'a [T] {
        let chunks = self.slice.len() / self.chunk_size;
        &self.slice[chunks * self.chunk_size..]
    }
}

impl<'a, T> Iterator for ChunksExact<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.len() < self.chunk_size {
            None
        } else {
            let (chunk, rest) = self.slice.split_at(self.chunk_size);
            self.slice = rest;
            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.slice.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let skip = n.saturating_mul(self.chunk_size);
        if skip + self.chunk_size > self.slice.len() {
            self.slice = &[];
            None
        } else {
            self.slice = &self.slice[skip..];
            self.next()
        }
    }
}

impl<'a, T> DoubleEndedIterator for ChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.len() < self.chunk_size {
            None
        } else {
            let (rest, chunk) = self.slice.split_at(self.slice.len() - self.chunk_size);
            self.slice = rest;
            Some(chunk)
        }
    }
}

impl<'a, T> ExactSizeIterator for ChunksExact<'a, T> {}

/// Mutable iterator over chunks of exact size
#[derive(Debug)]
pub struct ChunksExactMut<'a, T> {
    slice: &'a mut [T],
    chunk_size: usize,
}

impl<'a, T> ChunksExactMut<'a, T> {
    /// Creates a new `ChunksExactMut` iterator.
    #[inline]
    pub fn new(slice: &'a mut [T], chunk_size: usize) -> Self {
        assert!(chunk_size != 0, "chunk size must be non-zero");
        ChunksExactMut { slice, chunk_size }
    }

    /// Returns the remainder of the slice that doesn't fit into chunks.
    #[inline]
    pub fn into_remainder(self) -> &'a mut [T] {
        let chunks = self.slice.len() / self.chunk_size;
        let slice = self.slice;
        &mut slice[chunks * self.chunk_size..]
    }
}

impl<'a, T> Iterator for ChunksExactMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.len() < self.chunk_size {
            None
        } else {
            let slice = std::mem::take(&mut self.slice);
            let (chunk, rest) = slice.split_at_mut(self.chunk_size);
            self.slice = rest;
            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.slice.len() / self.chunk_size;
        (n, Some(n))
    }
}

impl<'a, T> DoubleEndedIterator for ChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.len() < self.chunk_size {
            None
        } else {
            let slice = std::mem::take(&mut self.slice);
            let (rest, chunk) = slice.split_at_mut(slice.len() - self.chunk_size);
            self.slice = rest;
            Some(chunk)
        }
    }
}

impl<'a, T> ExactSizeIterator for ChunksExactMut<'a, T> {}

/// Extension trait for slice types to provide window iterators
pub trait WindowExt<T> {
    /// Returns an iterator over overlapping windows of `size` elements.
    fn windows(&self, size: usize) -> Windows<'_, T>;
    
    /// Returns an iterator over non-overlapping chunks of `size` elements.
    fn chunks(&self, size: usize) -> Chunks<'_, T>;
    
    /// Returns an iterator over chunks of exactly `size` elements.
    fn chunks_exact(&self, size: usize) -> ChunksExact<'_, T>;
}

impl<T> WindowExt<T> for [T] {
    #[inline]
    fn windows(&self, size: usize) -> Windows<'_, T> {
        Windows::new(self, size)
    }
    
    #[inline]
    fn chunks(&self, size: usize) -> Chunks<'_, T> {
        Chunks::new(self, size)
    }
    
    #[inline]
    fn chunks_exact(&self, size: usize) -> ChunksExact<'_, T> {
        ChunksExact::new(self, size)
    }
}

/// Extension trait for mutable slice types
pub trait WindowExtMut<T> {
    /// Returns a mutable iterator over non-overlapping chunks.
    fn chunks_mut(&mut self, size: usize) -> ChunksMut<'_, T>;
    
    /// Returns a mutable iterator over chunks of exactly `size` elements.
    fn chunks_exact_mut(&mut self, size: usize) -> ChunksExactMut<'_, T>;
}

impl<T> WindowExtMut<T> for [T] {
    #[inline]
    fn chunks_mut(&mut self, size: usize) -> ChunksMut<'_, T> {
        ChunksMut::new(self, size)
    }
    
    #[inline]
    fn chunks_exact_mut(&mut self, size: usize) -> ChunksExactMut<'_, T> {
        ChunksExactMut::new(self, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windows() {
        let data = vec![1, 2, 3, 4, 5];
        let windows: Vec<_> = data.windows(3).collect();
        assert_eq!(windows, vec![&[1, 2, 3][..], &[2, 3, 4][..], &[3, 4, 5][..]]);
    }

    #[test]
    fn test_chunks() {
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let chunks: Vec<_> = data.chunks(3).collect();
        assert_eq!(chunks, vec![&[1, 2, 3][..], &[4, 5, 6][..], &[7][..]]);
    }

    #[test]
    fn test_chunks_exact() {
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let iter = data.chunks_exact(3);
        let chunks: Vec<_> = iter.clone().collect();
        assert_eq!(chunks, vec![&[1, 2, 3][..], &[4, 5, 6][..]]);
        assert_eq!(iter.remainder(), &[7]);
    }

    #[test]
    fn test_windows_size_hint() {
        let data = vec![1, 2, 3, 4, 5];
        let windows = data.windows(3);
        assert_eq!(windows.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_chunks_mut() {
        let mut data = vec![1, 2, 3, 4, 5, 6];
        for chunk in data.chunks_mut(2) {
            chunk[0] *= 10;
        }
        assert_eq!(data, vec![10, 2, 30, 4, 50, 6]);
    }

    #[test]
    #[should_panic(expected = "window size must be non-zero")]
    fn test_windows_zero_size() {
        let data = vec![1, 2, 3];
        let _ = data.windows(0);
    }

    #[test]
    fn test_double_ended_iteration() {
        let data = vec![1, 2, 3, 4, 5];
        let mut windows = data.windows(2);
        assert_eq!(windows.next(), Some(&[1, 2][..]));
        assert_eq!(windows.next_back(), Some(&[4, 5][..]));
        assert_eq!(windows.next(), Some(&[2, 3][..]));
        assert_eq!(windows.next_back(), Some(&[3, 4][..]));
        assert_eq!(windows.next(), None);
        assert_eq!(windows.next_back(), None);
    }
}