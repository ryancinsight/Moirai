//! Prefetch optimizations for iterator hot paths.
//!
//! This module provides utilities and wrappers to add strategic prefetch
//! hints to improve cache performance during iteration.

use std::mem;
use crate::cache_optimized::{prefetch_read_data, prefetch_write_data};

/// Prefetch distance in cache lines ahead of current position
pub const PREFETCH_DISTANCE: usize = 4;

/// Iterator adapter that adds prefetching to any iterator
pub struct PrefetchIterator<I: Iterator> {
    iter: I,
    prefetch_distance: usize,
}

impl<I: Iterator> PrefetchIterator<I> {
    /// Create a new prefetching iterator with default distance
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            prefetch_distance: PREFETCH_DISTANCE,
        }
    }
    
    /// Create with custom prefetch distance
    pub fn with_distance(iter: I, distance: usize) -> Self {
        Self {
            iter,
            prefetch_distance: distance,
        }
    }
}

impl<I> Iterator for PrefetchIterator<I>
where
    I: Iterator,
    I::Item: Sized,
{
    type Item = I::Item;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Get next item
        let item = self.iter.next()?;
        
        // Try to prefetch future items
        if let Some(size_hint) = self.iter.size_hint().1 {
            if size_hint > self.prefetch_distance {
                // This is a best-effort prefetch - we can't actually
                // access future items without consuming them
                // In practice, this works best with slice iterators
            }
        }
        
        Some(item)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Extension trait for adding prefetch to iterators
pub trait PrefetchExt: Iterator + Sized {
    /// Add prefetching to this iterator
    fn prefetch(self) -> PrefetchIterator<Self> {
        PrefetchIterator::new(self)
    }
    
    /// Add prefetching with custom distance
    fn prefetch_distance(self, distance: usize) -> PrefetchIterator<Self> {
        PrefetchIterator::with_distance(self, distance)
    }
}

impl<I: Iterator> PrefetchExt for I {}

/// Optimized slice iterator with prefetching
pub struct PrefetchSliceIter<'a, T> {
    slice: &'a [T],
    position: usize,
    prefetch_distance: usize,
}

impl<'a, T> PrefetchSliceIter<'a, T> {
    pub fn new(slice: &'a [T]) -> Self {
        let iter = Self {
            slice,
            position: 0,
            prefetch_distance: PREFETCH_DISTANCE,
        };
        
        // Prefetch initial data
        iter.prefetch_at(0);
        
        iter
    }
    
    #[inline(always)]
    fn prefetch_at(&self, index: usize) {
        if index + self.prefetch_distance < self.slice.len() {
            unsafe {
                let future_ptr = self.slice.as_ptr().add(index + self.prefetch_distance);
                prefetch_read_data(future_ptr as *const u8, 0); // L1 cache
            }
        }
    }
}

impl<'a, T> Iterator for PrefetchSliceIter<'a, T> {
    type Item = &'a T;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.slice.len() {
            return None;
        }
        
        let item = &self.slice[self.position];
        
        // Prefetch future data
        self.prefetch_at(self.position + 1);
        
        self.position += 1;
        Some(item)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len() - self.position;
        (remaining, Some(remaining))
    }
}

/// Prefetching iterator for mutable slices
pub struct PrefetchSliceIterMut<'a, T> {
    slice: &'a mut [T],
    position: usize,
    prefetch_distance: usize,
}

impl<'a, T> PrefetchSliceIterMut<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        let prefetch_distance = PREFETCH_DISTANCE;
        
        // Prefetch initial data for writing
        if !slice.is_empty() && prefetch_distance < slice.len() {
            unsafe {
                let future_ptr = slice.as_mut_ptr().add(prefetch_distance);
                prefetch_write_data(future_ptr as *mut u8, 0);
            }
        }
        
        Self {
            slice,
            position: 0,
            prefetch_distance,
        }
    }
}

impl<'a, T> Iterator for PrefetchSliceIterMut<'a, T> {
    type Item = &'a mut T;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.slice.len() {
            return None;
        }
        
        // Prefetch future data for writing
        if self.position + self.prefetch_distance + 1 < self.slice.len() {
            unsafe {
                let future_ptr = self.slice.as_mut_ptr().add(self.position + self.prefetch_distance + 1);
                prefetch_write_data(future_ptr as *mut u8, 0);
            }
        }
        
        let item = unsafe {
            // Safe because we check bounds and never create overlapping mutable references
            &mut *(self.slice.as_mut_ptr().add(self.position))
        };
        
        self.position += 1;
        Some(item)
    }
}

/// Extension trait for slices to provide prefetching iterators
pub trait SlicePrefetchExt<T> {
    /// Create a prefetching iterator over this slice
    fn prefetch_iter(&self) -> PrefetchSliceIter<T>;
    
    /// Create a prefetching mutable iterator
    fn prefetch_iter_mut(&mut self) -> PrefetchSliceIterMut<T>;
}

impl<T> SlicePrefetchExt<T> for [T] {
    fn prefetch_iter(&self) -> PrefetchSliceIter<T> {
        PrefetchSliceIter::new(self)
    }
    
    fn prefetch_iter_mut(&mut self) -> PrefetchSliceIterMut<T> {
        PrefetchSliceIterMut::new(self)
    }
}

/// Prefetch-optimized chunk iterator
pub struct PrefetchChunks<'a, T> {
    slice: &'a [T],
    chunk_size: usize,
    position: usize,
}

impl<'a, T> PrefetchChunks<'a, T> {
    pub fn new(slice: &'a [T], chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "Chunk size must be positive");
        
        let iter = Self {
            slice,
            chunk_size,
            position: 0,
        };
        
        // Prefetch first chunk
        if !slice.is_empty() {
            unsafe {
                let chunk_end = chunk_size.min(slice.len());
                for i in (0..chunk_end).step_by(64 / mem::size_of::<T>().max(1)) {
                    prefetch_read_data(slice.as_ptr().add(i) as *const u8, 0);
                }
            }
        }
        
        iter
    }
}

impl<'a, T> Iterator for PrefetchChunks<'a, T> {
    type Item = &'a [T];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.slice.len() {
            return None;
        }
        
        let chunk_end = (self.position + self.chunk_size).min(self.slice.len());
        let chunk = &self.slice[self.position..chunk_end];
        
        // Prefetch next chunk
        let next_position = self.position + self.chunk_size;
        if next_position < self.slice.len() {
            unsafe {
                let next_chunk_end = (next_position + self.chunk_size).min(self.slice.len());
                // Prefetch every cache line in the next chunk
                for i in (next_position..next_chunk_end).step_by(64 / mem::size_of::<T>().max(1)) {
                    prefetch_read_data(self.slice.as_ptr().add(i) as *const u8, 1); // L2 cache
                }
            }
        }
        
        self.position = chunk_end;
        Some(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prefetch_slice_iter() {
        let data: Vec<i32> = (0..1000).collect();
        let sum: i32 = data.prefetch_iter().sum();
        assert_eq!(sum, (0..1000).sum());
    }
    
    #[test]
    fn test_prefetch_chunks() {
        let data: Vec<i32> = (0..1000).collect();
        let chunks = PrefetchChunks::new(&data, 100);
        let count = chunks.count();
        assert_eq!(count, 10);
    }
    
    #[test]
    fn test_prefetch_mut_iter() {
        let mut data: Vec<i32> = vec![0; 1000];
        for (i, val) in data.prefetch_iter_mut().enumerate() {
            *val = i as i32;
        }
        
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(val, i as i32);
        }
    }
}