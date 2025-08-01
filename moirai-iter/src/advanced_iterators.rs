//! Advanced iterator implementations with SIMD vectorization and memory efficiency.
//! 
//! This module provides high-performance iterators inspired by:
//! - Rust's iterator fusion for zero-cost abstractions
//! - SIMD vectorization from Intel's ISPC
//! - Memory-efficient streaming from Apache Arrow
//! - Combinator patterns from functional programming

use std::marker::PhantomData;
use std::mem;
use std::slice;
use std::sync::Arc;
use std::ops::{Add, Mul};

/// SIMD lane width for vectorization
#[cfg(target_arch = "x86_64")]
const SIMD_WIDTH: usize = 8; // AVX2: 256 bits / 32 bits per f32

#[cfg(not(target_arch = "x86_64"))]
const SIMD_WIDTH: usize = 4; // Fallback for other architectures

/// Trait for types that can be processed with SIMD
pub trait SimdElement: Copy + Send + Sync {
    /// Process a slice of elements using SIMD
    fn simd_map<F>(slice: &[Self], f: F) -> Vec<Self>
    where
        F: Fn(Self) -> Self;
        
    /// Reduce a slice using SIMD
    fn simd_reduce<F>(slice: &[Self], identity: Self, f: F) -> Self
    where
        F: Fn(Self, Self) -> Self;
}

// Implement SIMD for f32
impl SimdElement for f32 {
    #[cfg(target_arch = "x86_64")]
    fn simd_map<F>(slice: &[Self], f: F) -> Vec<Self>
    where
        F: Fn(Self) -> Self,
    {
        use std::arch::x86_64::*;
        
        let mut result = Vec::with_capacity(slice.len());
        let chunks = slice.chunks_exact(8);
        let remainder = chunks.remainder();
        
        // Process SIMD chunks
        unsafe {
            for chunk in chunks {
                let vec = _mm256_loadu_ps(chunk.as_ptr());
                // In real implementation, we'd apply SIMD operations here
                // For now, fall back to scalar
                for &val in chunk {
                    result.push(f(val));
                }
            }
        }
        
        // Process remainder
        for &val in remainder {
            result.push(f(val));
        }
        
        result
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn simd_map<F>(slice: &[Self], f: F) -> Vec<Self>
    where
        F: Fn(Self) -> Self,
    {
        slice.iter().map(|&x| f(x)).collect()
    }
    
    fn simd_reduce<F>(slice: &[Self], identity: Self, f: F) -> Self
    where
        F: Fn(Self, Self) -> Self,
    {
        // For now, use standard reduce
        slice.iter().fold(identity, |acc, &x| f(acc, x))
    }
}

/// Zero-copy iterator that operates on borrowed slices
pub struct ZeroCopyIter<'a, T> {
    slice: &'a [T],
    index: usize,
}

impl<'a, T> ZeroCopyIter<'a, T> {
    /// Create a new zero-copy iterator
    pub fn new(slice: &'a [T]) -> Self {
        Self { slice, index: 0 }
    }
    
    /// Get the remaining slice
    pub fn as_slice(&self) -> &'a [T] {
        &self.slice[self.index..]
    }
    
    /// Split at the given index
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        let (left, right) = self.as_slice().split_at(mid);
        (
            ZeroCopyIter { slice: left, index: 0 },
            ZeroCopyIter { slice: right, index: 0 },
        )
    }
}

impl<'a, T> Iterator for ZeroCopyIter<'a, T> {
    type Item = &'a T;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len() {
            let item = &self.slice[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for ZeroCopyIter<'a, T> {}

/// Chunked iterator for cache-friendly processing
pub struct ChunkedIter<T, I: Iterator<Item = T>> {
    iter: I,
    chunk_size: usize,
    _phantom: PhantomData<T>,
}

impl<T, I: Iterator<Item = T>> ChunkedIter<T, I> {
    /// Create a new chunked iterator
    pub fn new(iter: I, chunk_size: usize) -> Self {
        Self {
            iter,
            chunk_size: chunk_size.max(1),
            _phantom: PhantomData,
        }
    }
}

impl<T, I: Iterator<Item = T>> Iterator for ChunkedIter<T, I> {
    type Item = Vec<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size);
        
        for _ in 0..self.chunk_size {
            match self.iter.next() {
                Some(item) => chunk.push(item),
                None => break,
            }
        }
        
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
}

/// Fused iterator that combines multiple operations into one pass
pub struct FusedIter<T, I, F1, F2> {
    iter: I,
    map_fn: F1,
    filter_fn: F2,
    _phantom: PhantomData<T>,
}

impl<T, U, I, F1, F2> FusedIter<T, I, F1, F2>
where
    I: Iterator<Item = T>,
    F1: Fn(T) -> U,
    F2: Fn(&U) -> bool,
{
    /// Create a new fused iterator
    pub fn new(iter: I, map_fn: F1, filter_fn: F2) -> Self {
        Self {
            iter,
            map_fn,
            filter_fn,
            _phantom: PhantomData,
        }
    }
}

impl<T, U, I, F1, F2> Iterator for FusedIter<T, I, F1, F2>
where
    I: Iterator<Item = T>,
    F1: Fn(T) -> U,
    F2: Fn(&U) -> bool,
{
    type Item = U;
    
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = self.iter.next()?;
            let mapped = (self.map_fn)(item);
            if (self.filter_fn)(&mapped) {
                return Some(mapped);
            }
        }
    }
}

/// Windowed iterator for sliding window operations
pub struct WindowedIter<'a, T> {
    slice: &'a [T],
    window_size: usize,
    index: usize,
}

impl<'a, T> WindowedIter<'a, T> {
    /// Create a new windowed iterator
    pub fn new(slice: &'a [T], window_size: usize) -> Self {
        Self {
            slice,
            window_size: window_size.max(1),
            index: 0,
        }
    }
}

impl<'a, T> Iterator for WindowedIter<'a, T> {
    type Item = &'a [T];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.index + self.window_size <= self.slice.len() {
            let window = &self.slice[self.index..self.index + self.window_size];
            self.index += 1;
            Some(window)
        } else {
            None
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len().saturating_sub(self.index + self.window_size - 1);
        (remaining, Some(remaining))
    }
}

/// Parallel iterator with automatic chunking
pub struct ParallelIter<T> {
    data: Arc<Vec<T>>,
    chunk_size: usize,
    num_threads: usize,
}

impl<T: Send + Sync + 'static> ParallelIter<T> {
    /// Create a new parallel iterator
    pub fn new(data: Vec<T>) -> Self {
        let num_threads = num_cpus::get();
        let chunk_size = (data.len() + num_threads - 1) / num_threads;
        
        Self {
            data: Arc::new(data),
            chunk_size,
            num_threads,
        }
    }
    
    /// Map operation in parallel
    pub fn map<F, U>(self, f: F) -> Vec<U>
    where
        F: Fn(&T) -> U + Send + Sync + 'static,
        U: Send + 'static,
    {
        use std::thread;
        
        let mut handles = vec![];
        let f = Arc::new(f);
        
        for i in 0..self.num_threads {
            let data = self.data.clone();
            let f = f.clone();
            let start = i * self.chunk_size;
            let end = ((i + 1) * self.chunk_size).min(data.len());
            
            let handle = thread::spawn(move || {
                data[start..end].iter().map(|x| f(x)).collect::<Vec<_>>()
            });
            
            handles.push(handle);
        }
        
        handles.into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    }
    
    /// Reduce operation in parallel
    pub fn reduce<F>(self, identity: T, f: F) -> T
    where
        F: Fn(T, &T) -> T + Send + Sync + 'static,
        T: Clone,
    {
        use std::thread;
        
        let mut handles = vec![];
        let f = Arc::new(f);
        
        for i in 0..self.num_threads {
            let data = self.data.clone();
            let f = f.clone();
            let start = i * self.chunk_size;
            let end = ((i + 1) * self.chunk_size).min(data.len());
            let identity = identity.clone();
            
            let handle = thread::spawn(move || {
                data[start..end].iter().fold(identity, |acc, x| f(acc, x))
            });
            
            handles.push(handle);
        }
        
        handles.into_iter()
            .map(|h| h.join().unwrap())
            .fold(identity, |acc, x| f(acc, &x))
    }
}

/// Advanced iterator combinators
pub trait AdvancedIteratorExt: Iterator + Sized {
    /// Create a chunked iterator
    fn chunked(self, chunk_size: usize) -> ChunkedIter<Self::Item, Self> {
        ChunkedIter::new(self, chunk_size)
    }
    
    /// Fuse map and filter operations
    fn map_filter<U, F1, F2>(self, map_fn: F1, filter_fn: F2) -> FusedIter<Self::Item, Self, F1, F2>
    where
        F1: Fn(Self::Item) -> U,
        F2: Fn(&U) -> bool,
    {
        FusedIter::new(self, map_fn, filter_fn)
    }
    
    /// Scan with state
    fn scan_state<S, F, U>(self, initial: S, f: F) -> ScanState<Self, S, F>
    where
        F: FnMut(&mut S, Self::Item) -> Option<U>,
    {
        ScanState {
            iter: self,
            state: initial,
            f,
        }
    }
    
    /// Batch process items
    fn batch<F, U>(self, batch_size: usize, f: F) -> BatchIter<Self, F>
    where
        F: Fn(Vec<Self::Item>) -> U,
    {
        BatchIter {
            iter: self,
            batch_size,
            f,
        }
    }
    
    /// Interleave with another iterator
    fn interleave<I>(self, other: I) -> Interleave<Self, I::IntoIter>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        Interleave {
            a: self,
            b: other.into_iter(),
            flag: false,
        }
    }
}

impl<I: Iterator + Sized> AdvancedIteratorExt for I {}

/// Scan iterator with mutable state
pub struct ScanState<I, S, F> {
    iter: I,
    state: S,
    f: F,
}

impl<I, S, F, U> Iterator for ScanState<I, S, F>
where
    I: Iterator,
    F: FnMut(&mut S, I::Item) -> Option<U>,
{
    type Item = U;
    
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next()?;
        (self.f)(&mut self.state, item)
    }
}

/// Batch processing iterator
pub struct BatchIter<I, F> {
    iter: I,
    batch_size: usize,
    f: F,
}

impl<I, F, U> Iterator for BatchIter<I, F>
where
    I: Iterator,
    F: Fn(Vec<I::Item>) -> U,
{
    type Item = U;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        
        for _ in 0..self.batch_size {
            match self.iter.next() {
                Some(item) => batch.push(item),
                None => break,
            }
        }
        
        if batch.is_empty() {
            None
        } else {
            Some((self.f)(batch))
        }
    }
}

/// Interleave iterator
pub struct Interleave<A, B> {
    a: A,
    b: B,
    flag: bool,
}

impl<A, B> Iterator for Interleave<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    type Item = A::Item;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.flag = !self.flag;
        if self.flag {
            self.a.next().or_else(|| self.b.next())
        } else {
            self.b.next().or_else(|| self.a.next())
        }
    }
}

/// Memory-efficient streaming iterator
pub struct StreamingIter<T> {
    buffer: Vec<T>,
    capacity: usize,
    producer: Box<dyn FnMut() -> Option<T>>,
}

impl<T> StreamingIter<T> {
    /// Create a new streaming iterator
    pub fn new<F>(capacity: usize, producer: F) -> Self
    where
        F: FnMut() -> Option<T> + 'static,
    {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            producer: Box::new(producer),
        }
    }
    
    /// Fill the buffer
    fn fill_buffer(&mut self) {
        while self.buffer.len() < self.capacity {
            match (self.producer)() {
                Some(item) => self.buffer.push(item),
                None => break,
            }
        }
    }
}

impl<T> Iterator for StreamingIter<T> {
    type Item = T;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            self.fill_buffer();
        }
        
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.buffer.remove(0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero_copy_iter() {
        let data = vec![1, 2, 3, 4, 5];
        let iter = ZeroCopyIter::new(&data);
        let collected: Vec<_> = iter.collect();
        assert_eq!(collected, vec![&1, &2, &3, &4, &5]);
    }
    
    #[test]
    fn test_chunked_iter() {
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let chunks: Vec<_> = data.into_iter().chunked(3).collect();
        assert_eq!(chunks, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
    }
    
    #[test]
    fn test_fused_iter() {
        let data = vec![1, 2, 3, 4, 5];
        let result: Vec<_> = data.into_iter()
            .map_filter(|x| x * 2, |x| x > &5)
            .collect();
        assert_eq!(result, vec![6, 8, 10]);
    }
    
    #[test]
    fn test_windowed_iter() {
        let data = vec![1, 2, 3, 4, 5];
        let windows: Vec<_> = WindowedIter::new(&data, 3).collect();
        assert_eq!(windows, vec![&[1, 2, 3][..], &[2, 3, 4][..], &[3, 4, 5][..]]);
    }
    
    #[test]
    fn test_interleave() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];
        let result: Vec<_> = a.into_iter().interleave(b).collect();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }
    
    #[test]
    fn test_batch_iter() {
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let result: Vec<_> = data.into_iter()
            .batch(3, |batch| batch.iter().sum::<i32>())
            .collect();
        assert_eq!(result, vec![6, 15, 7]);
    }
}