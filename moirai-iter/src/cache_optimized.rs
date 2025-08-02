//! Cache-optimized iterator implementations for maximum memory throughput.
//!
//! This module provides iterators that are designed to work efficiently
//! with CPU caches, minimizing cache misses and maximizing memory bandwidth.


use std::mem;
use std::ptr;
use std::sync::Arc;

// Import CacheAligned from moirai-core
use moirai_core::cache_aligned::CacheAligned;

/// Wrapper to make raw pointers Send
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}

impl<T> SendPtr<T> {
    unsafe fn as_ptr(&self) -> *mut T {
        self.0
    }
}

/// Wrapper to make const raw pointers Send  
struct SendConstPtr<T>(*const T);
unsafe impl<T> Send for SendConstPtr<T> {}

impl<T> SendConstPtr<T> {
    unsafe fn as_ptr(&self) -> *const T {
        self.0
    }
}

/// Cache line size for most modern x86_64 processors
pub const CACHE_LINE_SIZE: usize = 64;

/// Optimal chunk size for cache-friendly iteration (L1 cache size / 2)
pub const OPTIMAL_CHUNK_SIZE: usize = 16384; // 16KB, half of typical L1 cache

/// Zero-copy window iterator that processes data in cache-friendly chunks
pub struct WindowIterator<'a, T> {
    data: &'a [T],
    window_size: usize,
    stride: usize,
    position: usize,
}

impl<'a, T> WindowIterator<'a, T> {
    /// Create a new window iterator with specified window size and stride
    pub fn new(data: &'a [T], window_size: usize, stride: usize) -> Self {
        assert!(window_size > 0, "Window size must be positive");
        assert!(stride > 0, "Stride must be positive");
        
        Self {
            data,
            window_size,
            stride,
            position: 0,
        }
    }
    
    /// Create a window iterator with optimal cache-friendly parameters
    pub fn cache_optimized(data: &'a [T]) -> Self {
        let element_size = mem::size_of::<T>();
        let optimal_window = OPTIMAL_CHUNK_SIZE / element_size.max(1);
        
        Self::new(data, optimal_window, optimal_window)
    }
}

impl<'a, T> Iterator for WindowIterator<'a, T> {
    type Item = &'a [T];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.data.len() {
            return None;
        }
        
        let end = (self.position + self.window_size).min(self.data.len());
        let window = &self.data[self.position..end];
        
        self.position += self.stride;
        
        Some(window)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.position >= self.data.len() {
            return (0, Some(0));
        }
        
        let remaining = self.data.len() - self.position;
        let windows = (remaining + self.stride - 1) / self.stride;
        
        (windows, Some(windows))
    }
}

/// Zero-copy chunked iterator that aligns chunks to cache boundaries
pub struct CacheAlignedChunks<'a, T> {
    data: &'a [T],
    chunk_size: usize,
    position: usize,
}

impl<'a, T> CacheAlignedChunks<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        let element_size = mem::size_of::<T>();
        let elements_per_cache_line = CACHE_LINE_SIZE / element_size.max(1);
        let chunk_size = elements_per_cache_line * (OPTIMAL_CHUNK_SIZE / CACHE_LINE_SIZE);
        
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
        
        let end = (self.position + self.chunk_size).min(self.data.len());
        let chunk = &self.data[self.position..end];
        
        // Prefetch next chunk for better cache performance
        if end < self.data.len() {
            unsafe {
                let next_ptr = self.data.as_ptr().add(end);
                prefetch_read_data(next_ptr as *const u8, 3); // Prefetch to L1 cache
            }
        }
        
        self.position = end;
        Some(chunk)
    }
}

/// Prefetch data for reading with specified cache level
/// Level 0 = L1, 1 = L2, 2 = L3, 3 = all levels
#[inline(always)]
pub unsafe fn prefetch_read_data(ptr: *const u8, level: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        match level {
            0 => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
            1 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
            2 => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
            _ => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        // ARM doesn't have direct prefetch intrinsics in stable Rust
        // Would need inline assembly or compiler intrinsics
        // For now, this is a no-op on ARM
        let _ = (ptr, level);
    }
}

/// Prefetch data for writing with specified cache level
#[inline(always)]
pub unsafe fn prefetch_write_data(ptr: *mut u8, level: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // x86_64 doesn't distinguish between read and write prefetch
        match level {
            0 => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
            1 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
            2 => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
            _ => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        // ARM doesn't have direct prefetch intrinsics in stable Rust
        // Would need inline assembly or compiler intrinsics
        // For now, this is a no-op on ARM
        let _ = (ptr, level);
    }
}

/// A zero-copy parallel iterator that processes data in-place without allocation
pub struct ZeroCopyParallelIter<'a, T> {
    data: &'a [T],
    chunk_size: usize,
}

impl<'a, T: Sync> ZeroCopyParallelIter<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        
        // Calculate optimal chunk size based on cache and thread count
        let element_size = mem::size_of::<T>();
        let elements_per_cache = OPTIMAL_CHUNK_SIZE / element_size.max(1);
        let chunk_size = (data.len() / num_threads).max(elements_per_cache);
        
        Self { data, chunk_size }
    }
    
    /// Process items in parallel with zero allocations
    pub fn for_each<F>(&self, func: F)
    where
        F: Fn(&T) + Send + Sync,
    {
        std::thread::scope(|scope| {
            for chunk in self.data.chunks(self.chunk_size) {
                scope.spawn(|| {
                    // Process chunk with prefetching
                    let cache_line_elements = CACHE_LINE_SIZE / mem::size_of::<T>();
                    for (i, item) in chunk.iter().enumerate() {
                        // Prefetch next cache line
                        if i % cache_line_elements == 0 
                            && i + cache_line_elements < chunk.len() {
                            unsafe {
                                let next_ptr = chunk.as_ptr().add(i + cache_line_elements);
                                prefetch_read_data(next_ptr as *const u8, 0);
                            }
                        }
                        
                        func(item);
                    }
                });
            }
        });
    }
    
    /// Map items in parallel with minimal allocations
    pub fn map<F, R>(&self, func: F) -> Vec<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Send,
    {
        let mut results = Vec::with_capacity(self.data.len());
        unsafe { results.set_len(self.data.len()); }
        
        let results_ptr: *mut R = results.as_mut_ptr();
        let func = Arc::new(func);
        let data = Arc::new(self.data);
        let data_len = self.data.len();
        
        std::thread::scope(|scope| {
            let chunk_size = self.chunk_size;
            let num_chunks = (data_len + chunk_size - 1) / chunk_size;
            
            for chunk_idx in 0..num_chunks {
                let chunk_start = chunk_idx * chunk_size;
                let chunk_end = std::cmp::min(chunk_start + chunk_size, data_len);
                
                // Clone Arc references for the closure
                let data = Arc::clone(&data);
                let func = Arc::clone(&func);
                let results_ptr_wrapper = SendPtr(unsafe { results_ptr.add(chunk_start) });
                
                scope.spawn(move || {
                    for i in chunk_start..chunk_end {
                        unsafe {
                            let result = func(&data[i]);
                            let result_ptr = results_ptr_wrapper.as_ptr().add(i - chunk_start);
                            ptr::write(result_ptr, result);
                        }
                    }
                });
            }
        });
        
        results
    }
    
    /// Reduce items in parallel using tree reduction for optimal cache efficiency
    pub fn reduce<F>(&self, func: F) -> Option<T>
    where
        F: Fn(&T, &T) -> T + Send + Sync + Clone,
        T: Clone + Send,
    {
        if self.data.is_empty() {
            return None;
        }
        
        if self.data.len() == 1 {
            return Some(self.data[0].clone());
        }
        
        // Tree reduction with cache-friendly chunking
        let mut current_results: Vec<T> = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            
            for chunk in self.data.chunks(self.chunk_size) {
                let handle = scope.spawn(|| {
                    chunk.iter()
                        .cloned()
                        .reduce(|a, b| func(&a, &b))
                });
                handles.push(handle);
            }
            
            handles.into_iter()
                .filter_map(|h| h.join().ok().flatten())
                .collect()
        });
        
        // Continue reducing until we have a single result
        while current_results.len() > 1 {
            current_results = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                let len = current_results.len();
                let pairs_per_chunk = 32; // Process multiple pairs per thread
                
                // Process pairs in chunks for better efficiency
                for chunk_start in (0..len).step_by(pairs_per_chunk * 2) {
                    let chunk_end = std::cmp::min(chunk_start + pairs_per_chunk * 2, len);
                    let chunk = current_results[chunk_start..chunk_end].to_vec();
                    let func = func.clone();
                    
                    let handle = scope.spawn(move || {
                        let mut results = Vec::new();
                        let mut i = 0;
                        while i < chunk.len() {
                            if i + 1 < chunk.len() {
                                results.push(func(&chunk[i], &chunk[i + 1]));
                                i += 2;
                            } else {
                                results.push(chunk[i].clone());
                                i += 1;
                            }
                        }
                        results
                    });
                    handles.push(handle);
                }
                
                handles.into_iter()
                    .flat_map(|h| h.join().ok().unwrap_or_default())
                    .collect()
            });
        }
        
        current_results.into_iter().next()
    }
}

/// Extension trait for slices to provide cache-optimized iteration
pub trait CacheOptimizedExt<T> {
    /// Create a cache-optimized window iterator
    fn cache_windows(&self, window_size: usize) -> WindowIterator<T>;
    
    /// Create a cache-aligned chunk iterator
    fn cache_chunks(&self) -> CacheAlignedChunks<T>;
    
    /// Create a zero-copy parallel iterator
    fn zero_copy_par_iter(&self) -> ZeroCopyParallelIter<T>;
}

impl<T: Send + Sync> CacheOptimizedExt<T> for [T] {
    fn cache_windows(&self, window_size: usize) -> WindowIterator<T> {
        WindowIterator::new(self, window_size, window_size)
    }
    
    fn cache_chunks(&self) -> CacheAlignedChunks<T> {
        CacheAlignedChunks::new(self)
    }
    
    fn zero_copy_par_iter(&self) -> ZeroCopyParallelIter<T> {
        ZeroCopyParallelIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_window_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let windows: Vec<_> = WindowIterator::new(&data, 3, 2).collect();
        
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0], &[1, 2, 3]);
        assert_eq!(windows[1], &[3, 4, 5]);
        assert_eq!(windows[2], &[5, 6, 7]);
        assert_eq!(windows[3], &[7, 8]);
    }
    
    #[test]
    fn test_cache_aligned_chunks() {
        let data: Vec<i32> = (0..1000).collect();
        let chunks: Vec<_> = data.cache_chunks().collect();
        
        assert!(!chunks.is_empty());
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 1000);
    }
    
    #[test]
    fn test_zero_copy_parallel() {
        let data: Vec<i32> = (0..10000).collect();
        let sum = std::sync::atomic::AtomicI64::new(0);
        
        data.zero_copy_par_iter().for_each(|&x| {
            sum.fetch_add(x as i64, std::sync::atomic::Ordering::Relaxed);
        });
        
        let expected_sum: i64 = (0..10000).sum();
        assert_eq!(sum.load(std::sync::atomic::Ordering::Relaxed), expected_sum);
    }
}