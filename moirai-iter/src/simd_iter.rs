//! SIMD-optimized iterators for high-performance data processing.
//!
//! This module provides SIMD-accelerated iteration patterns that leverage
//! modern CPU vector instructions for maximum throughput.

use std::marker::PhantomData;
use std::sync::Arc;
use crate::base::SendPtr;

/// SIMD-optimized iterator for f32 operations.
///
/// Automatically vectorizes operations using platform-specific SIMD instructions.
pub struct SimdF32Iterator<'a> {
    data: &'a [f32],
    #[allow(dead_code)]
    chunk_size: usize,
}

impl<'a> SimdF32Iterator<'a> {
    pub fn new(data: &'a [f32]) -> Self {
        // AVX2 processes 8 f32 values at once
        #[cfg(target_arch = "x86_64")]
        let chunk_size = if is_x86_feature_detected!("avx2") { 8 } else { 1 };
        #[cfg(not(target_arch = "x86_64"))]
        let chunk_size = 1;
        
        Self {
            data,
            chunk_size,
        }
    }
    
    /// Apply vectorized addition with another slice
    pub fn simd_add(self, other: &'a [f32]) -> Vec<f32> {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");
        
        #[cfg(target_arch = "x86_64")]
        let use_simd = is_x86_feature_detected!("avx2") && self.data.len() >= 8;
        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = false;
        
        if !use_simd {
            // Fallback to scalar operations
            return self.data.iter()
                .zip(other.iter())
                .map(|(a, b)| a + b)
                .collect();
        }
        
        let mut result = vec![0.0f32; self.data.len()];
        
        // Process aligned chunks with SIMD
        let simd_len = (self.data.len() / 8) * 8;
        if simd_len > 0 {
            unsafe {
                moirai_utils::simd::vectorized_add_f32(
                    &self.data[..simd_len],
                    &other[..simd_len],
                    &mut result[..simd_len],
                );
            }
        }
        
        // Process remaining elements
        for i in simd_len..self.data.len() {
            result[i] = self.data[i] + other[i];
        }
        
        result
    }
    
    /// Apply vectorized multiplication with another slice
    pub fn simd_mul(self, other: &'a [f32]) -> Vec<f32> {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");
        
        #[cfg(target_arch = "x86_64")]
        let use_simd = is_x86_feature_detected!("avx2") && self.data.len() >= 8;
        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = false;
        
        if !use_simd {
            // Fallback to scalar operations
            return self.data.iter()
                .zip(other.iter())
                .map(|(a, b)| a * b)
                .collect();
        }
        
        let mut result = vec![0.0f32; self.data.len()];
        
        // Process aligned chunks with SIMD
        let simd_len = (self.data.len() / 8) * 8;
        if simd_len > 0 {
            unsafe {
                moirai_utils::simd::vectorized_mul_f32(
                    &self.data[..simd_len],
                    &other[..simd_len],
                    &mut result[..simd_len],
                );
            }
        }
        
        // Process remaining elements
        for i in simd_len..self.data.len() {
            result[i] = self.data[i] * other[i];
        }
        
        result
    }
    
    /// Compute dot product using SIMD
    pub fn simd_dot_product(self, other: &'a [f32]) -> f32 {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");
        
        #[cfg(target_arch = "x86_64")]
        let use_simd = is_x86_feature_detected!("avx2") && self.data.len() >= 8;
        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = false;
        
        if !use_simd {
            // Fallback to scalar operations
            return self.data.iter()
                .zip(other.iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        
        let simd_len = (self.data.len() / 8) * 8;
        let mut sum = 0.0f32;
        
        if simd_len > 0 {
            unsafe {
                sum = moirai_utils::simd::vectorized_dot_product_f32(
                    &self.data[..simd_len],
                    &other[..simd_len],
                );
            }
        }
        
        // Add remaining elements
        for i in simd_len..self.data.len() {
            sum += self.data[i] * other[i];
        }
        
        sum
    }
    
    /// Apply a scalar function with cache-friendly chunking and prefetching
    /// Note: The function itself is not vectorized, but the iteration is optimized
    pub fn map_with_prefetch<F>(self, func: F) -> Vec<f32>
    where
        F: Fn(f32) -> f32,
    {
        let mut result = Vec::with_capacity(self.data.len());
        
        // Process in cache-friendly chunks
        const CHUNK_SIZE: usize = 1024; // Fits in L1 cache
        
        for chunk in self.data.chunks(CHUNK_SIZE) {
            // Prefetch next chunk
            if let Some(next_chunk) = self.data.get(result.len() + CHUNK_SIZE..) {
                unsafe {
                    use crate::cache_optimized::prefetch_read_data;
                    prefetch_read_data(next_chunk.as_ptr() as *const u8, 1);
                }
            }
            
            // Process current chunk
            result.extend(chunk.iter().map(|&x| func(x)));
        }
        
        result
    }
}

/// SIMD-optimized parallel iterator combining SIMD and parallelism
pub struct SimdParallelIterator<'a, T> {
    data: &'a [T],
    chunk_size: usize,
    _phantom: PhantomData<T>,
}

impl<'a> SimdParallelIterator<'a, f32> {
    pub fn new(data: &'a [f32]) -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        
        // Each thread processes multiple SIMD chunks
        #[cfg(target_arch = "x86_64")]
        let simd_chunk = if is_x86_feature_detected!("avx2") { 8 } else { 1 };
        #[cfg(not(target_arch = "x86_64"))]
        let simd_chunk = 1;
        let chunk_size = (data.len() / num_threads).max(simd_chunk * 128); // At least 128 SIMD operations per thread
        
        Self {
            data,
            chunk_size,
            _phantom: PhantomData,
        }
    }
    
    /// Parallel SIMD addition
    pub fn par_simd_add(self, other: &'a [f32]) -> Vec<f32> {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");
        
        let mut result = vec![0.0f32; self.data.len()];
        let data = Arc::new(self.data.to_vec());
        let other = Arc::new(other.to_vec());
        let len = self.data.len();
        let chunk_size = self.chunk_size;
        
        std::thread::scope(|scope| {
            let result_ptr = result.as_mut_ptr();
            let num_chunks = (len + chunk_size - 1) / chunk_size;
            
            for chunk_idx in 0..num_chunks {
                let chunk_start = chunk_idx * chunk_size;
                let chunk_end = std::cmp::min(chunk_start + chunk_size, len);
                let chunk_len = chunk_end - chunk_start;
                
                // Clone Arc references
                let data = Arc::clone(&data);
                let other = Arc::clone(&other);
                let result_ptr_wrapper = SendPtr(unsafe { result_ptr.add(chunk_start) });
                
                scope.spawn(move || {
                    let chunk_a = &data[chunk_start..chunk_end];
                    let chunk_b = &other[chunk_start..chunk_end];
                    let chunk_result = SimdF32Iterator::new(chunk_a).simd_add(chunk_b);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            chunk_result.as_ptr(),
                            result_ptr_wrapper.as_ptr(),
                            chunk_len,
                        );
                    }
                });
            }
        });
        
        result
    }
    
    /// Parallel SIMD multiplication
    pub fn par_simd_mul(self, other: &'a [f32]) -> Vec<f32> {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");
        
        let mut result = vec![0.0f32; self.data.len()];
        let data = Arc::new(self.data.to_vec());
        let other = Arc::new(other.to_vec());
        let len = self.data.len();
        let chunk_size = self.chunk_size;
        
        std::thread::scope(|scope| {
            let result_ptr = result.as_mut_ptr();
            let num_chunks = (len + chunk_size - 1) / chunk_size;
            
            for chunk_idx in 0..num_chunks {
                let chunk_start = chunk_idx * chunk_size;
                let chunk_end = std::cmp::min(chunk_start + chunk_size, len);
                let chunk_len = chunk_end - chunk_start;
                
                // Clone Arc references
                let data = Arc::clone(&data);
                let other = Arc::clone(&other);
                let result_ptr_wrapper = SendPtr(unsafe { result_ptr.add(chunk_start) });
                
                scope.spawn(move || {
                    let chunk_a = &data[chunk_start..chunk_end];
                    let chunk_b = &other[chunk_start..chunk_end];
                    let chunk_result = SimdF32Iterator::new(chunk_a).simd_mul(chunk_b);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            chunk_result.as_ptr(),
                            result_ptr_wrapper.as_ptr(),
                            chunk_len,
                        );
                    }
                });
            }
        });
        
        result
    }
    
    /// Parallel SIMD dot product with tree reduction
    pub fn par_simd_dot_product(self, other: &'a [f32]) -> f32 {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");
        
        let partial_sums: Vec<f32> = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            
            for (chunk_a, chunk_b) in self.data.chunks(self.chunk_size)
                .zip(other.chunks(self.chunk_size))
            {
                let handle = scope.spawn(move || {
                    SimdF32Iterator::new(chunk_a).simd_dot_product(chunk_b)
                });
                handles.push(handle);
            }
            
            handles.into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        
        // Sum all partial results
        partial_sums.iter().sum()
    }
}

/// Extension trait for SIMD operations on slices
pub trait SimdIteratorExt {
    /// Create a SIMD-optimized iterator for f32 slices
    fn simd_iter(&self) -> SimdF32Iterator;
    
    /// Create a parallel SIMD iterator for f32 slices
    fn par_simd_iter(&self) -> SimdParallelIterator<f32>;
}

impl SimdIteratorExt for [f32] {
    fn simd_iter(&self) -> SimdF32Iterator {
        SimdF32Iterator::new(self)
    }
    
    fn par_simd_iter(&self) -> SimdParallelIterator<f32> {
        SimdParallelIterator::new(self)
    }
}

// Helper macro to check CPU features at compile time
#[cfg(target_arch = "x86_64")]
#[allow(unused_macros)]
macro_rules! is_x86_feature_detected {
    ("avx2") => {
        cfg!(target_feature = "avx2") || std::is_x86_feature_detected!("avx2")
    };
    ("sse2") => {
        cfg!(target_feature = "sse2") || std::is_x86_feature_detected!("sse2")
    };
}

#[cfg(not(target_arch = "x86_64"))]
macro_rules! is_x86_feature_detected {
    ($feature:literal) => { false };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_add() {
        let a = vec![1.0f32; 1024];
        let b = vec![2.0f32; 1024];
        
        let result = a.as_slice().simd_iter().simd_add(&b);
        
        assert_eq!(result.len(), 1024);
        for &val in &result {
            assert_eq!(val, 3.0);
        }
    }
    
    #[test]
    fn test_simd_mul() {
        let a = vec![2.0f32; 1024];
        let b = vec![3.0f32; 1024];
        
        let result = a.as_slice().simd_iter().simd_mul(&b);
        
        assert_eq!(result.len(), 1024);
        for &val in &result {
            assert_eq!(val, 6.0);
        }
    }
    
    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0f32; 1000];
        let b = vec![2.0f32; 1000];
        
        let result = a.as_slice().simd_iter().simd_dot_product(&b);
        
        assert_eq!(result, 2000.0);
    }
    
    #[test]
    fn test_parallel_simd() {
        let a = vec![1.0f32; 10000];
        let b = vec![2.0f32; 10000];
        
        let result = a.as_slice().par_simd_iter().par_simd_add(&b);
        
        assert_eq!(result.len(), 10000);
        for &val in &result {
            assert_eq!(val, 3.0);
        }
    }
}