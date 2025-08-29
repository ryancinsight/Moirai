//! SIMD-optimized iterators for high-performance data processing.
//!
//! This module provides SIMD-accelerated iteration patterns that leverage
//! modern CPU vector instructions for maximum throughput, based on techniques
//! from "Computer Systems: A Programmer's Perspective" and Intel optimization guides.

use crate::base::SendPtr;
use std::marker::PhantomData;
use std::sync::Arc;

// Import centralized constants for consistency
use moirai_core::constants::CACHE_LINE_SIZE;

/// Vector processing constants optimized for common CPU architectures
mod simd_constants {
    /// AVX2 vector width for f32 operations
    pub const AVX2_F32_WIDTH: usize = 8;
    /// SSE2 vector width for f32 operations  
    pub const SSE2_F32_WIDTH: usize = 4;
    /// Minimum vector size for vectorization to be beneficial
    pub const MIN_VECTORIZATION_SIZE: usize = 16;
    /// Cache-friendly chunk size for iterative processing
    pub const CACHE_FRIENDLY_CHUNK_SIZE: usize = CACHE_LINE_SIZE / std::mem::size_of::<f32>();
}

/// SIMD-optimized iterator for f32 operations with adaptive vectorization.
///
/// Automatically selects the best vectorization strategy based on:
/// - Available CPU features (runtime detection)
/// - Data size and alignment
/// - Cache characteristics
pub struct SimdF32Iterator<'a> {
    data: &'a [f32],
    chunk_size: usize,
    use_vectorization: bool,
}

impl<'a> SimdF32Iterator<'a> {
    pub fn new(data: &'a [f32]) -> Self {
        let (chunk_size, use_vectorization) = Self::determine_optimal_strategy(data);
        
        Self { 
            data, 
            chunk_size, 
            use_vectorization,
        }
    }

    /// Determine optimal processing strategy based on data characteristics
    fn determine_optimal_strategy(data: &[f32]) -> (usize, bool) {
        let len = data.len();
        
        // Don't vectorize small arrays - overhead isn't worth it
        if len < simd_constants::MIN_VECTORIZATION_SIZE {
            return (1, false);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return (simd_constants::AVX2_F32_WIDTH, true);
            } else if is_x86_feature_detected!("sse2") {
                return (simd_constants::SSE2_F32_WIDTH, true);
            }
        }

        // Fallback for other architectures or when SIMD is not available
        (simd_constants::CACHE_FRIENDLY_CHUNK_SIZE, false)
    }

    /// Apply vectorized addition with another slice using advanced techniques
    pub fn simd_add(self, other: &'a [f32]) -> Vec<f32> {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");

        if !self.use_vectorization {
            return self.scalar_add(other);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return self.avx2_add(other);
            } else if is_x86_feature_detected!("sse2") {
                return self.sse2_add(other);
            }
        }

        // Fallback to scalar operations
        self.scalar_add(other)
    }

    /// Scalar addition fallback
    fn scalar_add(self, other: &[f32]) -> Vec<f32> {
        self.data
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// AVX2-optimized addition (8 floats at a time)
    #[cfg(target_arch = "x86_64")]
    fn avx2_add(self, other: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;
        
        let len = self.data.len();
        let mut result = Vec::with_capacity(len);
        
        unsafe {
            let mut i = 0;
            
            // Process 8 elements at a time with AVX2
            while i + 8 <= len {
                let a = _mm256_loadu_ps(self.data.as_ptr().add(i));
                let b = _mm256_loadu_ps(other.as_ptr().add(i));
                let sum = _mm256_add_ps(a, b);
                
                // Store result
                let mut temp: [f32; 8] = [0.0; 8];
                _mm256_storeu_ps(temp.as_mut_ptr(), sum);
                result.extend_from_slice(&temp);
                
                i += 8;
            }
            
            // Handle remaining elements
            while i < len {
                result.push(self.data[i] + other[i]);
                i += 1;
            }
        }
        
        result
    }

    /// SSE2-optimized addition (4 floats at a time)
    #[cfg(target_arch = "x86_64")]
    fn sse2_add(self, other: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;
        
        let len = self.data.len();
        let mut result = Vec::with_capacity(len);
        
        unsafe {
            let mut i = 0;
            
            // Process 4 elements at a time with SSE2
            while i + 4 <= len {
                let a = _mm_loadu_ps(self.data.as_ptr().add(i));
                let b = _mm_loadu_ps(other.as_ptr().add(i));
                let sum = _mm_add_ps(a, b);
                
                // Store result
                let mut temp: [f32; 4] = [0.0; 4];
                _mm_storeu_ps(temp.as_mut_ptr(), sum);
                result.extend_from_slice(&temp);
                
                i += 4;
            }
            
            // Handle remaining elements
            while i < len {
                result.push(self.data[i] + other[i]);
                i += 1;
            }
        }
        
        result
    }

    /// Vectorized multiplication with automatic optimization
    pub fn simd_multiply(self, scalar: f32) -> Vec<f32> {
        if !self.use_vectorization {
            return self.data.iter().map(|x| x * scalar).collect();
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return self.avx2_multiply(scalar);
            }
        }

        // Fallback
        self.data.iter().map(|x| x * scalar).collect()
    }

    /// AVX2-optimized scalar multiplication
    #[cfg(target_arch = "x86_64")]
    fn avx2_multiply(self, scalar: f32) -> Vec<f32> {
        use std::arch::x86_64::*;
        
        let len = self.data.len();
        let mut result = Vec::with_capacity(len);
        
        unsafe {
            let scalar_vec = _mm256_set1_ps(scalar);
            let mut i = 0;
            
            while i + 8 <= len {
                let a = _mm256_loadu_ps(self.data.as_ptr().add(i));
                let product = _mm256_mul_ps(a, scalar_vec);
                
                let mut temp: [f32; 8] = [0.0; 8];
                _mm256_storeu_ps(temp.as_mut_ptr(), product);
                result.extend_from_slice(&temp);
                
                i += 8;
            }
            
            // Handle remaining elements
            while i < len {
                result.push(self.data[i] * scalar);
                i += 1;
            }
        }
        
        result
    }

    /// Compute dot product using SIMD acceleration
    pub fn simd_dot_product(self, other: &[f32]) -> f32 {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");

        if !self.use_vectorization {
            return self.data.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return self.avx2_dot_product(other);
            }
        }

        // Fallback
        self.data.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }

    /// AVX2-optimized dot product
    #[cfg(target_arch = "x86_64")]
    fn avx2_dot_product(self, other: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = self.data.len();
        
        unsafe {
            let mut sum_vec = _mm256_setzero_ps();
            let mut i = 0;
            
            // Process 8 elements at a time
            while i + 8 <= len {
                let a = _mm256_loadu_ps(self.data.as_ptr().add(i));
                let b = _mm256_loadu_ps(other.as_ptr().add(i));
                let product = _mm256_mul_ps(a, b);
                sum_vec = _mm256_add_ps(sum_vec, product);
                i += 8;
            }
            
            // Horizontal sum of the vector
            let mut temp: [f32; 8] = [0.0; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
            let mut result = temp.iter().sum::<f32>();
            
            // Handle remaining elements
            while i < len {
                result += self.data[i] * other[i];
                i += 1;
            }
            
            result
        }
    }
}

/// Cache-friendly iterator that processes data in cache-line sized chunks
/// for optimal memory access patterns. Based on "What Every Programmer Should Know About Memory".
pub struct CacheFriendlyIterator<T> {
    data: Vec<T>,
    chunk_size: usize,
}

impl<T: Clone> CacheFriendlyIterator<T> {
    pub fn new(data: Vec<T>) -> Self {
        let chunk_size = CACHE_LINE_SIZE / std::mem::size_of::<T>().max(1);
        Self { data, chunk_size }
    }

    /// Process data in cache-friendly chunks with given function
    pub fn process_chunks<F, R>(self, mut func: F) -> Vec<R>
    where
        F: FnMut(&[T]) -> R,
    {
        self.data
            .chunks(self.chunk_size)
            .map(|chunk| func(chunk))
            .collect()
    }

    /// Apply function to each element with prefetching hints
    pub fn map_with_prefetch<F, R>(self, func: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync,
        T: Send,
        R: Send,
    {
        let chunk_size = self.chunk_size;
        
        self.data
            .chunks(chunk_size)
            .flat_map(|chunk| {
                // Prefetch next chunk if available
                // In a real implementation, we'd use CPU prefetch instructions
                chunk.iter().cloned().map(&func).collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Advanced SIMD patterns for complex operations
pub struct AdvancedSimdOps;

impl AdvancedSimdOps {
    /// Parallel reduction using SIMD with tree-based combining
    /// Based on patterns from "Parallel Programming in C with MPI and OpenMP"
    pub fn simd_parallel_reduce<T, F, R>(data: &[T], identity: R, op: F) -> R
    where
        T: Copy + Send + Sync,
        F: Fn(R, T) -> R + Sync,
        R: Copy + Send + Sync,
    {
        if data.is_empty() {
            return identity;
        }

        // For now, use standard library reduce as a placeholder
        // Real implementation would use SIMD instructions and parallel tree reduction
        data.iter().fold(identity, |acc, &x| op(acc, x))
    }

    /// SIMD-accelerated filter operation
    pub fn simd_filter<T, P>(data: Vec<T>, predicate: P) -> Vec<T>
    where
        T: Copy,
        P: Fn(T) -> bool,
    {
        // Real implementation would use SIMD for the predicate evaluation
        // and vectorized compaction for collecting results
        data.into_iter().filter(predicate).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_addition() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        
        let iter = SimdF32Iterator::new(&a);
        let result = iter.simd_add(&b);
        
        let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_multiplication() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let iter = SimdF32Iterator::new(&data);
        let result = iter.simd_multiply(2.0);
        
        let expected = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        
        let iter = SimdF32Iterator::new(&a);
        let result = iter.simd_dot_product(&b);
        
        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert_eq!(result, 40.0);
    }

    #[test]
    fn test_cache_friendly_iterator() {
        let data = (0..100).collect::<Vec<i32>>();
        let iter = CacheFriendlyIterator::new(data);
        
        let results = iter.process_chunks(|chunk| chunk.iter().sum::<i32>());
        let total: i32 = results.iter().sum();
        
        // Sum of 0..100 = 99*100/2 = 4950
        assert_eq!(total, 4950);
    }
}
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
            return self
                .data
                .iter()
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
            return self.data.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
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
        for (&data_val, &other_val) in self.data[simd_len..].iter().zip(&other[simd_len..]) {
            sum += data_val * other_val;
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
                    use crate::cache::prefetch_read_data;
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
        let simd_chunk = if is_x86_feature_detected!("avx2") {
            8
        } else {
            1
        };
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
            let num_chunks = len.div_ceil(chunk_size);

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
            let num_chunks = len.div_ceil(chunk_size);

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

            for (chunk_a, chunk_b) in self
                .data
                .chunks(self.chunk_size)
                .zip(other.chunks(self.chunk_size))
            {
                let handle =
                    scope.spawn(move || SimdF32Iterator::new(chunk_a).simd_dot_product(chunk_b));
                handles.push(handle);
            }

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Sum all partial results
        partial_sums.iter().sum()
    }
}

/// Extension trait for SIMD operations on slices
pub trait SimdIteratorExt {
    /// Create a SIMD-optimized iterator for f32 slices
    fn simd_iter(&self) -> SimdF32Iterator<'_>;

    /// Create a parallel SIMD iterator for f32 slices
    fn par_simd_iter(&self) -> SimdParallelIterator<'_, f32>;
}

impl SimdIteratorExt for [f32] {
    fn simd_iter(&self) -> SimdF32Iterator<'_> {
        SimdF32Iterator::new(self)
    }

    fn par_simd_iter(&self) -> SimdParallelIterator<'_, f32> {
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
    ($feature:literal) => {
        false
    };
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
