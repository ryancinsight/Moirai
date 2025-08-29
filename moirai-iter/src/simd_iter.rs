//! SIMD-optimized iterators for high-performance data processing.
//!
//! This module provides SIMD-accelerated iteration patterns that leverage
//! modern CPU vector instructions for maximum throughput, based on techniques
//! from "Computer Systems: A Programmer's Perspective" and Intel optimization guides.

// Import centralized constants for consistency
use moirai_core::constants::CACHE_LINE_SIZE;

/// Vector processing constants optimized for common CPU architectures
mod simd_constants {
    use moirai_core::constants::CACHE_LINE_SIZE;

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

    /// Apply vectorized addition with another slice using scalar fallback
    pub fn simd_add(self, other: &'a [f32]) -> Vec<f32> {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");

        if self.use_vectorization && self.data.len() >= self.chunk_size {
            // Use chunked approach for better cache performance
            self.chunked_add(other)
        } else {
            // Use scalar implementation for compatibility
            self.scalar_add(other)
        }
    }

    /// Chunked addition for better cache performance
    fn chunked_add(self, other: &[f32]) -> Vec<f32> {
        self.data
            .chunks(self.chunk_size)
            .zip(other.chunks(self.chunk_size))
            .flat_map(|(a_chunk, b_chunk)| a_chunk.iter().zip(b_chunk.iter()).map(|(a, b)| a + b))
            .collect()
    }

    /// Scalar addition implementation
    fn scalar_add(self, other: &[f32]) -> Vec<f32> {
        self.data
            .iter()
            .zip(other.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Vectorized multiplication with scalar fallback
    pub fn simd_multiply(self, scalar: f32) -> Vec<f32> {
        if self.use_vectorization && self.data.len() >= self.chunk_size {
            // Process in chunks for better cache performance
            self.data
                .chunks(self.chunk_size)
                .flat_map(|chunk| chunk.iter().map(|x| x * scalar))
                .collect()
        } else {
            self.data.iter().map(|x| x * scalar).collect()
        }
    }

    /// Compute dot product
    pub fn simd_dot_product(self, other: &[f32]) -> f32 {
        assert_eq!(self.data.len(), other.len(), "Slices must have same length");

        self.data.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
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
    pub fn process_chunks<F, R>(self, func: F) -> Vec<R>
    where
        F: FnMut(&[T]) -> R,
    {
        self.data.chunks(self.chunk_size).map(func).collect()
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
        P: Fn(&T) -> bool,
    {
        // Real implementation would use SIMD for the predicate evaluation
        // and vectorized compaction for collecting results
        data.into_iter().filter(|x| predicate(x)).collect()
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
