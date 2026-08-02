//! SIMD-aware, cache-chunked iterator helpers.
//!
//! This module exposes one generic slice iterator instead of type-suffixed
//! entry points. Arithmetic executes in `T` through `SimdScalar`; callers choose
//! the scalar type at the call site and monomorphization removes the trait layer.

/// Cache line size for alignment optimizations
const CACHE_LINE_SIZE: usize = 64;
use std::iter::Sum;
use std::ops::{Add, Mul};

mod sealed {
    pub trait Sealed {}
}

/// Scalar contract for native-precision chunked arithmetic.
///
/// Implementations are sealed so the crate controls the arithmetic and layout
/// invariants used by the generic slice operations.
pub trait SimdScalar:
    sealed::Sealed + Copy + Send + Sync + Add<Output = Self> + Mul<Output = Self> + Sum<Self> + 'static
{
    /// Additive identity for the scalar type.
    const ZERO: Self;
}

impl sealed::Sealed for f32 {}
impl SimdScalar for f32 {
    const ZERO: Self = 0.0;
}

impl sealed::Sealed for f64 {}
impl SimdScalar for f64 {
    const ZERO: Self = 0.0;
}

impl sealed::Sealed for i32 {}
impl SimdScalar for i32 {
    const ZERO: Self = 0;
}

impl sealed::Sealed for i64 {}
impl SimdScalar for i64 {
    const ZERO: Self = 0;
}

impl sealed::Sealed for u32 {}
impl SimdScalar for u32 {
    const ZERO: Self = 0;
}

impl sealed::Sealed for u64 {}
impl SimdScalar for u64 {
    const ZERO: Self = 0;
}

impl sealed::Sealed for usize {}
impl SimdScalar for usize {
    const ZERO: Self = 0;
}

/// Generic SIMD-aware slice iterator over borrowed data.
pub struct SimdSliceIter<'a, T> {
    data: &'a [T],
}

impl<'a, T: SimdScalar> SimdSliceIter<'a, T> {
    /// Create a SIMD-aware iterator over the given slice.
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }

    /// Add `other` element-wise, returning a new vector.
    pub fn add_slice(self, other: &'a [T]) -> Vec<T> {
        assert_eq!(self.data.len(), other.len(), "slices must have same length");
        self.zip_map(other, |left, right| left + right)
    }

    /// Multiply every element by `scalar`, returning a new vector.
    pub fn scale(self, scalar: T) -> Vec<T> {
        self.data
            .iter()
            .copied()
            .map(|value| value * scalar)
            .collect()
    }

    /// Compute the dot product with `other`.
    pub fn dot(self, other: &'a [T]) -> T {
        assert_eq!(self.data.len(), other.len(), "slices must have same length");

        self.data
            .iter()
            .copied()
            .zip(other.iter().copied())
            .fold(T::ZERO, |acc, (left, right)| acc + left * right)
    }

    fn zip_map<F>(self, other: &'a [T], mut func: F) -> Vec<T>
    where
        F: FnMut(T, T) -> T,
    {
        self.data
            .iter()
            .copied()
            .zip(other.iter().copied())
            .map(|(left, right)| func(left, right))
            .collect()
    }
}

/// Cache-friendly iterator that processes data in cache-line sized chunks.
pub struct CacheFriendlyIterator<T> {
    data: Vec<T>,
    chunk_size: usize,
}

impl<T: Clone> CacheFriendlyIterator<T> {
    /// Create a cache-friendly iterator over `data`, sizing chunks to a cache line.
    pub fn new(data: Vec<T>) -> Self {
        let scalar_size = std::mem::size_of::<T>().max(1);
        let chunk_size = (CACHE_LINE_SIZE / scalar_size).max(1);
        Self { data, chunk_size }
    }

    /// Apply `func` to each cache-sized chunk, collecting the results.
    pub fn process_chunks<F, R>(self, func: F) -> Vec<R>
    where
        F: FnMut(&[T]) -> R,
    {
        self.data.chunks(self.chunk_size).map(func).collect()
    }

    /// Map every element through `func`, processing chunk-by-chunk.
    pub fn map_with_prefetch<F, R>(self, func: F) -> Vec<R>
    where
        F: Fn(T) -> R + Sync,
        T: Send,
        R: Send,
    {
        self.data
            .chunks(self.chunk_size)
            .flat_map(|chunk| chunk.iter().cloned().map(&func))
            .collect()
    }
}

/// Generic chunked operations for slice-oriented processing.
pub struct SimdOps;

impl SimdOps {
    /// Left-fold `op` over `data` with `identity`.
    pub fn reduce<T, F, R>(data: &[T], identity: R, op: F) -> R
    where
        T: Copy + Send + Sync,
        F: Fn(R, T) -> R + Sync,
        R: Copy + Send + Sync,
    {
        data.iter().copied().fold(identity, op)
    }

    /// Keep only the elements for which `predicate` returns true.
    pub fn filter<T, P>(data: Vec<T>, predicate: P) -> Vec<T>
    where
        T: Copy,
        P: Fn(&T) -> bool,
    {
        data.into_iter().filter(predicate).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_slice_addition_preserves_values() {
        let left = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
        let right = vec![1_u32, 1, 1, 1, 1, 1, 1, 1];

        let result = SimdSliceIter::new(&left).add_slice(&right);

        assert_eq!(result, vec![2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn generic_slice_scale_preserves_native_precision_values() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];

        let result = SimdSliceIter::new(&data).scale(2.0);

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn generic_slice_dot_preserves_values() {
        let left = vec![1_i32, 2, 3, 4];
        let right = vec![2_i32, 3, 4, 5];

        let result = SimdSliceIter::new(&left).dot(&right);

        assert_eq!(result, 40);
    }

    #[test]
    fn cache_friendly_iterator_processes_large_elements() {
        #[derive(Clone)]
        struct Large([u8; CACHE_LINE_SIZE * 2]);

        let data = vec![Large([3; CACHE_LINE_SIZE * 2]); 4];
        let iter = CacheFriendlyIterator::new(data);

        let results =
            iter.process_chunks(|chunk| chunk.iter().map(|item| item.0[0] as usize).sum::<usize>());
        let total: usize = results.iter().sum();

        assert_eq!(total, 12);
    }

    #[test]
    fn simd_ops_reduce_and_filter_are_value_semantic() {
        let data = vec![1_u64, 2, 3, 4, 5, 6];

        let reduced = SimdOps::reduce(&data, 10_u64, |acc, item| acc + item);
        let filtered = SimdOps::filter(data, |item| item % 2 == 0);

        assert_eq!(reduced, 31);
        assert_eq!(filtered, vec![2, 4, 6]);
    }
}
