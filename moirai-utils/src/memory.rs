//! Memory utilities for cache optimization and prefetching.
//!
//! This module provides utilities for optimizing memory access patterns,
//! including cache prefetching and aligned memory allocation.

use crate::cache::CacheAligned;

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Prefetch memory for reading.
///
/// This function hints to the processor that the specified memory location
/// will be read soon, allowing it to preload the data into cache.
/// On architectures that don't support prefetching, this is a no-op.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr; // Suppress unused variable warning
    }
}

/// Prefetch memory for writing.
///
/// This function hints to the processor that the specified memory location
/// will be written soon. On most architectures, this is equivalent to
/// prefetch_read since writing also requires reading the cache line.
#[inline(always)]
pub fn prefetch_write<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr; // Suppress unused variable warning
    }
}

/// Create a cache-aligned vector.
///
/// This function creates a vector filled with clones of the given value,
/// wrapped in a cache-aligned container to prevent false sharing.
///
/// # Arguments
/// * `value` - The value to clone for each element
/// * `count` - The number of elements in the vector
///
/// # Returns
/// A cache-aligned vector containing `count` copies of `value`
pub fn aligned_vec<T: Clone>(value: T, count: usize) -> CacheAligned<Vec<T>> {
    CacheAligned::new(vec![value; count])
}

/// Prefetch a range of memory for reading.
///
/// This function prefetches multiple cache lines starting from the given
/// pointer for the specified number of bytes.
///
/// # Arguments
/// * `ptr` - Starting memory address
/// * `bytes` - Number of bytes to prefetch
pub fn prefetch_range_read<T>(ptr: *const T, bytes: usize) {
    const CACHE_LINE_SIZE: usize = 64;
    let start = ptr as usize;
    let end = start + bytes;

    let mut addr = start & !(CACHE_LINE_SIZE - 1); // Align to cache line
    while addr < end {
        prefetch_read(addr as *const u8);
        addr += CACHE_LINE_SIZE;
    }
}

/// Prefetch a slice for reading.
///
/// This function prefetches all memory containing the given slice.
///
/// # Arguments
/// * `slice` - The slice to prefetch
pub fn prefetch_slice_read<T>(slice: &[T]) {
    if !slice.is_empty() {
        let bytes = std::mem::size_of_val(slice);
        prefetch_range_read(slice.as_ptr(), bytes);
    }
}

/// Check if a pointer is aligned to the given boundary.
///
/// # Arguments
/// * `ptr` - The pointer to check
/// * `alignment` - The alignment boundary (must be a power of 2)
///
/// # Returns
/// True if the pointer is aligned to the specified boundary
pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
    debug_assert!(
        alignment.is_power_of_two(),
        "Alignment must be a power of 2"
    );
    (ptr as usize) & (alignment - 1) == 0
}

/// Check if a pointer is cache-line aligned.
///
/// # Arguments
/// * `ptr` - The pointer to check
///
/// # Returns
/// True if the pointer is aligned to a cache line boundary
pub fn is_cache_aligned<T>(ptr: *const T) -> bool {
    is_aligned(ptr, 64) // Assuming 64-byte cache lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec() {
        let aligned = aligned_vec(42, 10);
        assert_eq!(aligned.len(), 10);
        assert!(aligned.iter().all(|&x| x == 42));
    }

    #[test]
    fn test_is_aligned() {
        let data = [1, 2, 3, 4];
        let ptr = data.as_ptr();

        // Should be aligned to 1-byte boundary
        assert!(is_aligned(ptr, 1));

        // May or may not be aligned to larger boundaries
        // depending on the allocator
    }

    #[test]
    fn test_prefetch_slice() {
        let data = vec![1, 2, 3, 4, 5];
        // This should not panic
        prefetch_slice_read(&data);

        // Test empty slice
        let empty: &[i32] = &[];
        prefetch_slice_read(empty);
    }

    #[test]
    fn test_prefetch_functions() {
        let data = 42;
        let ptr = &data as *const i32;

        // These should not panic
        prefetch_read(ptr);
        prefetch_write(ptr);
        prefetch_range_read(ptr, 4);
    }
}
