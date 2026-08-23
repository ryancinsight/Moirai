//! Memory utilities for cache optimization and prefetching.
//!
//! This module provides utilities for optimizing memory access patterns,
//! including cache prefetching and aligned memory allocation.

/// Prefetch memory for reading.
///
/// This function hints to the processor that the specified memory location
/// will be read soon, allowing it to preload the data into cache.
/// On architectures that don't support prefetching, this is a no-op.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: prefetch hints are non-faulting and advisory, so `ptr`
        // need not be dereferenceable; SSE is baseline on x86_64 and the
        // cast carries only the address.
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
        // SAFETY: prefetch hints are non-faulting and advisory, so `ptr`
        // need not be dereferenceable; SSE is baseline on x86_64 and the
        // cast carries only the address.
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr; // Suppress unused variable warning
    }
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
    // Transfer granularity, not the false-sharing separation: one prefetch
    // hint covers one line, so striding by the (larger) interference size
    // would skip every second line in the range.
    use crate::cache::CACHE_LINE_SIZE;

    let start = ptr as usize;
    // Saturating/checked address arithmetic: a range ending near `usize::MAX`
    // must not overflow (which panics under `overflow-checks`). Prefetch hints
    // are non-faulting, so the only requirement is that the loop terminates.
    let end = start.saturating_add(bytes);

    let mut addr = start & !(CACHE_LINE_SIZE - 1); // Align to cache line
    while addr < end {
        prefetch_read(addr as *const u8);
        match addr.checked_add(CACHE_LINE_SIZE) {
            Some(next) => addr = next,
            None => break,
        }
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
        let bytes = core::mem::size_of_val(slice);
        prefetch_range_read(slice.as_ptr(), bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
