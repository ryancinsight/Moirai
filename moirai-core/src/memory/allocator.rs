use crate::memory::CACHE_LINE_SIZE;
use std::alloc::Layout;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;

/// Cache-aligned memory allocator for high-performance data structures.
pub struct CacheAlignedAllocator;

impl CacheAlignedAllocator {
    /// Allocate cache-aligned memory for optimal performance
    pub fn allocate<T>(count: usize) -> Option<NonNull<T>> {
        let size = size_of::<T>() * count;
        // Transfer granularity: this places the *start* of the array on a line
        // boundary so element 0 does not straddle two lines. Separating two
        // concurrently written atomics is a different problem, solved by
        // `CacheAligned` at the field level, not by widening this alignment.
        let align = align_of::<T>().max(CACHE_LINE_SIZE);

        let layout = Layout::from_size_align(size, align).ok()?;
        // A zero-sized layout violates `GlobalAlloc::alloc`'s contract.
        if layout.size() == 0 {
            return None;
        }

        // SAFETY: `layout` is valid and non-zero-sized; allocation failure
        // is surfaced as `None` through the null check.
        unsafe {
            #[cfg(feature = "mnemosyne")]
            {
                use core::alloc::GlobalAlloc;
                let ptr = mnemosyne::Mnemosyne.alloc(layout);
                NonNull::new(ptr.cast::<T>())
            }
            #[cfg(not(feature = "mnemosyne"))]
            {
                let ptr = std::alloc::alloc(layout);
                NonNull::new(ptr.cast::<T>())
            }
        }
    }

    /// Deallocate cache-aligned memory
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was allocated by `allocate` with the same type and count
    /// - `ptr` is valid and properly aligned
    /// - No other references to the memory exist
    /// - The memory is not accessed after deallocation
    pub unsafe fn deallocate<T>(ptr: NonNull<T>, count: usize) {
        let size = size_of::<T>() * count;
        // Transfer granularity: this places the *start* of the array on a line
        // boundary so element 0 does not straddle two lines. Separating two
        // concurrently written atomics is a different problem, solved by
        // `CacheAligned` at the field level, not by widening this alignment.
        let align = align_of::<T>().max(CACHE_LINE_SIZE);

        if let Ok(layout) = Layout::from_size_align(size, align) {
            #[cfg(feature = "mnemosyne")]
            {
                use core::alloc::GlobalAlloc;
                mnemosyne::Mnemosyne.dealloc(ptr.as_ptr().cast::<u8>(), layout);
            }
            #[cfg(not(feature = "mnemosyne"))]
            {
                std::alloc::dealloc(ptr.as_ptr().cast::<u8>(), layout);
            }
        }
    }
}
