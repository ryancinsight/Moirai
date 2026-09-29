use crate::memory::CACHE_LINE_SIZE;
use std::alloc::Layout;
use std::ptr::NonNull;

/// Cache-aligned memory allocator for high-performance data structures.
pub struct CacheAlignedAllocator;

/// Layout of `count` values of `T` starting on a cache-line boundary.
///
/// Transfer granularity: this places the *start* of the array on a line
/// boundary so element 0 does not straddle two lines. Separating two
/// concurrently written atomics is a different problem, solved by
/// `CacheAligned` at the field level, not by widening this alignment. The size
/// comes from `Layout::array`, so a count whose byte size overflows `isize` is
/// refused instead of wrapping into an undersized allocation.
fn cache_aligned_array<T>(count: usize) -> Option<Layout> {
    Layout::array::<T>(count)
        .ok()?
        .align_to(CACHE_LINE_SIZE)
        .ok()
}

impl CacheAlignedAllocator {
    /// Allocate cache-aligned memory for optimal performance
    pub fn allocate<T>(count: usize) -> Option<NonNull<T>> {
        let layout = cache_aligned_array::<T>(count)?;
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
        unsafe {
            if let Some(layout) = cache_aligned_array::<T>(count) {
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
}
