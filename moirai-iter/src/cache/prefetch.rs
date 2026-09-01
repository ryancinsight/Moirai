//! Architecture-specific cache prefetch hints.

/// Prefetch data for reading with the selected cache level.
///
/// # Safety
///
/// The pointer must refer to readable memory. On x86-64, level selects T0,
/// T1, T2, or non-temporal for any other value. Other architectures currently
/// treat this as a no-op.
#[inline(always)]
pub unsafe fn prefetch_read_data(ptr: *const u8, level: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            _mm_prefetch, _MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2,
        };
        match level {
            0 => _mm_prefetch(ptr.cast(), _MM_HINT_T0),
            1 => _mm_prefetch(ptr.cast(), _MM_HINT_T1),
            2 => _mm_prefetch(ptr.cast(), _MM_HINT_T2),
            _ => _mm_prefetch(ptr.cast(), _MM_HINT_NTA),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ptr, level);
    }
}

/// Prefetch data for writing with the selected cache level.
///
/// # Safety
///
/// The pointer must refer to writable memory. On x86-64, level selects T0,
/// T1, T2, or non-temporal for any other value. Other architectures currently
/// treat this as a no-op.
#[inline(always)]
pub unsafe fn prefetch_write_data(ptr: *mut u8, level: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            _mm_prefetch, _MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2,
        };
        match level {
            0 => _mm_prefetch(ptr.cast_const().cast(), _MM_HINT_T0),
            1 => _mm_prefetch(ptr.cast_const().cast(), _MM_HINT_T1),
            2 => _mm_prefetch(ptr.cast_const().cast(), _MM_HINT_T2),
            _ => _mm_prefetch(ptr.cast_const().cast(), _MM_HINT_NTA),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ptr, level);
    }
}
