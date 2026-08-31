use super::arch;

#[inline]
pub(super) fn native_vector_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if arch::has_avx2_support() {
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if arch::has_neon_support() {
            return true;
        }
    }

    false
}

#[inline]
pub(super) fn uses_native_vector_path(len: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if arch::has_avx2_support() && len >= arch::LANES {
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if arch::has_neon_support() && len >= arch::LANES {
            return true;
        }
    }

    false
}

#[inline]
pub(super) fn native_wide_vector_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if arch::has_avx2_support() {
            return true;
        }
    }

    false
}

#[inline]
pub(super) fn uses_native_wide_vector_path(len: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if arch::has_avx2_support() && len >= arch::LANES {
            return true;
        }
    }

    let _ = len;
    false
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
pub(super) fn native_vector_chunk_len(len: usize) -> Option<usize> {
    native_vector_available()
        .then_some((len / arch::LANES) * arch::LANES)
        .filter(|chunk_len| *chunk_len != 0)
}
