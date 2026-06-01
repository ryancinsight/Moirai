//! Private native ISA backends for the generic SIMD surface.

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86;

/// Returns true when the current x86 CPU supports AVX2.
#[inline]
pub fn has_avx2_support() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std", not(target_arch = "wasm32")))]
    {
        std::is_x86_feature_detected!("avx2")
    }

    #[cfg(not(all(target_arch = "x86_64", feature = "std", not(target_arch = "wasm32"))))]
    {
        false
    }
}

/// Returns true when the current AArch64 CPU supports NEON.
#[inline]
pub fn has_neon_support() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{
    add, add_f64, dot, dot_f64, matrix_mul_square, mul, mul_f64, squared_diff_sum,
    squared_diff_sum_f64, sum, sum_f64, LANES,
};

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::{add, dot, matrix_mul_square, mul, squared_diff_sum, sum, LANES};
