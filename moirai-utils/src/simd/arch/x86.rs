//! x86 native vector backend (AVX2).
//!
//! Every kernel here is an `unsafe fn` carrying the same three preconditions, so
//! they are stated once and referenced by each `# Safety` section rather than
//! restated eleven times. All of them are discharged by the dispatcher in
//! `simd::scalar`, which is the only caller.
//!
//! # The shared contract
//!
//! 1. **AVX2 is available on this CPU.** These functions are compiled with
//!    `#[target_feature(enable = "avx2")]`, so executing one on a CPU without
//!    AVX2 is undefined behavior (in practice `SIGILL`). The dispatcher probes
//!    `has_avx2_support` — `is_x86_feature_detected!("avx2")` — before every
//!    call and otherwise takes the scalar path. Note that AVX2 transitively
//!    enables SSE4.1 in the compiler's feature hierarchy, which is what makes
//!    the `_mm_dp_ps` in `matrix_mul_square` legitimate under this attribute.
//! 2. **The slices have equal length.** Each kernel derives its trip count from
//!    one slice and indexes all of them with it, so a shorter operand would read
//!    or write out of bounds. Call sites pass `&x[..chunk_len]` for one shared
//!    `chunk_len`; slicing panics rather than producing a short slice, so the
//!    equality is enforced at the boundary and cannot reach these bodies.
//! 3. **The length is a whole number of blocks** (`len % LANES == 0`). Callers
//!    compute `chunk_len` as `(len / LANES) * LANES` and hand the remainder to
//!    the scalar tail, so partial blocks never arrive here.
//!
//! The `debug_assert_eq!`s in each body restate 2 and 3 so a violated contract
//! surfaces as a dev-build panic instead of release-build memory corruption.
//!
//! `matrix_mul_square` additionally requires all three slices to be exactly 16
//! elements; it is the one kernel using unchecked indexing, and the dispatcher
//! runs `scalar_matrix_shape` — real `assert_eq!`s, not debug-only — immediately
//! before calling it.
//!
//! # Block width
//!
//! `LANES` is 8 for both real widths, which is what lets one prefix/tail
//! contract serve both: single precision fills one 8-lane `__m256` per block,
//! and double precision fills two 4-lane `__m256d` registers per block.
//!
//! # Reduction order
//!
//! The reductions accumulate into vector lanes and fold them at the end, so they
//! sum in a different order than a sequential loop. Floating-point addition is
//! not associative: results agree with the scalar path to within a derived
//! roundoff bound, not bitwise, except where every partial sum is exactly
//! representable. Differential tests bound the difference accordingly.

use core::arch::x86_64::*;

// Backend block width. Single-precision uses one AVX register; the wide-real
// path uses two four-lane registers per block to keep one prefix/tail contract.
pub(crate) const LANES: usize = 8;

/// Element-wise `left + right` into `result`.
///
/// # Safety
/// The module's shared contract: AVX2 available, all three slices of equal
/// length, and that length a multiple of `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: `offset + LANES <= len` on every iteration, and all three
        // slices are `len` long, so each 8-lane access stays in bounds. The
        // loads and store are unaligned forms, so no alignment is required.
        unsafe {
            let left_values = _mm256_loadu_ps(left.as_ptr().add(offset));
            let right_values = _mm256_loadu_ps(right.as_ptr().add(offset));
            let output = _mm256_add_ps(left_values, right_values);
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), output);
        }
    }
}

/// Element-wise `left * right` into `result`.
///
/// # Safety
/// The module's shared contract: AVX2 available, all three slices of equal
/// length, and that length a multiple of `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn mul(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: as `add` — every 8-lane access lies within the common length.
        unsafe {
            let left_values = _mm256_loadu_ps(left.as_ptr().add(offset));
            let right_values = _mm256_loadu_ps(right.as_ptr().add(offset));
            let output = _mm256_mul_ps(left_values, right_values);
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), output);
        }
    }
}

/// Dot product of `left` and `right`.
///
/// Accumulates per lane and folds at the end, so the summation order differs
/// from a sequential loop (see the module's reduction-order note).
///
/// # Safety
/// The module's shared contract: AVX2 available, both slices of equal length,
/// and that length a multiple of `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % LANES, 0);

    let mut total = _mm256_setzero_ps();
    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: both slices are the same length and every 8-lane load lies
        // within it.
        unsafe {
            let left_values = _mm256_loadu_ps(left.as_ptr().add(offset));
            let right_values = _mm256_loadu_ps(right.as_ptr().add(offset));
            total = _mm256_add_ps(total, _mm256_mul_ps(left_values, right_values));
        }
    }

    horizontal_sum(total)
}

/// Sum of `data`.
///
/// Accumulates per lane and folds at the end (see the module's reduction-order
/// note).
///
/// # Safety
/// The module's shared contract: AVX2 available and `data.len()` a multiple of
/// `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sum(data: &[f32]) -> f32 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mut total = _mm256_setzero_ps();
    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: every 8-lane load lies within `data`.
        unsafe {
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));
            total = _mm256_add_ps(total, values);
        }
    }

    horizontal_sum(total)
}

/// Sum of `(value - mean)^2` over `data`, the variance numerator.
///
/// # Safety
/// The module's shared contract: AVX2 available and `data.len()` a multiple of
/// `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn squared_diff_sum(data: &[f32], mean: f32) -> f32 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mean_values = _mm256_set1_ps(mean);
    let mut total = _mm256_setzero_ps();

    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: every 8-lane load lies within `data`.
        unsafe {
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));
            let diff = _mm256_sub_ps(values, mean_values);
            total = _mm256_add_ps(total, _mm256_mul_ps(diff, diff));
        }
    }

    horizontal_sum(total)
}

/// Row-major 4x4 matrix product, `result = left * right`.
///
/// # Safety
/// AVX2 must be available, and all three slices must be exactly 16 elements —
/// this kernel indexes them unchecked. The dispatcher enforces the length with
/// `scalar_matrix_shape`, whose `assert_eq!`s run in release too.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn matrix_mul_square(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), 16);
    debug_assert_eq!(right.len(), 16);
    debug_assert_eq!(result.len(), 16);

    for row in 0..4 {
        // SAFETY: with all three slices 16 elements long, every index below is
        // in bounds: the row load covers `row * 4 .. row * 4 + 4` for `row < 4`,
        // the gathered column indices are `col`, `4 + col`, `8 + col`,
        // `12 + col` for `col < 4`, and the store index `row * 4 + col` is at
        // most 15. `_mm_dp_ps` is SSE4.1, which `avx2` transitively enables.
        unsafe {
            let left_row = _mm_loadu_ps(left.as_ptr().add(row * 4));
            for col in 0..4 {
                let right_col = _mm_set_ps(
                    *right.get_unchecked(12 + col),
                    *right.get_unchecked(8 + col),
                    *right.get_unchecked(4 + col),
                    *right.get_unchecked(col),
                );
                let dot = _mm_dp_ps(left_row, right_col, 0xF1);
                *result.get_unchecked_mut(row * 4 + col) = _mm_cvtss_f32(dot);
            }
        }
    }
}

/// Fold eight lanes to their scalar sum.
///
/// Safe despite `target_feature`: it takes an already-formed vector and touches
/// no memory, so the only requirement is AVX2, which the compiler enforces by
/// permitting calls solely from functions that enable the same feature.
#[inline]
#[target_feature(enable = "avx2")]
fn horizontal_sum(values: __m256) -> f32 {
    let high = _mm256_extractf128_ps(values, 1);
    let low = _mm256_castps256_ps128(values);
    let combined = _mm_add_ps(high, low);
    let shuffled = _mm_shuffle_ps(combined, combined, 0b0100_1110);
    let pairs = _mm_add_ps(combined, shuffled);
    let total = _mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 0b0000_0001));
    _mm_cvtss_f32(total)
}

/// Element-wise `left + right` into `result`, double precision.
///
/// # Safety
/// The module's shared contract: AVX2 available, all three slices of equal
/// length, and that length a multiple of `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add_wide(left: &[f64], right: &[f64], result: &mut [f64]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: a block spans `offset .. offset + LANES` and is covered by two
        // 4-lane accesses at `offset` and `offset + 4`, so the whole block lies
        // within the common length.
        unsafe {
            let left_v0 = _mm256_loadu_pd(left.as_ptr().add(offset));
            let left_v1 = _mm256_loadu_pd(left.as_ptr().add(offset + 4));
            let right_v0 = _mm256_loadu_pd(right.as_ptr().add(offset));
            let right_v1 = _mm256_loadu_pd(right.as_ptr().add(offset + 4));
            let output_v0 = _mm256_add_pd(left_v0, right_v0);
            let output_v1 = _mm256_add_pd(left_v1, right_v1);
            _mm256_storeu_pd(result.as_mut_ptr().add(offset), output_v0);
            _mm256_storeu_pd(result.as_mut_ptr().add(offset + 4), output_v1);
        }
    }
}

/// Element-wise `left * right` into `result`, double precision.
///
/// # Safety
/// The module's shared contract: AVX2 available, all three slices of equal
/// length, and that length a multiple of `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn mul_wide(left: &[f64], right: &[f64], result: &mut [f64]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: as `add_wide` — the two 4-lane accesses cover the block.
        unsafe {
            let left_v0 = _mm256_loadu_pd(left.as_ptr().add(offset));
            let left_v1 = _mm256_loadu_pd(left.as_ptr().add(offset + 4));
            let right_v0 = _mm256_loadu_pd(right.as_ptr().add(offset));
            let right_v1 = _mm256_loadu_pd(right.as_ptr().add(offset + 4));
            let output_v0 = _mm256_mul_pd(left_v0, right_v0);
            let output_v1 = _mm256_mul_pd(left_v1, right_v1);
            _mm256_storeu_pd(result.as_mut_ptr().add(offset), output_v0);
            _mm256_storeu_pd(result.as_mut_ptr().add(offset + 4), output_v1);
        }
    }
}

/// Dot product of `left` and `right`, double precision.
///
/// # Safety
/// The module's shared contract: AVX2 available, both slices of equal length,
/// and that length a multiple of `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_wide(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % LANES, 0);

    let mut total_v0 = _mm256_setzero_pd();
    let mut total_v1 = _mm256_setzero_pd();
    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: as `add_wide` — the two 4-lane loads cover the block.
        unsafe {
            let left_v0 = _mm256_loadu_pd(left.as_ptr().add(offset));
            let left_v1 = _mm256_loadu_pd(left.as_ptr().add(offset + 4));
            let right_v0 = _mm256_loadu_pd(right.as_ptr().add(offset));
            let right_v1 = _mm256_loadu_pd(right.as_ptr().add(offset + 4));
            total_v0 = _mm256_add_pd(total_v0, _mm256_mul_pd(left_v0, right_v0));
            total_v1 = _mm256_add_pd(total_v1, _mm256_mul_pd(left_v1, right_v1));
        }
    }

    horizontal_sum_wide(_mm256_add_pd(total_v0, total_v1))
}

/// Sum of `data`, double precision.
///
/// # Safety
/// The module's shared contract: AVX2 available and `data.len()` a multiple of
/// `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sum_wide(data: &[f64]) -> f64 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mut total_v0 = _mm256_setzero_pd();
    let mut total_v1 = _mm256_setzero_pd();
    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: as `add_wide` — the two 4-lane loads cover the block.
        unsafe {
            let values_v0 = _mm256_loadu_pd(data.as_ptr().add(offset));
            let values_v1 = _mm256_loadu_pd(data.as_ptr().add(offset + 4));
            total_v0 = _mm256_add_pd(total_v0, values_v0);
            total_v1 = _mm256_add_pd(total_v1, values_v1);
        }
    }

    horizontal_sum_wide(_mm256_add_pd(total_v0, total_v1))
}

/// Sum of `(value - mean)^2` over `data`, double precision.
///
/// # Safety
/// The module's shared contract: AVX2 available and `data.len()` a multiple of
/// `LANES`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn squared_diff_sum_wide(data: &[f64], mean: f64) -> f64 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mean_values = _mm256_set1_pd(mean);
    let mut total_v0 = _mm256_setzero_pd();
    let mut total_v1 = _mm256_setzero_pd();

    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: as `add_wide` — the two 4-lane loads cover the block.
        unsafe {
            let values_v0 = _mm256_loadu_pd(data.as_ptr().add(offset));
            let values_v1 = _mm256_loadu_pd(data.as_ptr().add(offset + 4));
            let diff_v0 = _mm256_sub_pd(values_v0, mean_values);
            let diff_v1 = _mm256_sub_pd(values_v1, mean_values);
            total_v0 = _mm256_add_pd(total_v0, _mm256_mul_pd(diff_v0, diff_v0));
            total_v1 = _mm256_add_pd(total_v1, _mm256_mul_pd(diff_v1, diff_v1));
        }
    }

    horizontal_sum_wide(_mm256_add_pd(total_v0, total_v1))
}

/// Fold four lanes to their scalar sum. Safe for the same reason as
/// [`horizontal_sum`].
#[inline]
#[target_feature(enable = "avx2")]
fn horizontal_sum_wide(values: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(values, 1);
    let low = _mm256_castpd256_pd128(values);
    let combined = _mm_add_pd(high, low);
    let shuffled = _mm_shuffle_pd(combined, combined, 1);
    let total = _mm_add_sd(combined, shuffled);
    _mm_cvtsd_f64(total)
}
