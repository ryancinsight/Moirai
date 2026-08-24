//! AArch64 native vector backend.

use core::arch::aarch64::*;

pub(crate) const LANES: usize = 4;

/// Adds two lane-multiple f32 slices element-wise into `result`.
///
/// # Safety
///
/// - NEON must be available: mandatory on every aarch64 target this crate
///   builds for, and the dispatcher additionally probes
///   [`super::has_neon_support`] before reaching this function.
/// - All three slices must have equal, `LANES`-multiple lengths. Every call
///   site slices its inputs to `native_vector_chunk_len` (scalar.rs), which
///   returns `(len / LANES) * LANES` only when `len >= LANES`.
/// - `result` must not overlap `left` or `right`; the safe surface enforces
///   this by borrowing (`&mut [f32]` against two `&[f32]`).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: `offset + LANES <= left.len()` holds every iteration because
        // `left.len()` is a LANES multiple, so each load/store window is in
        // bounds; disjointness follows from the fn contract on `result`.
        unsafe {
            let left_values = vld1q_f32(left.as_ptr().add(offset));
            let right_values = vld1q_f32(right.as_ptr().add(offset));
            let output = vaddq_f32(left_values, right_values);
            vst1q_f32(result.as_mut_ptr().add(offset), output);
        }
    }
}

/// Multiplies two lane-multiple f32 slices element-wise into `result`.
///
/// # Safety
///
/// Same contract as [`add`]: NEON available, equal `LANES`-multiple lengths
/// sliced by the dispatcher, and `result` disjoint from both inputs.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mul(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: identical bound argument to `add`'s loop.
        unsafe {
            let left_values = vld1q_f32(left.as_ptr().add(offset));
            let right_values = vld1q_f32(right.as_ptr().add(offset));
            let output = vmulq_f32(left_values, right_values);
            vst1q_f32(result.as_mut_ptr().add(offset), output);
        }
    }
}

/// Lane-multiple f32 dot product, native-precision accumulated.
///
/// # Safety
///
/// NEON available (mandatory on aarch64 targets; dispatcher-probed) and both
/// slices of equal `LANES`-multiple length per the dispatcher's chunking.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % LANES, 0);

    let mut total = vdupq_n_f32(0.0);
    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: read-only windows at `offset + LANES <= left.len()`;
        // lengths are equal LANES multiples per contract.
        unsafe {
            let left_values = vld1q_f32(left.as_ptr().add(offset));
            let right_values = vld1q_f32(right.as_ptr().add(offset));
            total = vaddq_f32(total, vmulq_f32(left_values, right_values));
        }
    }

    horizontal_sum(total)
}

/// Lane-multiple f32 sum, native-precision accumulated.
///
/// # Safety
///
/// NEON available and `data.len()` a `LANES` multiple per the dispatcher's
/// chunking (`native_vector_chunk_len`).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn sum(data: &[f32]) -> f32 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mut total = vdupq_n_f32(0.0);
    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: read-only windows at `offset + LANES <= data.len()`.
        unsafe {
            let values = vld1q_f32(data.as_ptr().add(offset));
            total = vaddq_f32(total, values);
        }
    }

    horizontal_sum(total)
}

/// Sum of squared deviations from `mean`, native-precision accumulated.
///
/// # Safety
///
/// NEON available and `data.len()` a `LANES` multiple per the dispatcher's
/// chunking.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn squared_diff_sum(data: &[f32], mean: f32) -> f32 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mean_values = vdupq_n_f32(mean);
    let mut total = vdupq_n_f32(0.0);

    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        // SAFETY: read-only windows at `offset + LANES <= data.len()`.
        unsafe {
            let values = vld1q_f32(data.as_ptr().add(offset));
            let diff = vsubq_f32(values, mean_values);
            total = vaddq_f32(total, vmulq_f32(diff, diff));
        }
    }

    horizontal_sum(total)
}

/// 4x4 row-major matrix product through NEON.
///
/// # Safety
///
/// All three slices must hold exactly 16 elements (4x4). The only caller
/// gates on `N == 4` behind a const-generic public surface (scalar.rs),
/// which fixes the shape; NEON availability follows the shared contract.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn matrix_mul_square(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), 16);
    debug_assert_eq!(right.len(), 16);
    debug_assert_eq!(result.len(), 16);

    for row in 0..4 {
        // SAFETY: with len-16 contracts, `row * 4 + col < 16` bounds every
        // unchecked access, and the column gather reads offsets {0,4,8,12}+col,
        // all < 16; `result` disjointness follows from safe-surface borrowing.
        unsafe {
            let left_row = vld1q_f32(left.as_ptr().add(row * 4));
            for col in 0..4 {
                let right_col = [
                    *right.get_unchecked(col),
                    *right.get_unchecked(4 + col),
                    *right.get_unchecked(8 + col),
                    *right.get_unchecked(12 + col),
                ];
                let right_values = vld1q_f32(right_col.as_ptr());
                *result.get_unchecked_mut(row * 4 + col) =
                    horizontal_sum(vmulq_f32(left_row, right_values));
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn horizontal_sum(values: float32x4_t) -> f32 {
    let pairs = vpadd_f32(vget_low_f32(values), vget_high_f32(values));
    let total = vpadd_f32(pairs, pairs);
    vget_lane_f32(total, 0)
}
