//! AArch64 native vector backend.

use core::arch::aarch64::*;

pub(crate) const LANES: usize = 4;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn add(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let left_values = vld1q_f32(left.as_ptr().add(offset));
            let right_values = vld1q_f32(right.as_ptr().add(offset));
            let output = vaddq_f32(left_values, right_values);
            vst1q_f32(result.as_mut_ptr().add(offset), output);
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn mul(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let left_values = vld1q_f32(left.as_ptr().add(offset));
            let right_values = vld1q_f32(right.as_ptr().add(offset));
            let output = vmulq_f32(left_values, right_values);
            vst1q_f32(result.as_mut_ptr().add(offset), output);
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % LANES, 0);

    let mut total = vdupq_n_f32(0.0);
    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let left_values = vld1q_f32(left.as_ptr().add(offset));
            let right_values = vld1q_f32(right.as_ptr().add(offset));
            total = vaddq_f32(total, vmulq_f32(left_values, right_values));
        }
    }

    horizontal_sum(total)
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn sum(data: &[f32]) -> f32 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mut total = vdupq_n_f32(0.0);
    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let values = vld1q_f32(data.as_ptr().add(offset));
            total = vaddq_f32(total, values);
        }
    }

    horizontal_sum(total)
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn squared_diff_sum(data: &[f32], mean: f32) -> f32 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mean_values = vdupq_n_f32(mean);
    let mut total = vdupq_n_f32(0.0);

    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let values = vld1q_f32(data.as_ptr().add(offset));
            let diff = vsubq_f32(values, mean_values);
            total = vaddq_f32(total, vmulq_f32(diff, diff));
        }
    }

    horizontal_sum(total)
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn matrix_mul_square(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), 16);
    debug_assert_eq!(right.len(), 16);
    debug_assert_eq!(result.len(), 16);

    for row in 0..4 {
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
