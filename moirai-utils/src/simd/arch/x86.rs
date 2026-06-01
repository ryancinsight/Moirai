//! x86 native vector backend.

use core::arch::x86_64::*;

pub(crate) const LANES: usize = 8;

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let left_values = _mm256_loadu_ps(left.as_ptr().add(offset));
            let right_values = _mm256_loadu_ps(right.as_ptr().add(offset));
            let output = _mm256_add_ps(left_values, right_values);
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), output);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn mul(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let left_values = _mm256_loadu_ps(left.as_ptr().add(offset));
            let right_values = _mm256_loadu_ps(right.as_ptr().add(offset));
            let output = _mm256_mul_ps(left_values, right_values);
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), output);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % LANES, 0);

    let mut total = _mm256_setzero_ps();
    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let left_values = _mm256_loadu_ps(left.as_ptr().add(offset));
            let right_values = _mm256_loadu_ps(right.as_ptr().add(offset));
            total = _mm256_add_ps(total, _mm256_mul_ps(left_values, right_values));
        }
    }

    horizontal_sum(total)
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sum(data: &[f32]) -> f32 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mut total = _mm256_setzero_ps();
    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));
            total = _mm256_add_ps(total, values);
        }
    }

    horizontal_sum(total)
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn squared_diff_sum(data: &[f32], mean: f32) -> f32 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mean_values = _mm256_set1_ps(mean);
    let mut total = _mm256_setzero_ps();

    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));
            let diff = _mm256_sub_ps(values, mean_values);
            total = _mm256_add_ps(total, _mm256_mul_ps(diff, diff));
        }
    }

    horizontal_sum(total)
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn matrix_mul_square(left: &[f32], right: &[f32], result: &mut [f32]) {
    debug_assert_eq!(left.len(), 16);
    debug_assert_eq!(right.len(), 16);
    debug_assert_eq!(result.len(), 16);

    for row in 0..4 {
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
