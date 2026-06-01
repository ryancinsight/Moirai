//! x86 native vector backend.

use core::arch::x86_64::*;

// Backend block width. Single-precision uses one AVX register; the wide-real
// path uses two four-lane registers per block to keep one prefix/tail contract.
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

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add_wide(left: &[f64], right: &[f64], result: &mut [f64]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
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

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn mul_wide(left: &[f64], right: &[f64], result: &mut [f64]) {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), result.len());
    debug_assert_eq!(left.len() % LANES, 0);

    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
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

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_wide(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len() % LANES, 0);

    let mut total_v0 = _mm256_setzero_pd();
    let mut total_v1 = _mm256_setzero_pd();
    for lane in 0..left.len() / LANES {
        let offset = lane * LANES;
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

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sum_wide(data: &[f64]) -> f64 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mut total_v0 = _mm256_setzero_pd();
    let mut total_v1 = _mm256_setzero_pd();
    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
        unsafe {
            let values_v0 = _mm256_loadu_pd(data.as_ptr().add(offset));
            let values_v1 = _mm256_loadu_pd(data.as_ptr().add(offset + 4));
            total_v0 = _mm256_add_pd(total_v0, values_v0);
            total_v1 = _mm256_add_pd(total_v1, values_v1);
        }
    }

    horizontal_sum_wide(_mm256_add_pd(total_v0, total_v1))
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn squared_diff_sum_wide(data: &[f64], mean: f64) -> f64 {
    debug_assert_eq!(data.len() % LANES, 0);

    let mean_values = _mm256_set1_pd(mean);
    let mut total_v0 = _mm256_setzero_pd();
    let mut total_v1 = _mm256_setzero_pd();

    for lane in 0..data.len() / LANES {
        let offset = lane * LANES;
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
