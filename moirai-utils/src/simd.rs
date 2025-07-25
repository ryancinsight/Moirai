//! SIMD-optimized vector operations for high-performance computing.
//!
//! This module provides vectorized implementations of common operations
//! used in task scheduling and data processing pipelines.

use core::arch::x86_64::*;

// ARM NEON support for broader platform compatibility
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// Import the feature detection macro
#[cfg(all(target_arch = "x86_64", feature = "std"))]
use std::is_x86_feature_detected;

/// SIMD-optimized vector addition for f32 arrays.
///
/// # Safety
/// This function uses unsafe SIMD intrinsics but is safe when:
/// - Input slices have the same length
/// - Length is a multiple of 8 (AVX2 requirement)
/// - Target CPU supports AVX2 instruction set
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
/// * `result` - Output vector (must be same length as inputs)
///
/// # Examples
/// ```
/// use moirai_utils::simd::vectorized_add_f32;
/// 
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let mut result = [0.0; 8];
/// 
/// unsafe {
///     vectorized_add_f32(&a, &b, &mut result);
/// }
/// 
/// assert_eq!(result, [9.0; 8]);
/// ```
#[target_feature(enable = "avx2")]
pub unsafe fn vectorized_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    assert_eq!(a.len() % 8, 0, "Length must be multiple of 8 for AVX2");
    
    let chunks = a.len() / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        
        // Load 8 f32 values using AVX2
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        
        // Perform vectorized addition
        let vresult = _mm256_add_ps(va, vb);
        
        // Store result
        _mm256_storeu_ps(result.as_mut_ptr().add(offset), vresult);
    }
}

/// SIMD-optimized vector multiplication for f32 arrays.
///
/// # Safety
/// Same safety requirements as `vectorized_add_f32`.
#[target_feature(enable = "avx2")]
pub unsafe fn vectorized_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    assert_eq!(a.len() % 8, 0, "Length must be multiple of 8 for AVX2");
    
    let chunks = a.len() / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vresult = _mm256_mul_ps(va, vb);
        
        _mm256_storeu_ps(result.as_mut_ptr().add(offset), vresult);
    }
}

/// SIMD-optimized dot product for f32 vectors.
///
/// # Safety
/// Same safety requirements as other SIMD functions.
#[target_feature(enable = "avx2")]
pub unsafe fn vectorized_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 8, 0, "Length must be multiple of 8 for AVX2");
    
    let chunks = a.len() / 8;
    let mut sum = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let offset = i * 8;
        
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vmul = _mm256_mul_ps(va, vb);
        
        sum = _mm256_add_ps(sum, vmul);
    }
    
    // Horizontal sum of the 8 f32 values
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum_combined = _mm_add_ps(sum_high, sum_low);
    
    let sum_shuffled = _mm_shuffle_ps(sum_combined, sum_combined, 0b01001110);
    let sum_2 = _mm_add_ps(sum_combined, sum_shuffled);
    let sum_final = _mm_add_ss(sum_2, _mm_shuffle_ps(sum_2, sum_2, 0b00000001));
    
    _mm_cvtss_f32(sum_final)
}

/// SIMD-optimized matrix multiplication for 4x4 f32 matrices.
///
/// # Safety
/// This function uses unsafe SIMD intrinsics for optimal performance.
/// The matrices must be stored in row-major order.
///
/// # Arguments
/// * `a` - First 4x4 matrix (16 elements)
/// * `b` - Second 4x4 matrix (16 elements)  
/// * `result` - Output 4x4 matrix (16 elements)
#[target_feature(enable = "avx2")]
pub unsafe fn vectorized_matrix_mul_4x4_f32(a: &[f32; 16], b: &[f32; 16], result: &mut [f32; 16]) {
    // Compute C = A * B for 4x4 row-major matrices
    // A[i][j] = a[i*4 + j], B[i][j] = b[i*4 + j], C[i][j] = result[i*4 + j]
    
    for i in 0..4 {
        // Load row i of matrix A
        let a_row = _mm_loadu_ps(&a[i * 4]);
        
        // Compute row i of result matrix
        for j in 0..4 {
            // Extract column j from matrix B
            let b_col = _mm_set_ps(b[12 + j], b[8 + j], b[4 + j], b[j]);
            
            // Compute dot product of a_row and b_col
            let dot = _mm_dp_ps(a_row, b_col, 0xF1);
            
            // Store result[i][j] = a_row · b_col
            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), dot);
            result[i * 4 + j] = temp[0];
        }
    }
}

/// SIMD-optimized vector sum reduction.
///
/// # Safety
/// Same safety requirements as other SIMD functions.
#[target_feature(enable = "avx2")]
pub unsafe fn vectorized_sum_f32(data: &[f32]) -> f32 {
    assert_eq!(data.len() % 8, 0, "Length must be multiple of 8 for AVX2");
    
    let chunks = data.len() / 8;
    let mut sum = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(data.as_ptr().add(offset));
        sum = _mm256_add_ps(sum, v);
    }
    
    // Horizontal sum
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum_combined = _mm_add_ps(sum_high, sum_low);
    
    let sum_shuffled = _mm_shuffle_ps(sum_combined, sum_combined, 0b01001110);
    let sum_2 = _mm_add_ps(sum_combined, sum_shuffled);
    let sum_final = _mm_add_ss(sum_2, _mm_shuffle_ps(sum_2, sum_2, 0b00000001));
    
    _mm_cvtss_f32(sum_final)
}

/// SIMD-optimized vector mean calculation.
///
/// # Safety
/// Same safety requirements as other SIMD functions.
#[target_feature(enable = "avx2")]
pub unsafe fn vectorized_mean_f32(data: &[f32]) -> f32 {
    let sum = vectorized_sum_f32(data);
    sum / data.len() as f32
}

/// SIMD-optimized vector variance calculation.
///
/// # Safety
/// Same safety requirements as other SIMD functions.
#[target_feature(enable = "avx2")]
pub unsafe fn vectorized_variance_f32(data: &[f32]) -> f32 {
    assert_eq!(data.len() % 8, 0, "Length must be multiple of 8 for AVX2");
    
    let mean = vectorized_mean_f32(data);
    let mean_vec = _mm256_set1_ps(mean);
    
    let chunks = data.len() / 8;
    let mut sum_sq_diff = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(data.as_ptr().add(offset));
        let diff = _mm256_sub_ps(v, mean_vec);
        let sq_diff = _mm256_mul_ps(diff, diff);
        sum_sq_diff = _mm256_add_ps(sum_sq_diff, sq_diff);
    }
    
    // Horizontal sum of squared differences
    let sum_high = _mm256_extractf128_ps(sum_sq_diff, 1);
    let sum_low = _mm256_castps256_ps128(sum_sq_diff);
    let sum_combined = _mm_add_ps(sum_high, sum_low);
    
    let sum_shuffled = _mm_shuffle_ps(sum_combined, sum_combined, 0b01001110);
    let sum_2 = _mm_add_ps(sum_combined, sum_shuffled);
    let sum_final = _mm_add_ss(sum_2, _mm_shuffle_ps(sum_2, sum_2, 0b00000001));
    
    _mm_cvtss_f32(sum_final) / data.len() as f32
}

/// ARM NEON optimized vector addition for f32 arrays.
///
/// # Safety
/// This function uses unsafe NEON intrinsics for ARM64 platforms.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vectorized_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    assert_eq!(a.len() % 4, 0, "Length must be multiple of 4 for NEON");
    
    let chunks = a.len() / 4;
    
    for i in 0..chunks {
        let offset = i * 4;
        
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let vresult = vaddq_f32(va, vb);
        
        vst1q_f32(result.as_mut_ptr().add(offset), vresult);
    }
}

/// ARM NEON optimized dot product for f32 vectors.
///
/// # Safety
/// This function uses unsafe NEON intrinsics for ARM64 platforms.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vectorized_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 4, 0, "Length must be multiple of 4 for NEON");
    
    let chunks = a.len() / 4;
    let mut sum = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let offset = i * 4;
        
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let vmul = vmulq_f32(va, vb);
        
        sum = vaddq_f32(sum, vmul);
    }
    
    // Horizontal sum
    let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let sum_final = vpadd_f32(sum_pair, sum_pair);
    
    vget_lane_f32(sum_final, 0)
}

/// ARM NEON-optimized 4x4 matrix multiplication.
///
/// # Safety
/// This function uses unsafe NEON intrinsics for ARM64 platforms.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vectorized_matrix_mul_4x4_f32(a: &[f32; 16], b: &[f32; 16], result: &mut [f32; 16]) {
    // Compute C = A * B for 4x4 row-major matrices
    // A[i][j] = a[i*4 + j], B[i][j] = b[i*4 + j], C[i][j] = result[i*4 + j]
    
    for i in 0..4 {
        // Load row i of matrix A
        let a_row = vld1q_f32(&a[i * 4]);
        
        // Compute row i of result matrix
        for j in 0..4 {
            // Extract column j from matrix B and compute dot product
            let b_col = [b[j], b[4 + j], b[8 + j], b[12 + j]];
            let b_col_v = vld1q_f32(b_col.as_ptr());
            
            // Compute dot product of a_row and b_col
            let mul = vmulq_f32(a_row, b_col_v);
            let sum_pair = vpadd_f32(vget_low_f32(mul), vget_high_f32(mul));
            let final_sum = vpadd_f32(sum_pair, sum_pair);
            
            // Store result[i][j] = a_row · b_col
            result[i * 4 + j] = vget_lane_f32(final_sum, 0);
        }
    }
}

/// ARM NEON-optimized vector sum.
///
/// # Safety
/// This function uses unsafe NEON intrinsics for ARM64 platforms.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vectorized_sum_f32(data: &[f32]) -> f32 {
    assert_eq!(data.len() % 4, 0, "Length must be multiple of 4 for NEON");
    
    let chunks = data.len() / 4;
    let mut sum = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let offset = i * 4;
        let v = vld1q_f32(data.as_ptr().add(offset));
        sum = vaddq_f32(sum, v);
    }
    
    // Horizontal sum
    let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let final_sum = vpadd_f32(sum_pair, sum_pair);
    vget_lane_f32(final_sum, 0)
}

/// ARM NEON-optimized vector mean.
///
/// # Safety
/// This function uses unsafe NEON intrinsics for ARM64 platforms.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vectorized_mean_f32(data: &[f32]) -> f32 {
    neon_vectorized_sum_f32(data) / data.len() as f32
}

/// ARM NEON-optimized vector variance.
///
/// # Safety
/// This function uses unsafe NEON intrinsics for ARM64 platforms.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_vectorized_variance_f32(data: &[f32]) -> f32 {
    assert_eq!(data.len() % 4, 0, "Length must be multiple of 4 for NEON");
    
    let mean = neon_vectorized_mean_f32(data);
    let mean_v = vdupq_n_f32(mean);
    
    let chunks = data.len() / 4;
    let mut sum_sq_diff = vdupq_n_f32(0.0);
    
    for i in 0..chunks {
        let offset = i * 4;
        let v = vld1q_f32(data.as_ptr().add(offset));
        let diff = vsubq_f32(v, mean_v);
        let sq_diff = vmulq_f32(diff, diff);
        sum_sq_diff = vaddq_f32(sum_sq_diff, sq_diff);
    }
    
    // Horizontal sum
    let sum_pair = vpadd_f32(vget_low_f32(sum_sq_diff), vget_high_f32(sum_sq_diff));
    let final_sum = vpadd_f32(sum_pair, sum_pair);
    let total_sq_diff = vget_lane_f32(final_sum, 0);
    
    total_sq_diff / data.len() as f32
}

/// Runtime detection of SIMD capabilities.
///
/// Returns true if the current CPU supports AVX2 instructions.
pub fn has_avx2_support() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
    {
        false
    }
}

/// Runtime detection of ARM NEON capabilities.
///
/// Returns true if the current CPU supports NEON instructions.
pub fn has_neon_support() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is mandatory on AArch64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Safe wrapper for vectorized operations with fallback.
///
/// Automatically detects SIMD support and falls back to scalar operations
/// if SIMD is not available. Records performance metrics.
pub fn safe_vectorized_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_support() && a.len() % 8 == 0 {
            unsafe {
                vectorized_add_f32(a, b, result);
            }
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(a.len());
            return;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon_support() && a.len() % 4 == 0 {
            unsafe {
                neon_vectorized_add_f32(a, b, result);
            }
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(a.len());
            return;
        }
    }
    
    // Fallback to scalar implementation
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
    #[cfg(feature = "std")]
    crate::global_simd_counter().record_scalar_op(a.len());
}

/// Safe wrapper for vectorized multiplication with fallback.
pub fn safe_vectorized_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    if has_avx2_support() && a.len() % 8 == 0 {
        unsafe {
            vectorized_mul_f32(a, b, result);
        }
        #[cfg(feature = "std")]
        crate::global_simd_counter().record_vectorized_op(a.len());
    } else {
        // Fallback to scalar implementation
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
        #[cfg(feature = "std")]
        crate::global_simd_counter().record_scalar_op(a.len());
    }
}

/// Safe wrapper for vectorized dot product with fallback.
pub fn safe_vectorized_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_support() && a.len() % 8 == 0 {
            let result = unsafe {
                vectorized_dot_product_f32(a, b)
            };
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(a.len());
            return result;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon_support() && a.len() % 4 == 0 {
            let result = unsafe {
                neon_vectorized_dot_product_f32(a, b)
            };
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(a.len());
            return result;
        }
    }
    
    // Fallback to scalar implementation
    #[cfg(feature = "std")]
    crate::global_simd_counter().record_scalar_op(a.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Safe wrapper for matrix multiplication with fallback.
pub fn safe_vectorized_matrix_mul_4x4_f32(a: &[f32; 16], b: &[f32; 16], result: &mut [f32; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_support() {
            unsafe {
                vectorized_matrix_mul_4x4_f32(a, b, result);
            }
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(16);
            return;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon_support() {
            unsafe {
                neon_vectorized_matrix_mul_4x4_f32(a, b, result);
            }
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(16);
            return;
        }
    }
    
    // Fallback to scalar implementation
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[i * 4 + k] * b[k * 4 + j];
            }
            result[i * 4 + j] = sum;
        }
    }
    #[cfg(feature = "std")]
    crate::global_simd_counter().record_scalar_op(16);
}

/// Safe wrapper for vector sum with fallback.
pub fn safe_vectorized_sum_f32(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_support() && data.len() % 8 == 0 {
            let result = unsafe {
                vectorized_sum_f32(data)
            };
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(data.len());
            return result;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon_support() && data.len() % 4 == 0 {
            let result = unsafe {
                neon_vectorized_sum_f32(data)
            };
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(data.len());
            return result;
        }
    }
    
    // Fallback to scalar implementation
    #[cfg(feature = "std")]
    crate::global_simd_counter().record_scalar_op(data.len());
    data.iter().sum()
}

/// Safe wrapper for vector mean with fallback.
pub fn safe_vectorized_mean_f32(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_support() && data.len() % 8 == 0 {
            let result = unsafe {
                vectorized_mean_f32(data)
            };
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(data.len());
            return result;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon_support() && data.len() % 4 == 0 {
            let result = unsafe {
                neon_vectorized_mean_f32(data)
            };
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(data.len());
            return result;
        }
    }
    
    // Fallback to scalar implementation
    #[cfg(feature = "std")]
    crate::global_simd_counter().record_scalar_op(data.len());
    data.iter().sum::<f32>() / data.len() as f32
}

/// Safe wrapper for vector variance with fallback.
pub fn safe_vectorized_variance_f32(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_support() && data.len() % 8 == 0 {
            let result = unsafe {
                vectorized_variance_f32(data)
            };
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(data.len());
            return result;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if has_neon_support() && data.len() % 4 == 0 {
            let result = unsafe {
                neon_vectorized_variance_f32(data)
            };
            #[cfg(feature = "std")]
            crate::global_simd_counter().record_vectorized_op(data.len());
            return result;
        }
    }
    
    // Fallback to scalar implementation
    #[cfg(feature = "std")]
    crate::global_simd_counter().record_scalar_op(data.len());
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use std::println;
    
    #[test]
    fn test_simd_detection() {
        // This test will pass regardless of CPU capabilities
        let has_avx2 = has_avx2_support();
        let has_neon = has_neon_support();
        println!("AVX2 support detected: {}", has_avx2);
        println!("NEON support detected: {}", has_neon);
    }
    
    #[test]
    fn test_safe_vectorized_add() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = [0.0; 8];
        
        safe_vectorized_add_f32(&a, &b, &mut result);
        
        assert_eq!(result, [9.0; 8]);
    }
    
    #[test]
    fn test_safe_vectorized_add_unaligned() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut result = [0.0; 3];
        
        safe_vectorized_add_f32(&a, &b, &mut result);
        
        assert_eq!(result, [5.0, 7.0, 9.0]);
    }
    
    #[test]
    fn test_safe_vectorized_mul() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut result = [0.0; 8];
        
        safe_vectorized_mul_f32(&a, &b, &mut result);
        
        assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }
    
    #[test]
    fn test_safe_vectorized_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        
        let result = safe_vectorized_dot_product_f32(&a, &b);
        
        // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
        assert_eq!(result, 120.0);
    }
    
    #[test]
    fn test_safe_vectorized_matrix_mul() {
        // Test 1: Identity matrix (original test)
        let a = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut result = [0.0; 16];
        
        safe_vectorized_matrix_mul_4x4_f32(&a, &identity, &mut result);
        assert_eq!(result, a);
        
        // Test 2: Non-symmetrical matrices to verify correct A * B computation
        let a_test = [
            1.0, 2.0, 3.0, 4.0,    // [1 2 3 4]
            5.0, 6.0, 7.0, 8.0,    // [5 6 7 8]
            9.0, 10.0, 11.0, 12.0, // [9 10 11 12]
            13.0, 14.0, 15.0, 16.0, // [13 14 15 16]
        ];
        let b_test = [
            1.0, 0.0, 0.0, 1.0,    // [1 0 0 1]
            0.0, 1.0, 0.0, 2.0,    // [0 1 0 2]
            0.0, 0.0, 1.0, 3.0,    // [0 0 1 3]
            1.0, 0.0, 0.0, 4.0,    // [1 0 0 4]
        ];
        
        let expected = [
            // Row 0: [1,2,3,4] * B = [1*1+2*0+3*0+4*1, 1*0+2*1+3*0+4*0, 1*0+2*0+3*1+4*0, 1*1+2*2+3*3+4*4]
            5.0, 2.0, 3.0, 30.0,   // [5, 2, 3, 30]
            // Row 1: [5,6,7,8] * B = [5*1+6*0+7*0+8*1, 5*0+6*1+7*0+8*0, 5*0+6*0+7*1+8*0, 5*1+6*2+7*3+8*4]
            13.0, 6.0, 7.0, 70.0,  // [13, 6, 7, 70]
            // Row 2: [9,10,11,12] * B = [9*1+10*0+11*0+12*1, 9*0+10*1+11*0+12*0, 9*0+10*0+11*1+12*0, 9*1+10*2+11*3+12*4]
            21.0, 10.0, 11.0, 110.0, // [21, 10, 11, 110]
            // Row 3: [13,14,15,16] * B = [13*1+14*0+15*0+16*1, 13*0+14*1+15*0+16*0, 13*0+14*0+15*1+16*0, 13*1+14*2+15*3+16*4]
            29.0, 14.0, 15.0, 150.0, // [29, 14, 15, 150]
        ];
        
        safe_vectorized_matrix_mul_4x4_f32(&a_test, &b_test, &mut result);
        
        // Verify each element with tolerance for floating point precision
        for i in 0..16 {
            assert!((result[i] - expected[i]).abs() < 1e-5, 
                   "Mismatch at index {}: expected {}, got {}", i, expected[i], result[i]);
        }
        
        // Test 3: Verify non-commutativity (A*B != B*A for general matrices)
        let mut result_ba = [0.0; 16];
        safe_vectorized_matrix_mul_4x4_f32(&b_test, &a_test, &mut result_ba);
        
        // Should be different from A*B (unless matrices commute)
        let mut different = false;
        for i in 0..16 {
            if (result[i] - result_ba[i]).abs() > 1e-5 {
                different = true;
                break;
            }
        }
        assert!(different, "A*B should not equal B*A for non-commuting matrices");
    }
    
    #[test]
    fn test_safe_vectorized_statistics() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let sum = safe_vectorized_sum_f32(&data);
        let mean = safe_vectorized_mean_f32(&data);
        let variance = safe_vectorized_variance_f32(&data);
        
        assert_eq!(sum, 36.0);
        assert_eq!(mean, 4.5);
        assert!((variance - 5.25).abs() < 0.001); // Variance of 1..8 is 5.25
    }
    
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn test_vectorized_operations() {
        if !has_avx2_support() {
            println!("Skipping SIMD tests - AVX2 not supported");
            return;
        }
        
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut add_result = [0.0; 8];
        let mut mul_result = [0.0; 8];
        
        unsafe {
            vectorized_add_f32(&a, &b, &mut add_result);
            vectorized_mul_f32(&a, &b, &mut mul_result);
            
            let dot_product = vectorized_dot_product_f32(&a, &b);
            let sum = vectorized_sum_f32(&a);
            let mean = vectorized_mean_f32(&a);
            let variance = vectorized_variance_f32(&a);
            
            assert_eq!(add_result, [9.0; 8]);
            assert_eq!(mul_result, [8.0, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0]);
            assert_eq!(dot_product, 120.0); // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
            assert_eq!(sum, 36.0);
            assert_eq!(mean, 4.5);
            assert!((variance - 5.25).abs() < 0.001);
        }
    }
    
    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    fn test_neon_operations() {
        if !has_neon_support() {
            println!("Skipping NEON tests - NEON not supported");
            return;
        }
        
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let mut add_result = [0.0; 4];
        
        unsafe {
            neon_vectorized_add_f32(&a, &b, &mut add_result);
            let dot_product = neon_vectorized_dot_product_f32(&a, &b);
            
            assert_eq!(add_result, [5.0, 5.0, 5.0, 5.0]);
            assert_eq!(dot_product, 20.0); // 1*4 + 2*3 + 3*2 + 4*1
        }
    }
}