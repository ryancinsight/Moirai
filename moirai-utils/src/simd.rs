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
    // Load rows of matrix A
    let row0 = _mm_loadu_ps(&a[0]);
    let row1 = _mm_loadu_ps(&a[4]);
    let row2 = _mm_loadu_ps(&a[8]);
    let row3 = _mm_loadu_ps(&a[12]);
    
    // Process each column of matrix B
    for col in 0..4 {
        let b_col = _mm_set_ps(b[12 + col], b[8 + col], b[4 + col], b[col]);
        
        let mut res_col = _mm_mul_ps(_mm_shuffle_ps(b_col, b_col, 0x00), row0);
        res_col = _mm_add_ps(res_col, _mm_mul_ps(_mm_shuffle_ps(b_col, b_col, 0x55), row1));
        res_col = _mm_add_ps(res_col, _mm_mul_ps(_mm_shuffle_ps(b_col, b_col, 0xAA), row2));
        res_col = _mm_add_ps(res_col, _mm_mul_ps(_mm_shuffle_ps(b_col, b_col, 0xFF), row3));
        
        _mm_storeu_ps(&mut result[col * 4], res_col);
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
    if has_avx2_support() {
        unsafe {
            vectorized_matrix_mul_4x4_f32(a, b, result);
        }
    } else {
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
    }
}

/// Safe wrapper for vector sum with fallback.
pub fn safe_vectorized_sum_f32(data: &[f32]) -> f32 {
    if has_avx2_support() && data.len() % 8 == 0 {
        unsafe {
            vectorized_sum_f32(data)
        }
    } else {
        data.iter().sum()
    }
}

/// Safe wrapper for vector mean with fallback.
pub fn safe_vectorized_mean_f32(data: &[f32]) -> f32 {
    if has_avx2_support() && data.len() % 8 == 0 {
        unsafe {
            vectorized_mean_f32(data)
        }
    } else {
        data.iter().sum::<f32>() / data.len() as f32
    }
}

/// Safe wrapper for vector variance with fallback.
pub fn safe_vectorized_variance_f32(data: &[f32]) -> f32 {
    if has_avx2_support() && data.len() % 8 == 0 {
        unsafe {
            vectorized_variance_f32(data)
        }
    } else {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
    }
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
        let a = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let b = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]; // Identity matrix
        let mut result = [0.0; 16];
        
        safe_vectorized_matrix_mul_4x4_f32(&a, &b, &mut result);
        
        // Result should be the same as matrix a
        assert_eq!(result, a);
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