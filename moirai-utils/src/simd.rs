//! SIMD-optimized vector operations for high-performance computing.
//!
//! This module provides vectorized implementations of common operations
//! used in task scheduling and data processing pipelines.

use core::arch::x86_64::*;

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

/// Safe wrapper for vectorized operations with fallback.
///
/// Automatically detects SIMD support and falls back to scalar operations
/// if SIMD is not available.
pub fn safe_vectorized_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    if has_avx2_support() && a.len() % 8 == 0 {
        unsafe {
            vectorized_add_f32(a, b, result);
        }
    } else {
        // Fallback to scalar implementation
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
}

/// Safe wrapper for vectorized multiplication with fallback.
pub fn safe_vectorized_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    if has_avx2_support() && a.len() % 8 == 0 {
        unsafe {
            vectorized_mul_f32(a, b, result);
        }
    } else {
        // Fallback to scalar implementation
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }
}

/// Safe wrapper for vectorized dot product with fallback.
pub fn safe_vectorized_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    if has_avx2_support() && a.len() % 8 == 0 {
        unsafe {
            vectorized_dot_product_f32(a, b)
        }
    } else {
        // Fallback to scalar implementation
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
        println!("AVX2 support detected: {}", has_avx2);
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
            
            assert_eq!(add_result, [9.0; 8]);
            assert_eq!(mul_result, [8.0, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0]);
            assert_eq!(dot_product, 120.0); // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
        }
    }
}