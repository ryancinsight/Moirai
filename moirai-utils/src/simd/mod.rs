//! Generic SIMD-aware vector operations.
//!
//! Public operations are expressed once over sealed scalar traits. The concrete
//! scalar type is selected at the call site, and monomorphization removes the
//! trait layer before execution. Native ISA kernels are private backend details.

mod arch;
mod scalar;

#[cfg(test)]
mod tests;

pub use arch::{has_avx2_support, has_neon_support};
pub use scalar::{SimdReal, SimdScalar};

#[inline]
fn record_dispatch(vectorized: bool, elements: usize) {
    if vectorized {
        crate::global_simd_counter().record_vectorized_op(elements);
    } else {
        crate::global_simd_counter().record_scalar_op(elements);
    }
}

#[inline]
fn assert_same_len<T>(left: &[T], right: &[T], result: &[T]) {
    assert_eq!(left.len(), right.len(), "input slices must match");
    assert_eq!(left.len(), result.len(), "output slice must match inputs");
}

/// Returns whether `T` has a native vector backend on the current CPU.
#[inline]
pub fn has_native_vector_path<T: SimdScalar>() -> bool {
    T::native_vector_available()
}

/// Adds two slices into `result`.
///
/// The operation executes in the native precision of `T`. For scalar types with
/// a private native vector backend, the backend is selected with static dispatch
/// after runtime ISA detection; all other scalar types use the monomorphized
/// scalar loop.
#[inline]
pub fn add<T: SimdScalar>(left: &[T], right: &[T], result: &mut [T]) {
    assert_same_len(left, right, result);
    let vectorized = T::uses_native_vector_path(left.len());
    T::add_slices(left, right, result);
    record_dispatch(vectorized, left.len());
}

/// Multiplies two slices into `result`.
#[inline]
pub fn mul<T: SimdScalar>(left: &[T], right: &[T], result: &mut [T]) {
    assert_same_len(left, right, result);
    let vectorized = T::uses_native_vector_path(left.len());
    T::mul_slices(left, right, result);
    record_dispatch(vectorized, left.len());
}

/// Computes a native-precision dot product.
#[inline]
pub fn dot<T: SimdScalar>(left: &[T], right: &[T]) -> T {
    assert_eq!(left.len(), right.len(), "input slices must match");
    let vectorized = T::uses_native_vector_path(left.len());
    let result = T::dot_slice(left, right);
    record_dispatch(vectorized, left.len());
    result
}

/// Computes a native-precision sum.
#[inline]
pub fn sum<T: SimdScalar>(data: &[T]) -> T {
    let vectorized = T::uses_native_vector_path(data.len());
    let result = T::sum_slice(data);
    record_dispatch(vectorized, data.len());
    result
}

/// Computes a native-precision arithmetic mean.
#[inline]
pub fn mean<T: SimdReal>(data: &[T]) -> T {
    assert!(!data.is_empty(), "mean requires at least one value");
    let vectorized = T::uses_native_vector_path(data.len());
    let result = T::mean_slice(data);
    record_dispatch(vectorized, data.len());
    result
}

/// Computes a native-precision population variance.
#[inline]
pub fn variance<T: SimdReal>(data: &[T]) -> T {
    assert!(!data.is_empty(), "variance requires at least one value");
    let vectorized = T::uses_native_vector_path(data.len());
    let result = T::variance_slice(data);
    record_dispatch(vectorized, data.len());
    result
}

/// Multiplies two row-major square matrices.
///
/// `N` is the matrix dimension. Each slice must contain exactly `N * N`
/// elements. The const parameter makes the structural arity visible to the
/// optimizer without encoding the dimension in the function name.
#[inline]
pub fn matrix_mul_square<T: SimdScalar, const N: usize>(left: &[T], right: &[T], result: &mut [T]) {
    T::matrix_mul_square::<N>(left, right, result);
    record_dispatch(T::matrix_vector_path_available::<N>(), result.len());
}
