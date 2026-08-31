//! Sealed scalar contracts and scalar fallback kernels.

use super::arch;
use core::iter::Sum;
use core::ops::{Add, Div, Mul, Sub};

mod capability;
mod validation;

use capability::{
    native_vector_available, native_vector_chunk_len, native_wide_vector_available,
    uses_native_vector_path, uses_native_wide_vector_path,
};
use validation::scalar_matrix_shape;

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Native-precision scalar contract for SIMD-aware slice operations.
///
/// Implementations are sealed so native backend invariants stay under crate
/// control. The hidden methods form the monomorphized backend dispatch surface
/// used by the public functions in this module.
#[allow(private_bounds)]
pub trait SimdScalar:
    sealed::Sealed + Copy + Send + Sync + Add<Output = Self> + Mul<Output = Self> + Sum<Self> + 'static
{
    /// Additive identity.
    const ZERO: Self;

    #[doc(hidden)]
    #[inline]
    fn native_vector_available() -> bool {
        false
    }

    #[doc(hidden)]
    #[inline]
    fn uses_native_vector_path(len: usize) -> bool {
        let _ = len;
        false
    }

    #[doc(hidden)]
    #[inline]
    fn matrix_vector_path_available<const N: usize>() -> bool {
        let _ = N;
        false
    }

    #[doc(hidden)]
    #[inline]
    fn add_slices(left: &[Self], right: &[Self], result: &mut [Self]) {
        scalar_add(left, right, result);
    }

    #[doc(hidden)]
    #[inline]
    fn mul_slices(left: &[Self], right: &[Self], result: &mut [Self]) {
        scalar_mul(left, right, result);
    }

    #[doc(hidden)]
    #[inline]
    fn dot_slice(left: &[Self], right: &[Self]) -> Self {
        scalar_dot(left, right)
    }

    #[doc(hidden)]
    #[inline]
    fn sum_slice(data: &[Self]) -> Self {
        data.iter().copied().sum()
    }

    #[doc(hidden)]
    #[inline]
    fn matrix_mul_square<const N: usize>(left: &[Self], right: &[Self], result: &mut [Self]) {
        scalar_matrix_mul_square::<Self, N>(left, right, result);
    }
}

/// Scalar contract for native-precision real-valued statistics.
#[allow(private_bounds)]
pub trait SimdReal: SimdScalar + Sub<Output = Self> + Div<Output = Self> {
    /// Converts a non-zero slice length into the scalar's native representation.
    fn from_len(len: usize) -> Self;

    #[doc(hidden)]
    #[inline]
    fn mean_slice(data: &[Self]) -> Self {
        Self::sum_slice(data) / Self::from_len(data.len())
    }

    #[doc(hidden)]
    #[inline]
    fn variance_slice(data: &[Self]) -> Self {
        let mean = Self::mean_slice(data);
        data.iter()
            .copied()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<Self>()
            / Self::from_len(data.len())
    }
}

#[inline]
fn scalar_add<T: SimdScalar>(left: &[T], right: &[T], result: &mut [T]) {
    for ((left, right), output) in left.iter().zip(right.iter()).zip(result.iter_mut()) {
        *output = *left + *right;
    }
}

#[inline]
fn scalar_mul<T: SimdScalar>(left: &[T], right: &[T], result: &mut [T]) {
    for ((left, right), output) in left.iter().zip(right.iter()).zip(result.iter_mut()) {
        *output = *left * *right;
    }
}

#[inline]
fn scalar_dot<T: SimdScalar>(left: &[T], right: &[T]) -> T {
    left.iter()
        .copied()
        .zip(right.iter().copied())
        .fold(T::ZERO, |acc, (left, right)| acc + left * right)
}

#[inline]
fn scalar_matrix_mul_square<T: SimdScalar, const N: usize>(
    left: &[T],
    right: &[T],
    result: &mut [T],
) {
    assert!(N != 0, "matrix dimension must be non-zero");
    let expected = N.checked_mul(N).expect("matrix dimension overflow");
    assert_eq!(left.len(), expected, "left matrix size must equal N * N");
    assert_eq!(right.len(), expected, "right matrix size must equal N * N");
    assert_eq!(
        result.len(),
        expected,
        "result matrix size must equal N * N"
    );

    for row in 0..N {
        for col in 0..N {
            let mut acc = T::ZERO;
            for index in 0..N {
                acc = acc + left[row * N + index] * right[index * N + col];
            }
            result[row * N + col] = acc;
        }
    }
}

impl sealed::Sealed for f32 {}
impl SimdScalar for f32 {
    const ZERO: Self = 0.0;

    #[inline]
    fn native_vector_available() -> bool {
        native_vector_available()
    }

    #[inline]
    fn uses_native_vector_path(len: usize) -> bool {
        uses_native_vector_path(len)
    }

    #[inline]
    fn matrix_vector_path_available<const N: usize>() -> bool {
        N == 4 && native_vector_available()
    }

    #[inline]
    fn add_slices(left: &[Self], right: &[Self], result: &mut [Self]) {
        let len = left.len();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                // SAFETY: `chunk_len` is a LANES multiple <= len with the ISA
                // feature probed, so the sliced arguments satisfy the arch
                // fn contract.
                unsafe {
                    arch::add(
                        &left[..chunk_len],
                        &right[..chunk_len],
                        &mut result[..chunk_len],
                    );
                }
                if chunk_len < len {
                    scalar_add(
                        &left[chunk_len..],
                        &right[chunk_len..],
                        &mut result[chunk_len..],
                    );
                }
                return;
            }
        }
        scalar_add(left, right, result);
    }

    #[inline]
    fn mul_slices(left: &[Self], right: &[Self], result: &mut [Self]) {
        let len = left.len();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                // SAFETY: `chunk_len` is a LANES multiple <= len with the ISA
                // feature probed, so the sliced arguments satisfy the arch
                // fn contract.
                unsafe {
                    arch::mul(
                        &left[..chunk_len],
                        &right[..chunk_len],
                        &mut result[..chunk_len],
                    );
                }
                if chunk_len < len {
                    scalar_mul(
                        &left[chunk_len..],
                        &right[chunk_len..],
                        &mut result[chunk_len..],
                    );
                }
                return;
            }
        }
        scalar_mul(left, right, result);
    }

    #[inline]
    fn dot_slice(left: &[Self], right: &[Self]) -> Self {
        let len = left.len();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                let mut sum = unsafe { arch::dot(&left[..chunk_len], &right[..chunk_len]) };
                if chunk_len < len {
                    sum += scalar_dot(&left[chunk_len..], &right[chunk_len..]);
                }
                return sum;
            }
        }
        scalar_dot(left, right)
    }

    #[inline]
    fn sum_slice(data: &[Self]) -> Self {
        let len = data.len();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                let mut total = unsafe { arch::sum(&data[..chunk_len]) };
                if chunk_len < len {
                    total += data[chunk_len..].iter().copied().sum::<Self>();
                }
                return total;
            }
        }
        data.iter().copied().sum()
    }

    #[inline]
    fn matrix_mul_square<const N: usize>(left: &[Self], right: &[Self], result: &mut [Self]) {
        if N == 4 && native_vector_available() {
            scalar_matrix_shape::<N>(left, right, result);
            // SAFETY: N == 4 fixes the 16-element shape and the feature
            // probe passed, satisfying the arch fn contract.
            unsafe {
                arch::matrix_mul_square(left, right, result);
            }
        } else {
            scalar_matrix_mul_square::<Self, N>(left, right, result);
        }
    }
}

impl SimdReal for f32 {
    #[inline]
    fn from_len(len: usize) -> Self {
        len as Self
    }

    #[inline]
    fn variance_slice(data: &[Self]) -> Self {
        let len = data.len();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                let mean = Self::mean_slice(data);
                let mut total = unsafe { arch::squared_diff_sum(&data[..chunk_len], mean) };
                if chunk_len < len {
                    total += data[chunk_len..]
                        .iter()
                        .copied()
                        .map(|value| {
                            let diff = value - mean;
                            diff * diff
                        })
                        .sum::<Self>();
                }
                return total / Self::from_len(len);
            }
        }

        let mean = Self::mean_slice(data);
        data.iter()
            .copied()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<Self>()
            / Self::from_len(len)
    }
}

impl sealed::Sealed for f64 {}
impl SimdScalar for f64 {
    const ZERO: Self = 0.0;

    #[inline]
    fn native_vector_available() -> bool {
        native_wide_vector_available()
    }

    #[inline]
    fn uses_native_vector_path(len: usize) -> bool {
        uses_native_wide_vector_path(len)
    }

    #[inline]
    fn add_slices(left: &[Self], right: &[Self], result: &mut [Self]) {
        #[cfg(target_arch = "x86_64")]
        {
            let len = left.len();
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                // SAFETY: `chunk_len` is an 8-lane multiple <= len with AVX2
                // probed, so the sliced arguments satisfy the arch fn
                // contract.
                unsafe {
                    arch::add_wide(
                        &left[..chunk_len],
                        &right[..chunk_len],
                        &mut result[..chunk_len],
                    );
                }
                if chunk_len < len {
                    scalar_add(
                        &left[chunk_len..],
                        &right[chunk_len..],
                        &mut result[chunk_len..],
                    );
                }
                return;
            }
        }
        scalar_add(left, right, result);
    }

    #[inline]
    fn mul_slices(left: &[Self], right: &[Self], result: &mut [Self]) {
        #[cfg(target_arch = "x86_64")]
        {
            let len = left.len();
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                // SAFETY: `chunk_len` is an 8-lane multiple <= len with AVX2
                // probed, so the sliced arguments satisfy the arch fn
                // contract.
                unsafe {
                    arch::mul_wide(
                        &left[..chunk_len],
                        &right[..chunk_len],
                        &mut result[..chunk_len],
                    );
                }
                if chunk_len < len {
                    scalar_mul(
                        &left[chunk_len..],
                        &right[chunk_len..],
                        &mut result[chunk_len..],
                    );
                }
                return;
            }
        }
        scalar_mul(left, right, result);
    }

    #[inline]
    fn dot_slice(left: &[Self], right: &[Self]) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let len = left.len();
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                let mut sum = unsafe { arch::dot_wide(&left[..chunk_len], &right[..chunk_len]) };
                if chunk_len < len {
                    sum += scalar_dot(&left[chunk_len..], &right[chunk_len..]);
                }
                return sum;
            }
        }
        scalar_dot(left, right)
    }

    #[inline]
    fn sum_slice(data: &[Self]) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let len = data.len();
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                let mut total = unsafe { arch::sum_wide(&data[..chunk_len]) };
                if chunk_len < len {
                    total += data[chunk_len..].iter().copied().sum::<Self>();
                }
                return total;
            }
        }
        data.iter().copied().sum()
    }
}

impl SimdReal for f64 {
    #[inline]
    fn from_len(len: usize) -> Self {
        len as Self
    }

    #[inline]
    fn variance_slice(data: &[Self]) -> Self {
        let len = data.len();
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(chunk_len) = native_vector_chunk_len(len) {
                let mean = Self::mean_slice(data);
                let mut total = unsafe { arch::squared_diff_sum_wide(&data[..chunk_len], mean) };
                if chunk_len < len {
                    total += data[chunk_len..]
                        .iter()
                        .copied()
                        .map(|value| {
                            let diff = value - mean;
                            diff * diff
                        })
                        .sum::<Self>();
                }
                return total / Self::from_len(len);
            }
        }

        let mean = Self::mean_slice(data);
        data.iter()
            .copied()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<Self>()
            / Self::from_len(len)
    }
}

impl sealed::Sealed for i32 {}
impl SimdScalar for i32 {
    const ZERO: Self = 0;
}

impl sealed::Sealed for i64 {}
impl SimdScalar for i64 {
    const ZERO: Self = 0;
}

impl sealed::Sealed for u32 {}
impl SimdScalar for u32 {
    const ZERO: Self = 0;
}

impl sealed::Sealed for u64 {}
impl SimdScalar for u64 {
    const ZERO: Self = 0;
}

impl sealed::Sealed for usize {}
impl SimdScalar for usize {
    const ZERO: Self = 0;
}
