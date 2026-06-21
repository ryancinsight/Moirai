use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// Unified data type trait for all numeric operations in Moirai.
///
/// This trait consolidates the scattered usage of primitive types throughout
/// the codebase, providing a single abstraction for numeric computations.
/// All algorithms should be generic over Dtype rather than hardcoded to
/// specific primitive types.
///
/// # Safety Guarantees
/// - All operations are checked for overflow/underflow
/// - NaN and infinity handling for floating point types
/// - Consistent behavior across integer and floating point types
pub trait Dtype:
    Copy
    + Clone
    + Debug
    + Display
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialEq
    + PartialOrd
    + Default
{
    /// The underlying primitive type
    type Primitive;

    /// Zero value for this type
    const ZERO: Self;

    /// One value for this type  
    const ONE: Self;

    /// Minimum representable value
    const MIN: Self;

    /// Maximum representable value
    const MAX: Self;

    /// Create from primitive value
    fn from_primitive(value: Self::Primitive) -> Self;

    /// Convert to primitive value
    fn to_primitive(self) -> Self::Primitive;

    /// Checked addition that returns None on overflow
    fn checked_add(self, other: Self) -> Option<Self>;

    /// Checked subtraction that returns None on underflow
    fn checked_sub(self, other: Self) -> Option<Self>;

    /// Checked multiplication that returns None on overflow
    fn checked_mul(self, other: Self) -> Option<Self>;

    /// Checked division that returns None on division by zero
    fn checked_div(self, other: Self) -> Option<Self>;

    /// Saturating addition (clamps to max on overflow)
    fn saturating_add(self, other: Self) -> Self;

    /// Saturating subtraction (clamps to min on underflow)
    fn saturating_sub(self, other: Self) -> Self;

    /// Absolute value
    fn abs(self) -> Self;

    /// Check if value is zero
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }

    /// Check if value is positive
    fn is_positive(self) -> bool {
        self > Self::ZERO
    }

    /// Check if value is negative  
    fn is_negative(self) -> bool {
        self < Self::ZERO
    }

    /// Convert to f64 for high-precision calculations
    fn to_f64(self) -> f64;

    /// Create from f64 (may lose precision)
    fn from_f64(value: f64) -> Option<Self>;
}
