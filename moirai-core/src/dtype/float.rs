use super::Dtype;

/// Floating-point specific operations
pub trait FloatDtype: Dtype {
    /// Check if value is NaN
    fn is_nan(self) -> bool;

    /// Check if value is infinite
    fn is_infinite(self) -> bool;

    /// Check if value is finite
    fn is_finite(self) -> bool;

    /// Floor operation
    fn floor(self) -> Self;

    /// Ceiling operation
    fn ceil(self) -> Self;

    /// Round to nearest integer
    fn round(self) -> Self;

    /// Truncate to integer
    fn trunc(self) -> Self;

    /// Square root
    fn sqrt(self) -> Self;

    /// Natural logarithm
    fn ln(self) -> Self;

    /// Exponential function
    fn exp(self) -> Self;

    /// Power function
    fn powf(self, exp: Self) -> Self;

    /// Sine function
    fn sin(self) -> Self;

    /// Cosine function
    fn cos(self) -> Self;

    /// Epsilon for floating-point comparisons
    fn epsilon() -> Self;

    /// Compare with epsilon tolerance
    fn approx_eq(self, other: Self) -> bool {
        (self - other).abs() < Self::epsilon()
    }
}

// Implementations for floating-point types
macro_rules! impl_float_dtype {
    ($($t:ty),*) => {
        $(
            impl Dtype for $t {
                type Primitive = $t;

                const ZERO: Self = 0.0;
                const ONE: Self = 1.0;
                const MIN: Self = <$t>::MIN;
                const MAX: Self = <$t>::MAX;

                #[inline]
                fn from_primitive(value: Self::Primitive) -> Self {
                    value
                }

                #[inline]
                fn to_primitive(self) -> Self::Primitive {
                    self
                }

                #[inline]
                fn checked_add(self, other: Self) -> Option<Self> {
                    let result = self + other;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }

                #[inline]
                fn checked_sub(self, other: Self) -> Option<Self> {
                    let result = self - other;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }

                #[inline]
                fn checked_mul(self, other: Self) -> Option<Self> {
                    let result = self * other;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }

                #[inline]
                fn checked_div(self, other: Self) -> Option<Self> {
                    if other == Self::ZERO {
                        None
                    } else {
                        let result = self / other;
                        if result.is_finite() {
                            Some(result)
                        } else {
                            None
                        }
                    }
                }

                #[inline]
                fn saturating_add(self, other: Self) -> Self {
                    let result = self + other;
                    if result.is_finite() {
                        result
                    } else if result.is_infinite() && result > Self::ZERO {
                        Self::MAX
                    } else {
                        Self::MIN
                    }
                }

                #[inline]
                fn saturating_sub(self, other: Self) -> Self {
                    let result = self - other;
                    if result.is_finite() {
                        result
                    } else if result.is_infinite() && result > Self::ZERO {
                        Self::MAX
                    } else {
                        Self::MIN
                    }
                }

                #[inline]
                fn abs(self) -> Self {
                    if self < Self::ZERO {
                        -self
                    } else {
                        self
                    }
                }

                #[inline]
                fn to_f64(self) -> f64 {
                    f64::from(self)
                }

                #[inline]
                fn from_f64(value: f64) -> Option<Self> {
                    if std::mem::size_of::<Self>() == std::mem::size_of::<f64>() {
                        Some(value as Self)
                    } else {
                        if value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX) {
                            let result = value as Self;
                            if result.is_finite() {
                                Some(result)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                }
            }

            impl FloatDtype for $t {
                #[inline]
                fn is_nan(self) -> bool {
                    self.is_nan()
                }

                #[inline]
                fn is_infinite(self) -> bool {
                    self.is_infinite()
                }

                #[inline]
                fn is_finite(self) -> bool {
                    self.is_finite()
                }

                #[inline]
                fn floor(self) -> Self {
                    self.floor()
                }

                #[inline]
                fn ceil(self) -> Self {
                    self.ceil()
                }

                #[inline]
                fn round(self) -> Self {
                    self.round()
                }

                #[inline]
                fn trunc(self) -> Self {
                    self.trunc()
                }

                #[inline]
                fn sqrt(self) -> Self {
                    self.sqrt()
                }

                #[inline]
                fn ln(self) -> Self {
                    self.ln()
                }

                #[inline]
                fn exp(self) -> Self {
                    self.exp()
                }

                #[inline]
                fn powf(self, exp: Self) -> Self {
                    self.powf(exp)
                }

                #[inline]
                fn sin(self) -> Self {
                    self.sin()
                }

                #[inline]
                fn cos(self) -> Self {
                    self.cos()
                }

                fn epsilon() -> Self {
                    <$t>::EPSILON
                }
            }
        )*
    };
}

impl_float_dtype!(f32, f64);
