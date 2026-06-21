use super::Dtype;

/// Integer-specific operations
pub trait IntegerDtype: Dtype {
    /// Check if value is even
    fn is_even(self) -> bool;

    /// Check if value is odd
    fn is_odd(self) -> bool;

    /// Bit count (population count)
    fn count_ones(self) -> u32;

    /// Leading zeros
    fn leading_zeros(self) -> u32;

    /// Trailing zeros
    fn trailing_zeros(self) -> u32;
}

// Implementations for common integer types
macro_rules! impl_integer_dtype {
    ($($t:ty),*) => {
        $(
            #[allow(clippy::cast_precision_loss)]
            #[allow(clippy::cast_possible_truncation)]
            impl Dtype for $t {
                type Primitive = $t;

                const ZERO: Self = 0;
                const ONE: Self = 1;
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
                    self.checked_add(other)
                }

                #[inline]
                fn checked_sub(self, other: Self) -> Option<Self> {
                    self.checked_sub(other)
                }

                #[inline]
                fn checked_mul(self, other: Self) -> Option<Self> {
                    self.checked_mul(other)
                }

                #[inline]
                fn checked_div(self, other: Self) -> Option<Self> {
                    self.checked_div(other)
                }

                #[inline]
                fn saturating_add(self, other: Self) -> Self {
                    self.saturating_add(other)
                }

                #[inline]
                fn saturating_sub(self, other: Self) -> Self {
                    self.saturating_sub(other)
                }

                #[inline]
                fn abs(self) -> Self {
                    if self < Self::ZERO {
                        self.wrapping_neg()
                    } else {
                        self
                    }
                }

                #[inline]
                fn to_f64(self) -> f64 {
                    #[allow(clippy::cast_lossless)]
                    match std::mem::size_of::<Self>() {
                        1 | 2 | 4 => {
                            self as f64
                        }
                        _ => {
                            self as f64
                        }
                    }
                }

                #[inline]
                fn from_f64(value: f64) -> Option<Self> {
                    #[allow(clippy::cast_lossless)]
                    let (min_val, max_val) = match std::mem::size_of::<Self>() {
                        1 | 2 | 4 => {
                            (Self::MIN as f64, Self::MAX as f64)
                        }
                        _ => {
                            (Self::MIN as f64, Self::MAX as f64)
                        }
                    };

                    if value >= min_val && value <= max_val {
                        Some(value as Self)
                    } else {
                        None
                    }
                }
            }

            impl IntegerDtype for $t {
                #[inline]
                fn is_even(self) -> bool {
                    self % 2 == 0
                }

                #[inline]
                fn is_odd(self) -> bool {
                    self % 2 != 0
                }

                #[inline]
                fn count_ones(self) -> u32 {
                    self.count_ones()
                }

                #[inline]
                fn leading_zeros(self) -> u32 {
                    self.leading_zeros()
                }

                #[inline]
                fn trailing_zeros(self) -> u32 {
                    self.trailing_zeros()
                }
            }
        )*
    };
}

impl_integer_dtype!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
