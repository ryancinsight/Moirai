//! Bit manipulation utilities for high-performance computing.
//!
//! This module provides efficient bit manipulation functions commonly used
//! in performance-critical code, particularly for power-of-two calculations
//! and bit counting operations.

/// Count the number of set bits in a u64.
///
/// This function returns the population count (number of 1 bits) in the
/// given 64-bit unsigned integer.
#[must_use]
pub const fn popcount_u64(x: u64) -> u32 {
    x.count_ones()
}

/// Find the position of the least significant set bit.
///
/// Returns the number of trailing zeros, which is equivalent to the
/// position of the least significant set bit. Returns 64 if no bits are set.
#[must_use]
pub const fn trailing_zeros_u64(x: u64) -> u32 {
    x.trailing_zeros()
}

/// Find the position of the most significant set bit.
///
/// Returns the number of leading zeros. Combined with the bit width,
/// this gives the position of the most significant set bit.
/// Returns 64 if no bits are set.
#[must_use]
pub const fn leading_zeros_u64(x: u64) -> u32 {
    x.leading_zeros()
}

/// Check if a number is a power of 2.
///
/// Uses the classic bit manipulation trick: a power of 2 has exactly
/// one bit set, so (x & (x-1)) equals 0 for powers of 2.
#[must_use]
pub const fn is_power_of_two(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

/// Round up to the next power of 2.
///
/// If the input is already a power of 2, returns the input unchanged.
/// For 0, returns 1.
#[must_use]
pub const fn next_power_of_two(x: u64) -> u64 {
    if x <= 1 {
        1
    } else {
        1 << (64 - (x - 1).leading_zeros())
    }
}

/// Extract the lowest set bit.
///
/// Returns a value with only the lowest set bit of the input.
/// For example, isolate_lowest_bit(0b1100) returns 0b0100.
#[must_use]
pub const fn isolate_lowest_bit(x: u64) -> u64 {
    x & x.wrapping_neg()
}

/// Clear the lowest set bit.
///
/// Returns the input with its lowest set bit cleared.
/// For example, clear_lowest_bit(0b1100) returns 0b1000.
#[must_use]
pub const fn clear_lowest_bit(x: u64) -> u64 {
    x & (x - 1)
}

/// Check if exactly one bit is set.
///
/// Returns true if the input has exactly one bit set (is a power of 2),
/// false otherwise.
#[must_use]
pub const fn is_single_bit(x: u64) -> bool {
    is_power_of_two(x)
}

/// Calculate log2 of a power of 2.
///
/// Returns the base-2 logarithm of the input, which must be a power of 2.
/// For non-powers of 2, the behavior is undefined.
#[must_use]
pub const fn log2_power_of_two(x: u64) -> u32 {
    63 - x.leading_zeros()
}

/// Reverse the bits in a u64.
///
/// Returns the input with all bits reversed.
#[must_use]
pub const fn reverse_bits(x: u64) -> u64 {
    x.reverse_bits()
}

/// Swap bytes in a u64 (change endianness).
///
/// Returns the input with bytes swapped.
#[must_use]
pub const fn swap_bytes(x: u64) -> u64 {
    x.swap_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcount() {
        assert_eq!(popcount_u64(0), 0);
        assert_eq!(popcount_u64(1), 1);
        assert_eq!(popcount_u64(0b1111), 4);
        assert_eq!(popcount_u64(u64::MAX), 64);
    }

    #[test]
    fn test_trailing_zeros() {
        assert_eq!(trailing_zeros_u64(1), 0);
        assert_eq!(trailing_zeros_u64(2), 1);
        assert_eq!(trailing_zeros_u64(8), 3);
        assert_eq!(trailing_zeros_u64(0), 64);
    }

    #[test]
    fn test_leading_zeros() {
        assert_eq!(leading_zeros_u64(1), 63);
        assert_eq!(leading_zeros_u64(2), 62);
        assert_eq!(leading_zeros_u64(0), 64);
        assert_eq!(leading_zeros_u64(u64::MAX), 0);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(15));
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(15), 16);
        assert_eq!(next_power_of_two(16), 16);
    }

    #[test]
    fn test_isolate_lowest_bit() {
        assert_eq!(isolate_lowest_bit(0b1100), 0b0100);
        assert_eq!(isolate_lowest_bit(0b1010), 0b0010);
        assert_eq!(isolate_lowest_bit(0b1000), 0b1000);
        assert_eq!(isolate_lowest_bit(0), 0);
    }

    #[test]
    fn test_clear_lowest_bit() {
        assert_eq!(clear_lowest_bit(0b1100), 0b1000);
        assert_eq!(clear_lowest_bit(0b1010), 0b1000);
        assert_eq!(clear_lowest_bit(0b0001), 0b0000);
    }

    #[test]
    fn test_log2_power_of_two() {
        assert_eq!(log2_power_of_two(1), 0);
        assert_eq!(log2_power_of_two(2), 1);
        assert_eq!(log2_power_of_two(4), 2);
        assert_eq!(log2_power_of_two(8), 3);
        assert_eq!(log2_power_of_two(1024), 10);
    }
}
