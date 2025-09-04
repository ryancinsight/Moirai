//! Fast pseudo-random number generation for performance-critical scenarios.
//!
//! This module provides lightweight random number generation using the XorShift
//! algorithm, optimized for speed rather than cryptographic security.

/// Xorshift-based random number generator.
///
/// This is a fast, lightweight pseudo-random number generator suitable for
/// performance-critical scenarios where cryptographic security is not required.
/// The XorShift algorithm provides good statistical properties with minimal
/// computational overhead.
#[derive(Debug, Clone)]
pub struct XorshiftRng {
    state: u64,
}

impl XorshiftRng {
    /// Create a new random number generator with a seed.
    ///
    /// # Arguments
    /// * `seed` - The initial seed value. If 0, it will be changed to 1.
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Create a new random number generator with a default seed.
    ///
    /// In std environments, this uses the current system time as a seed.
    /// In no-std environments, this uses a fixed seed.
    pub fn default_seed() -> Self {
        #[cfg(feature = "std")]
        {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::time::SystemTime;

            let mut hasher = DefaultHasher::new();
            SystemTime::now().hash(&mut hasher);
            Self::new(hasher.finish())
        }

        #[cfg(not(feature = "std"))]
        {
            Self::new(0x123456789abcdef0)
        }
    }

    /// Generate the next random number.
    ///
    /// Uses XorShift algorithm for fast pseudo-random number generation.
    /// This method advances the internal state and returns the new value.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate a random number in the range [0, max).
    ///
    /// # Arguments
    /// * `max` - The exclusive upper bound for the random number
    ///
    /// # Returns
    /// A random number in the range [0, max), or 0 if max is 0
    pub fn next_range(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        self.next_u64() % max
    }

    /// Generate a random boolean.
    ///
    /// # Returns
    /// A random boolean value with 50% probability for each outcome
    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Generate a random f64 in the range [0.0, 1.0).
    ///
    /// # Returns
    /// A random floating-point number with uniform distribution in [0.0, 1.0)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a random usize value.
    ///
    /// # Returns
    /// A random usize value using the full range of the type
    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    /// Generate a random number in the range [min, max).
    ///
    /// # Arguments
    /// * `min` - The inclusive lower bound
    /// * `max` - The exclusive upper bound
    ///
    /// # Returns
    /// A random number in the specified range, or min if max <= min
    pub fn next_range_bounds(&mut self, min: u64, max: u64) -> u64 {
        if max <= min {
            return min;
        }
        min + self.next_range(max - min)
    }

    /// Reseed the random number generator.
    ///
    /// # Arguments
    /// * `new_seed` - The new seed value
    pub fn reseed(&mut self, new_seed: u64) {
        self.state = if new_seed == 0 { 1 } else { new_seed };
    }

    /// Get the current state of the generator.
    ///
    /// This can be useful for debugging or saving/restoring state.
    pub fn state(&self) -> u64 {
        self.state
    }
}

impl Default for XorshiftRng {
    fn default() -> Self {
        Self::default_seed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_basic() {
        let mut rng = XorshiftRng::new(12345);
        
        let a = rng.next_u64();
        let b = rng.next_u64();
        
        // Values should be different
        assert_ne!(a, b);
        
        // Should not be zero (very unlikely)
        assert_ne!(a, 0);
        assert_ne!(b, 0);
    }

    #[test]
    fn test_xorshift_range() {
        let mut rng = XorshiftRng::new(54321);
        
        for _ in 0..100 {
            let val = rng.next_range(10);
            assert!(val < 10);
        }
        
        // Test edge case
        assert_eq!(rng.next_range(0), 0);
    }

    #[test]
    fn test_xorshift_bool() {
        let mut rng = XorshiftRng::new(98765);
        
        let mut true_count = 0;
        let mut false_count = 0;
        
        for _ in 0..1000 {
            if rng.next_bool() {
                true_count += 1;
            } else {
                false_count += 1;
            }
        }
        
        // Should have roughly equal distribution (within reason)
        assert!(true_count > 300 && true_count < 700);
        assert!(false_count > 300 && false_count < 700);
    }

    #[test]
    fn test_xorshift_f64() {
        let mut rng = XorshiftRng::new(13579);
        
        for _ in 0..100 {
            let val = rng.next_f64();
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = XorshiftRng::new(42);
        let mut rng2 = XorshiftRng::new(42);
        
        // Same seed should produce same sequence
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_reseed() {
        let mut rng = XorshiftRng::new(1);
        let initial_state = rng.state();
        
        rng.next_u64(); // Change state
        assert_ne!(rng.state(), initial_state);
        
        rng.reseed(1);
        assert_eq!(rng.state(), initial_state);
    }
}