//! Standalone SHA-256, HMAC-SHA256 and fixed-width comparison primitives.
//!
//! These APIs share the RustCrypto implementations used by the optional TLS
//! provider, but do not require the provider feature or a TLS dependency in a
//! consumer that only needs protocol authentication.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256 as Sha256Digest};

/// Streaming SHA-256 state.
#[derive(Clone, Debug)]
pub struct Sha256(Sha256Digest);

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Creates an empty SHA-256 state.
    #[must_use]
    pub fn new() -> Self {
        Self(Sha256Digest::new())
    }

    /// Appends bytes to the hash state.
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    /// Finalizes the state and returns the 32-byte digest.
    #[must_use]
    pub fn finalize(self) -> [u8; 32] {
        let digest = self.0.finalize();
        let mut output = [0u8; 32];
        output.copy_from_slice(&digest);
        output
    }
}

/// Computes a SHA-256 digest in one pass.
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

/// Computes HMAC-SHA256 according to RFC 2104.
#[must_use]
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut mac = Hmac::<Sha256Digest>::new_from_slice(key)
        .expect("invariant: HMAC accepts keys of every length");
    mac.update(message);
    let tag = mac.finalize().into_bytes();
    let mut output = [0u8; 32];
    output.copy_from_slice(&tag);
    output
}

/// Compares two 32-byte values without a source-level early exit.
///
/// The XOR accumulation has constant work for every input pair. This API does
/// not claim a machine-code timing proof; release code generation is checked
/// separately by the provider's verification workflow.
#[must_use]
pub fn constant_time_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut difference = 0u8;
    for (&left, &right) in a.iter().zip(b) {
        difference |= std::hint::black_box(left ^ right);
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_fips_vectors() {
        assert_eq!(
            sha256(b""),
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
                0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
                0x78, 0x52, 0xb8, 0x55,
            ]
        );
        assert_eq!(
            sha256(b"abc"),
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad,
            ]
        );
    }

    #[test]
    fn hmac_sha256_matches_rfc_4231_vector() {
        assert_eq!(
            hmac_sha256(&[0x0b; 20], b"Hi There"),
            [
                0xb0, 0x34, 0x4c, 0x61, 0xd8, 0xdb, 0x38, 0x53, 0x5c, 0xa8, 0xaf, 0xce, 0xaf, 0x0b,
                0xf1, 0x2b, 0x88, 0x1d, 0xc2, 0x00, 0xc9, 0x83, 0x3d, 0xa7, 0x26, 0xe9, 0x37, 0x6c,
                0x2e, 0x32, 0xcf, 0xf7,
            ]
        );
    }

    #[test]
    fn streaming_state_matches_one_shot_across_padding_boundaries() {
        for length in [0, 1, 55, 56, 63, 64, 65, 127, 128, 129] {
            let input = vec![0x5a; length];
            for split in 0..=length {
                let mut state = Sha256::new();
                state.update(&input[..split]);
                state.update(&input[split..]);
                assert_eq!(state.finalize(), sha256(&input));
            }
        }
    }

    #[test]
    fn fixed_width_comparison_checks_all_value_cases() {
        let equal = [0x42; 32];
        let mut different = equal;
        different[31] ^= 1;
        assert!(constant_time_eq_32(&equal, &equal));
        assert!(!constant_time_eq_32(&equal, &different));
    }
}
