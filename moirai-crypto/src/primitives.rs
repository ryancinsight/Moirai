//! Standalone protocol hashing, HMAC and fixed-width comparison primitives.
//!
//! SHA-256 and HMAC-SHA256 support authentication; SHA-1 and Base64 are exposed
//! only for wire-protocol compatibility. These APIs share the RustCrypto
//! implementations used by the optional TLS provider, but do not require the
//! provider feature or a TLS dependency in a consumer that only needs protocol
//! primitives.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256 as Sha256Digest};

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Streaming SHA-1 state for protocol compatibility.
///
/// SHA-1 is provided only for wire protocols that require it, such as the
/// RFC 6455 WebSocket handshake. It is not suitable for authentication or
/// collision-resistant application data.
#[derive(Clone, Debug)]
pub struct Sha1 {
    state: [u32; 5],
    buffer: [u8; 64],
    buffered: usize,
    length: u64,
}

impl Default for Sha1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha1 {
    /// Creates an empty SHA-1 state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: [
                0x6745_2301,
                0xefcd_ab89,
                0x98ba_dcfe,
                0x1032_5476,
                0xc3d2_e1f0,
            ],
            buffer: [0; 64],
            buffered: 0,
            length: 0,
        }
    }

    /// Appends bytes to the SHA-1 state.
    ///
    /// The message-length field follows SHA-1's specified modulo-2^64
    /// encoding, so repeated updates do not panic when the counter wraps.
    pub fn update(&mut self, data: &[u8]) {
        let length = u64::try_from(data.len()).map_or(u64::MAX, |length| length);
        self.length = self.length.wrapping_add(length);
        let mut input = data;
        if self.buffered != 0 {
            let needed = 64 - self.buffered;
            let copied = input.len().min(needed);
            self.buffer[self.buffered..self.buffered + copied].copy_from_slice(&input[..copied]);
            self.buffered += copied;
            input = &input[copied..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.process_block(&block);
                self.buffered = 0;
            }
        }
        let mut blocks = input.chunks_exact(64);
        for block in &mut blocks {
            self.process_block(block);
        }
        let remainder = blocks.remainder();
        self.buffer[..remainder.len()].copy_from_slice(remainder);
        self.buffered = remainder.len();
    }

    /// Finalizes the state and returns the 20-byte digest.
    #[must_use]
    pub fn finalize(mut self) -> [u8; 20] {
        let bit_length = self.length.wrapping_mul(8);
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            let block = self.buffer;
            self.process_block(&block);
            self.buffered = 0;
        }
        self.buffer[self.buffered..56].fill(0);
        self.buffer[56..].copy_from_slice(&bit_length.to_be_bytes());
        let block = self.buffer;
        self.process_block(&block);

        let mut output = [0u8; 20];
        for (chunk, word) in output.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        output
    }

    fn process_block(&mut self, block: &[u8]) {
        let mut words = [0u32; 80];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            let bytes: [u8; 4] = chunk
                .try_into()
                .expect("invariant: chunks_exact yields four-byte words");
            words[index] = u32::from_be_bytes(bytes);
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }

        let [mut a, mut b, mut c, mut d, mut e] = self.state;
        for (index, word) in words.into_iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
    }
}

/// Computes the protocol-only SHA-1 digest of `data`.
#[must_use]
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut state = Sha1::new();
    state.update(data);
    state.finalize()
}

/// Encodes bytes as standard RFC 4648 Base64 with padding.
#[must_use]
pub fn base64_encode(data: &[u8]) -> String {
    let mut output = String::new();
    for chunk in data.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);
        output.push(char::from(BASE64_ALPHABET[usize::from(first >> 2)]));
        output.push(char::from(
            BASE64_ALPHABET[usize::from(((first & 0x03) << 4) | (second >> 4))],
        ));
        output.push(if chunk.len() > 1 {
            char::from(BASE64_ALPHABET[usize::from(((second & 0x0f) << 2) | (third >> 6))])
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            char::from(BASE64_ALPHABET[usize::from(third & 0x3f)])
        } else {
            '='
        });
    }
    output
}

/// Decodes padded standard RFC 4648 Base64.
///
/// Returns `None` for non-ASCII, malformed, or unpadded input.
pub fn base64_decode(input: &[u8]) -> Option<Vec<u8>> {
    if input.is_empty() {
        return Some(Vec::new());
    }
    if !input.len().is_multiple_of(4) {
        return None;
    }
    let groups = input.len() / 4;
    let padding = input.last().is_some_and(|last| *last == b'=') as usize
        + input
            .get(input.len().checked_sub(2)?)
            .is_some_and(|second_last| *second_last == b'=') as usize;
    let output_len = groups.checked_mul(3)?.checked_sub(padding)?;
    let mut output = Vec::with_capacity(output_len);
    for (group_index, chunk) in input.chunks_exact(4).enumerate() {
        let last_group = group_index + 1 == groups;
        let a = base64_value(chunk[0])?;
        let b = base64_value(chunk[1])?;
        let c = if chunk[2] == b'=' {
            if !last_group || chunk[3] != b'=' {
                return None;
            }
            0
        } else {
            base64_value(chunk[2])?
        };
        let d = if chunk[3] == b'=' {
            if !last_group {
                return None;
            }
            0
        } else {
            base64_value(chunk[3])?
        };
        if chunk[2] == b'=' && (b & 0x0f) != 0 {
            return None;
        }
        if chunk[3] == b'=' && chunk[2] != b'=' && (c & 0x03) != 0 {
            return None;
        }
        output.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            output.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            output.push((c << 6) | d);
        }
    }
    Some(output)
}

fn base64_value(value: u8) -> Option<u8> {
    match value {
        b'A'..=b'Z' => Some(value - b'A'),
        b'a'..=b'z' => Some(value - b'a' + 26),
        b'0'..=b'9' => Some(value - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

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
    fn sha1_matches_fips_vector() {
        assert_eq!(
            sha1(b""),
            [
                0xda, 0x39, 0xa3, 0xee, 0x5e, 0x6b, 0x4b, 0x0d, 0x32, 0x55, 0xbf, 0xef, 0x95, 0x60,
                0x18, 0x90, 0xaf, 0xd8, 0x07, 0x09,
            ]
        );
        assert_eq!(
            sha1(b"abc"),
            [
                0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e, 0x25, 0x71, 0x78, 0x50,
                0xc2, 0x6c, 0x9c, 0xd0, 0xd8, 0x9d,
            ]
        );

        let million = vec![b'a'; 1_000_000];
        assert_eq!(
            sha1(&million),
            [
                0x34, 0xaa, 0x97, 0x3c, 0xd4, 0xc4, 0xda, 0xa4, 0xf6, 0x1e, 0xeb, 0x2b, 0xdb, 0xad,
                0x27, 0x31, 0x65, 0x34, 0x01, 0x6f,
            ]
        );
    }

    #[test]
    fn base64_round_trip_preserves_padding_cases() {
        for input in [b"f".as_slice(), b"fo", b"foo", b"foobar"] {
            let encoded = base64_encode(input);
            assert_eq!(base64_decode(encoded.as_bytes()).as_deref(), Some(input));
        }
        assert_eq!(base64_decode(b""), Some(Vec::new()));
        assert_eq!(base64_decode(b"Zm8"), None);
        assert_eq!(base64_decode(b"Zm=8"), None);
        assert_eq!(base64_decode(b"Zh=="), None);
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
