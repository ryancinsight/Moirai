//! TLS 1.3 HKDF implemented with the `hkdf` crate.
//!
//! Extraction and expansion use the `hkdf` crate directly, while the
//! `hmac_sign` operation (used for TLS 1.3 Finished verification) is delegated
//! to the crate's [`crate::hmac`] providers.

use hkdf::Hkdf as HkdfImpl;
use rustls::crypto::hmac::{Hmac, Tag};
use rustls::crypto::tls13::{Hkdf, HkdfExpander, OkmBlock, OutputLengthError};
use sha2::{Sha256, Sha384};

use crate::hmac::{HMAC_SHA256, HMAC_SHA384};

/// HKDF using SHA-256.
pub static HKDF_SHA256: HkdfSha256 = HkdfSha256;

/// HKDF using SHA-384.
pub static HKDF_SHA384: HkdfSha384 = HkdfSha384;

/// HKDF-SHA-256 [`Hkdf`] implementation.
#[derive(Debug)]
pub struct HkdfSha256;

/// HKDF-SHA-384 [`Hkdf`] implementation.
#[derive(Debug)]
pub struct HkdfSha384;

impl Hkdf for HkdfSha256 {
    fn extract_from_zero_ikm(&self, salt: Option<&[u8]>) -> Box<dyn HkdfExpander> {
        let (_, hk) = HkdfImpl::<Sha256>::extract(salt, &[0u8; 32]);
        Box::new(ExpanderSha256(hk))
    }

    fn extract_from_secret(&self, salt: Option<&[u8]>, secret: &[u8]) -> Box<dyn HkdfExpander> {
        let (_, hk) = HkdfImpl::<Sha256>::extract(salt, secret);
        Box::new(ExpanderSha256(hk))
    }

    fn expander_for_okm(&self, okm: &OkmBlock) -> Box<dyn HkdfExpander> {
        let hk = HkdfImpl::<Sha256>::from_prk(okm.as_ref()).expect("PRK has valid length");
        Box::new(ExpanderSha256(hk))
    }

    fn hmac_sign(&self, key: &OkmBlock, message: &[u8]) -> Tag {
        HMAC_SHA256.with_key(key.as_ref()).sign(&[message])
    }
}

impl Hkdf for HkdfSha384 {
    fn extract_from_zero_ikm(&self, salt: Option<&[u8]>) -> Box<dyn HkdfExpander> {
        let (_, hk) = HkdfImpl::<Sha384>::extract(salt, &[0u8; 48]);
        Box::new(ExpanderSha384(hk))
    }

    fn extract_from_secret(&self, salt: Option<&[u8]>, secret: &[u8]) -> Box<dyn HkdfExpander> {
        let (_, hk) = HkdfImpl::<Sha384>::extract(salt, secret);
        Box::new(ExpanderSha384(hk))
    }

    fn expander_for_okm(&self, okm: &OkmBlock) -> Box<dyn HkdfExpander> {
        let hk = HkdfImpl::<Sha384>::from_prk(okm.as_ref()).expect("PRK has valid length");
        Box::new(ExpanderSha384(hk))
    }

    fn hmac_sign(&self, key: &OkmBlock, message: &[u8]) -> Tag {
        HMAC_SHA384.with_key(key.as_ref()).sign(&[message])
    }
}

struct ExpanderSha256(HkdfImpl<Sha256>);

impl HkdfExpander for ExpanderSha256 {
    fn expand_slice(&self, info: &[&[u8]], output: &mut [u8]) -> Result<(), OutputLengthError> {
        self.0
            .expand_multi_info(info, output)
            .map_err(|_| OutputLengthError)
    }

    fn expand_block(&self, info: &[&[u8]]) -> OkmBlock {
        let mut buf = [0u8; 32];
        self.0
            .expand_multi_info(info, &mut buf)
            .expect("HashLen output is always valid");
        OkmBlock::new(&buf)
    }

    fn hash_len(&self) -> usize {
        32
    }
}

struct ExpanderSha384(HkdfImpl<Sha384>);

impl HkdfExpander for ExpanderSha384 {
    fn expand_slice(&self, info: &[&[u8]], output: &mut [u8]) -> Result<(), OutputLengthError> {
        self.0
            .expand_multi_info(info, output)
            .map_err(|_| OutputLengthError)
    }

    fn expand_block(&self, info: &[&[u8]]) -> OkmBlock {
        let mut buf = [0u8; 48];
        self.0
            .expand_multi_info(info, &mut buf)
            .expect("HashLen output is always valid");
        OkmBlock::new(&buf)
    }

    fn hash_len(&self) -> usize {
        48
    }
}
