//! HMAC-SHA-256 and HMAC-SHA-384 implemented with the `hmac` crate.

use hmac::{Hmac as HmacImpl, Mac};
use rustls::crypto::hmac::{Hmac, Key, Tag};
use sha2::{Sha256, Sha384};

type HmacSha256Impl = HmacImpl<Sha256>;
type HmacSha384Impl = HmacImpl<Sha384>;


/// HMAC-SHA-256 provider.
pub static HMAC_SHA256: HmacSha256 = HmacSha256;

/// HMAC-SHA-384 provider.
pub static HMAC_SHA384: HmacSha384 = HmacSha384;

/// HMAC-SHA-256 [`Hmac`] implementation.
#[derive(Debug)]
pub struct HmacSha256;

/// HMAC-SHA-384 [`Hmac`] implementation.
#[derive(Debug)]
pub struct HmacSha384;

impl Hmac for HmacSha256 {
    fn with_key(&self, key: &[u8]) -> Box<dyn Key> {
        Box::new(HmacKey256(
            HmacSha256Impl::new_from_slice(key).expect("HMAC accepts keys of any length"),
        ))
    }

    fn hash_output_len(&self) -> usize {
        32
    }
}

impl Hmac for HmacSha384 {
    fn with_key(&self, key: &[u8]) -> Box<dyn Key> {
        Box::new(HmacKey384(
            HmacSha384Impl::new_from_slice(key).expect("HMAC accepts keys of any length"),
        ))
    }

    fn hash_output_len(&self) -> usize {
        48
    }
}

struct HmacKey256(HmacSha256Impl);

impl Key for HmacKey256 {
    fn sign_concat(&self, first: &[u8], middle: &[&[u8]], last: &[u8]) -> Tag {
        let mut ctx = self.0.clone();
        ctx.update(first);
        for m in middle {
            ctx.update(m);
        }
        ctx.update(last);
        Tag::new(&ctx.finalize().into_bytes()[..])
    }

    fn tag_len(&self) -> usize {
        32
    }
}

struct HmacKey384(HmacSha384Impl);

impl Key for HmacKey384 {
    fn sign_concat(&self, first: &[u8], middle: &[&[u8]], last: &[u8]) -> Tag {
        let mut ctx = self.0.clone();
        ctx.update(first);
        for m in middle {
            ctx.update(m);
        }
        ctx.update(last);
        Tag::new(&ctx.finalize().into_bytes()[..])
    }

    fn tag_len(&self) -> usize {
        48
    }
}
