//! # moirai-crypto
//!
//! A pure-Rust [`rustls::crypto::CryptoProvider`] backed entirely by
//! [RustCrypto](https://github.com/RustCrypto) primitives, with **zero C
//! dependencies**. It is a drop-in replacement for the `ring`-based provider
//! used by the Moirai TLS stack (`moirai-tls`).
//!
//! The provider supports the three TLS 1.3 cipher suites, X25519 and P-256 key
//! exchange, and the ECDSA/RSA signature verification algorithms required to
//! validate typical web PKI certificate chains — all implemented in safe,
//! portable Rust.
//!
//! ```no_run
//! let provider = moirai_crypto::provider();
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod aead;
mod hash;
mod hkdf;
mod hmac;
mod kx;
mod random;
mod sign;
mod verify;

use std::sync::Arc;

use rustls::crypto::{CryptoProvider, KeyProvider};
use rustls::pki_types::PrivateKeyDer;
use rustls::sign::SigningKey;
use rustls::{CipherSuite, CipherSuiteCommon, Error, SupportedCipherSuite, Tls13CipherSuite};

/// Construct a [`CryptoProvider`] backed by RustCrypto (no C dependencies).
///
/// The returned provider offers:
/// - Cipher suites: `TLS13_CHACHA20_POLY1305_SHA256`, `TLS13_AES_128_GCM_SHA256`,
///   `TLS13_AES_256_GCM_SHA384`.
/// - Key exchange groups: X25519 and NIST P-256 (secp256r1).
/// - Signature verification: ECDSA (P-256/P-384) and RSA (PKCS#1 v1.5 and PSS)
///   with SHA-256/SHA-384.
#[must_use]
pub fn provider() -> CryptoProvider {
    CryptoProvider {
        cipher_suites: vec![
            SupportedCipherSuite::Tls13(&TLS13_CHACHA20_POLY1305_SHA256_SUITE),
            SupportedCipherSuite::Tls13(&TLS13_AES_128_GCM_SHA256_SUITE),
            SupportedCipherSuite::Tls13(&TLS13_AES_256_GCM_SHA384_SUITE),
        ],
        kx_groups: vec![&kx::X25519, &kx::P256],
        signature_verification_algorithms: verify::ALGORITHMS,
        secure_random: &random::Getrandom,
        key_provider: &RustCryptoKeyProvider,
    }
}

static TLS13_CHACHA20_POLY1305_SHA256_SUITE: Tls13CipherSuite = Tls13CipherSuite {
    common: CipherSuiteCommon {
        suite: CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
        hash_provider: &hash::SHA256,
        confidentiality_limit: u64::MAX,
    },
    hkdf_provider: &hkdf::HKDF_SHA256,
    aead_alg: &aead::ChaCha20Poly1305,
    quic: None,
};

static TLS13_AES_128_GCM_SHA256_SUITE: Tls13CipherSuite = Tls13CipherSuite {
    common: CipherSuiteCommon {
        suite: CipherSuite::TLS13_AES_128_GCM_SHA256,
        hash_provider: &hash::SHA256,
        confidentiality_limit: 1 << 23,
    },
    hkdf_provider: &hkdf::HKDF_SHA256,
    aead_alg: &aead::Aes128Gcm,
    quic: None,
};

static TLS13_AES_256_GCM_SHA384_SUITE: Tls13CipherSuite = Tls13CipherSuite {
    common: CipherSuiteCommon {
        suite: CipherSuite::TLS13_AES_256_GCM_SHA384,
        hash_provider: &hash::SHA384,
        confidentiality_limit: 1 << 23,
    },
    hkdf_provider: &hkdf::HKDF_SHA384,
    aead_alg: &aead::Aes256Gcm,
    quic: None,
};

/// Private-key loader for the provider.
///
/// Loads ECDSA (P-256/P-384) and RSA private keys from PKCS#8, SEC1, or PKCS#1
/// DER, enabling server authentication and client-certificate authentication in
/// addition to certificate verification.
#[derive(Debug)]
struct RustCryptoKeyProvider;

impl KeyProvider for RustCryptoKeyProvider {
    fn load_private_key(
        &self,
        key_der: PrivateKeyDer<'static>,
    ) -> Result<Arc<dyn SigningKey>, Error> {
        sign::any_supported_type(&key_der)
    }
}
