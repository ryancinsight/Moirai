//! Software private-key signing (ECDSA P-256/P-384 and RSA) used by the
//! [`crate::provider`]'s [`rustls::crypto::KeyProvider`].
//!
//! This enables the provider to be used for TLS server authentication and TLS
//! client-certificate authentication, in addition to certificate verification.

use std::sync::Arc;

use rustls::pki_types::PrivateKeyDer;
use rustls::sign::{Signer, SigningKey};
use rustls::{Error, SignatureAlgorithm, SignatureScheme};

use p256::pkcs8::DecodePrivateKey as _;
use rsa::pkcs1::DecodeRsaPrivateKey as _;
use signature::{RandomizedSigner, SignatureEncoding, Signer as _};

/// Parse any supported private key (ECDSA P-256/P-384 or RSA) from DER.
pub fn any_supported_type(der: &PrivateKeyDer<'static>) -> Result<Arc<dyn SigningKey>, Error> {
    if let Ok(key) = EcdsaP256SigningKey::from_der(der) {
        return Ok(Arc::new(key));
    }
    if let Ok(key) = EcdsaP384SigningKey::from_der(der) {
        return Ok(Arc::new(key));
    }
    if let Ok(key) = RsaSigningKey::from_der(der) {
        return Ok(Arc::new(key));
    }
    Err(Error::General(
        "moirai-crypto: unsupported or malformed private key".into(),
    ))
}

// -------------------------------------------------------------------------
// ECDSA
// -------------------------------------------------------------------------

#[derive(Debug)]
struct EcdsaP256SigningKey(Arc<p256::ecdsa::SigningKey>);

impl EcdsaP256SigningKey {
    fn from_der(der: &PrivateKeyDer<'static>) -> Result<Self, ()> {
        let key = match der {
            PrivateKeyDer::Pkcs8(k) => {
                p256::ecdsa::SigningKey::from_pkcs8_der(k.secret_pkcs8_der()).map_err(|_| ())?
            }
            PrivateKeyDer::Sec1(k) => {
                p256::ecdsa::SigningKey::from_slice(k.secret_sec1_der()).map_err(|_| ())?
            }
            _ => return Err(()),
        };
        Ok(Self(Arc::new(key)))
    }
}

impl SigningKey for EcdsaP256SigningKey {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
        offered
            .contains(&SignatureScheme::ECDSA_NISTP256_SHA256)
            .then(|| Box::new(EcdsaP256Signer(self.0.clone())) as Box<dyn Signer>)
    }

    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::ECDSA
    }
}

#[derive(Debug)]
struct EcdsaP256Signer(Arc<p256::ecdsa::SigningKey>);

impl Signer for EcdsaP256Signer {
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
        let sig: p256::ecdsa::DerSignature = self
            .0
            .try_sign(message)
            .map_err(|e| Error::General(format!("ECDSA P-256 signing failed: {e}")))?;
        Ok(sig.to_vec())
    }

    fn scheme(&self) -> SignatureScheme {
        SignatureScheme::ECDSA_NISTP256_SHA256
    }
}

#[derive(Debug)]
struct EcdsaP384SigningKey(Arc<p384::ecdsa::SigningKey>);

impl EcdsaP384SigningKey {
    fn from_der(der: &PrivateKeyDer<'static>) -> Result<Self, ()> {
        let key = match der {
            PrivateKeyDer::Pkcs8(k) => {
                p384::ecdsa::SigningKey::from_pkcs8_der(k.secret_pkcs8_der()).map_err(|_| ())?
            }
            PrivateKeyDer::Sec1(k) => {
                p384::ecdsa::SigningKey::from_slice(k.secret_sec1_der()).map_err(|_| ())?
            }
            _ => return Err(()),
        };
        Ok(Self(Arc::new(key)))
    }
}

impl SigningKey for EcdsaP384SigningKey {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
        offered
            .contains(&SignatureScheme::ECDSA_NISTP384_SHA384)
            .then(|| Box::new(EcdsaP384Signer(self.0.clone())) as Box<dyn Signer>)
    }

    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::ECDSA
    }
}

#[derive(Debug)]
struct EcdsaP384Signer(Arc<p384::ecdsa::SigningKey>);

impl Signer for EcdsaP384Signer {
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
        let sig: p384::ecdsa::DerSignature = self
            .0
            .try_sign(message)
            .map_err(|e| Error::General(format!("ECDSA P-384 signing failed: {e}")))?;
        Ok(sig.to_vec())
    }

    fn scheme(&self) -> SignatureScheme {
        SignatureScheme::ECDSA_NISTP384_SHA384
    }
}

// -------------------------------------------------------------------------
// RSA
// -------------------------------------------------------------------------

/// RSA schemes we can produce, in descending order of preference.
const RSA_SCHEMES: &[SignatureScheme] = &[
    SignatureScheme::RSA_PSS_SHA512,
    SignatureScheme::RSA_PSS_SHA384,
    SignatureScheme::RSA_PSS_SHA256,
    SignatureScheme::RSA_PKCS1_SHA512,
    SignatureScheme::RSA_PKCS1_SHA384,
    SignatureScheme::RSA_PKCS1_SHA256,
];

#[derive(Debug)]
struct RsaSigningKey(Arc<rsa::RsaPrivateKey>);

impl RsaSigningKey {
    fn from_der(der: &PrivateKeyDer<'static>) -> Result<Self, ()> {
        let key = match der {
            PrivateKeyDer::Pkcs8(k) => {
                rsa::RsaPrivateKey::from_pkcs8_der(k.secret_pkcs8_der()).map_err(|_| ())?
            }
            PrivateKeyDer::Pkcs1(k) => {
                rsa::RsaPrivateKey::from_pkcs1_der(k.secret_pkcs1_der()).map_err(|_| ())?
            }
            _ => return Err(()),
        };
        Ok(Self(Arc::new(key)))
    }
}

impl SigningKey for RsaSigningKey {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
        RSA_SCHEMES
            .iter()
            .find(|scheme| offered.contains(scheme))
            .map(|&scheme| {
                Box::new(RsaSigner {
                    key: self.0.clone(),
                    scheme,
                }) as Box<dyn Signer>
            })
    }

    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::RSA
    }
}

#[derive(Debug)]
struct RsaSigner {
    key: Arc<rsa::RsaPrivateKey>,
    scheme: SignatureScheme,
}

impl Signer for RsaSigner {
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
        let key = (*self.key).clone();
        let sig = match self.scheme {
            SignatureScheme::RSA_PKCS1_SHA256 => {
                rsa::pkcs1v15::SigningKey::<sha2::Sha256>::new(key)
                    .try_sign(message)
                    .map(|s| s.to_vec())
            }
            SignatureScheme::RSA_PKCS1_SHA384 => {
                rsa::pkcs1v15::SigningKey::<sha2::Sha384>::new(key)
                    .try_sign(message)
                    .map(|s| s.to_vec())
            }
            SignatureScheme::RSA_PKCS1_SHA512 => {
                rsa::pkcs1v15::SigningKey::<sha2::Sha512>::new(key)
                    .try_sign(message)
                    .map(|s| s.to_vec())
            }
            SignatureScheme::RSA_PSS_SHA256 => Ok(rsa::pss::SigningKey::<sha2::Sha256>::new(key)
                .sign_with_rng(&mut rand_core::OsRng, message)
                .to_vec()),
            SignatureScheme::RSA_PSS_SHA384 => Ok(rsa::pss::SigningKey::<sha2::Sha384>::new(key)
                .sign_with_rng(&mut rand_core::OsRng, message)
                .to_vec()),
            SignatureScheme::RSA_PSS_SHA512 => Ok(rsa::pss::SigningKey::<sha2::Sha512>::new(key)
                .sign_with_rng(&mut rand_core::OsRng, message)
                .to_vec()),
            other => {
                return Err(Error::General(format!(
                    "moirai-crypto: unsupported RSA scheme {other:?}"
                )))
            }
        };
        sig.map_err(|e| Error::General(format!("RSA signing failed: {e}")))
    }

    fn scheme(&self) -> SignatureScheme {
        self.scheme
    }
}
