//! Certificate signature verification algorithms implemented with RustCrypto
//! (`p256`, `p384`, `rsa`), exposed as [`SignatureVerificationAlgorithm`]s.

use rustls::crypto::WebPkiSupportedAlgorithms;
use rustls::pki_types::{alg_id, AlgorithmIdentifier, InvalidSignature, SignatureVerificationAlgorithm};
use rustls::SignatureScheme;

use rsa::pkcs1::DecodeRsaPublicKey;
use signature::Verifier;

/// The set of signature verification algorithms supported by this provider,
/// together with their mapping to TLS [`SignatureScheme`]s.
pub static ALGORITHMS: WebPkiSupportedAlgorithms = WebPkiSupportedAlgorithms {
    all: &[
        ECDSA_P256_SHA256,
        ECDSA_P384_SHA384,
        RSA_PKCS1_SHA256,
        RSA_PKCS1_SHA384,
        RSA_PSS_SHA256,
        RSA_PSS_SHA384,
    ],
    mapping: &[
        (
            SignatureScheme::ECDSA_NISTP256_SHA256,
            &[ECDSA_P256_SHA256],
        ),
        (
            SignatureScheme::ECDSA_NISTP384_SHA384,
            &[ECDSA_P384_SHA384],
        ),
        (SignatureScheme::RSA_PSS_SHA256, &[RSA_PSS_SHA256]),
        (SignatureScheme::RSA_PSS_SHA384, &[RSA_PSS_SHA384]),
        (SignatureScheme::RSA_PKCS1_SHA256, &[RSA_PKCS1_SHA256]),
        (SignatureScheme::RSA_PKCS1_SHA384, &[RSA_PKCS1_SHA384]),
    ],
};

static ECDSA_P256_SHA256: &dyn SignatureVerificationAlgorithm = &EcdsaP256Sha256;
static ECDSA_P384_SHA384: &dyn SignatureVerificationAlgorithm = &EcdsaP384Sha384;
static RSA_PKCS1_SHA256: &dyn SignatureVerificationAlgorithm = &RsaPkcs1Sha256;
static RSA_PKCS1_SHA384: &dyn SignatureVerificationAlgorithm = &RsaPkcs1Sha384;
static RSA_PSS_SHA256: &dyn SignatureVerificationAlgorithm = &RsaPssSha256;
static RSA_PSS_SHA384: &dyn SignatureVerificationAlgorithm = &RsaPssSha384;

#[derive(Debug)]
struct EcdsaP256Sha256;

impl SignatureVerificationAlgorithm for EcdsaP256Sha256 {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        let key =
            p256::ecdsa::VerifyingKey::from_sec1_bytes(public_key).map_err(|_| InvalidSignature)?;
        let sig = p256::ecdsa::Signature::from_der(signature).map_err(|_| InvalidSignature)?;
        key.verify(message, &sig).map_err(|_| InvalidSignature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::ECDSA_P256
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::ECDSA_SHA256
    }
}

#[derive(Debug)]
struct EcdsaP384Sha384;

impl SignatureVerificationAlgorithm for EcdsaP384Sha384 {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        let key =
            p384::ecdsa::VerifyingKey::from_sec1_bytes(public_key).map_err(|_| InvalidSignature)?;
        let sig = p384::ecdsa::Signature::from_der(signature).map_err(|_| InvalidSignature)?;
        key.verify(message, &sig).map_err(|_| InvalidSignature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::ECDSA_P384
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::ECDSA_SHA384
    }
}

#[derive(Debug)]
struct RsaPkcs1Sha256;

impl SignatureVerificationAlgorithm for RsaPkcs1Sha256 {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        let key = rsa::RsaPublicKey::from_pkcs1_der(public_key).map_err(|_| InvalidSignature)?;
        let vk = rsa::pkcs1v15::VerifyingKey::<sha2::Sha256>::new(key);
        let sig = rsa::pkcs1v15::Signature::try_from(signature).map_err(|_| InvalidSignature)?;
        vk.verify(message, &sig).map_err(|_| InvalidSignature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_ENCRYPTION
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_PKCS1_SHA256
    }
}

#[derive(Debug)]
struct RsaPkcs1Sha384;

impl SignatureVerificationAlgorithm for RsaPkcs1Sha384 {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        let key = rsa::RsaPublicKey::from_pkcs1_der(public_key).map_err(|_| InvalidSignature)?;
        let vk = rsa::pkcs1v15::VerifyingKey::<sha2::Sha384>::new(key);
        let sig = rsa::pkcs1v15::Signature::try_from(signature).map_err(|_| InvalidSignature)?;
        vk.verify(message, &sig).map_err(|_| InvalidSignature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_ENCRYPTION
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_PKCS1_SHA384
    }
}

#[derive(Debug)]
struct RsaPssSha256;

impl SignatureVerificationAlgorithm for RsaPssSha256 {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        let key = rsa::RsaPublicKey::from_pkcs1_der(public_key).map_err(|_| InvalidSignature)?;
        let vk = rsa::pss::VerifyingKey::<sha2::Sha256>::new(key);
        let sig = rsa::pss::Signature::try_from(signature).map_err(|_| InvalidSignature)?;
        vk.verify(message, &sig).map_err(|_| InvalidSignature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_ENCRYPTION
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_PSS_SHA256
    }
}

#[derive(Debug)]
struct RsaPssSha384;

impl SignatureVerificationAlgorithm for RsaPssSha384 {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        let key = rsa::RsaPublicKey::from_pkcs1_der(public_key).map_err(|_| InvalidSignature)?;
        let vk = rsa::pss::VerifyingKey::<sha2::Sha384>::new(key);
        let sig = rsa::pss::Signature::try_from(signature).map_err(|_| InvalidSignature)?;
        vk.verify(message, &sig).map_err(|_| InvalidSignature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_ENCRYPTION
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        alg_id::RSA_PSS_SHA384
    }
}
