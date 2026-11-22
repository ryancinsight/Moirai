#![cfg(feature = "provider")]

use rsa::pkcs1::{EncodeRsaPrivateKey, EncodeRsaPublicKey};
use rsa::pkcs8::DecodePrivateKey;
use rsa::sha2::Sha512;
use rsa::signature::Verifier;
use rustls::SignatureScheme;
use rustls::pki_types::{PrivateKeyDer, pem::PemObject};

/// Each offered scheme must authenticate the original message and reject a
/// changed message. TLS verification advertises four schemes; for SHA-512 the
/// signing-only contract is checked directly against the named RSA verifier.
#[test]
fn provider_signatures_bind_message_and_scheme_for_both_key_formats() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .finish();
    let _trace = tracing::subscriber::set_default(subscriber);
    let (provider, fixture, legacy, public, public_der) =
        tracing::info_span!("arrange").in_scope(|| {
            let fixture = PrivateKeyDer::from_pem_slice(include_bytes!(
                "../../tests/fixtures/localhost-key.pem"
            ))
            .unwrap();
            let private = rsa::PrivateKey::from_pkcs8_der(fixture.secret_der()).unwrap();
            let public = private.to_public_key();
            let public_der = public.to_pkcs1_der().unwrap();
            let legacy =
                PrivateKeyDer::Pkcs1(private.to_pkcs1_der().unwrap().as_bytes().to_vec().into());
            (
                moirai_crypto::provider(),
                fixture,
                legacy,
                public,
                public_der,
            )
        });
    for key in [fixture, legacy] {
        let signing_key = provider.key_provider.load_private_key(key).unwrap();
        for scheme in [
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
        ] {
            let signature = tracing::info_span!("act", ?scheme).in_scope(|| {
                let signer = signing_key.choose_scheme(&[scheme]).unwrap();
                assert_eq!(signer.scheme(), scheme);
                signer.sign(b"original").unwrap()
            });
            tracing::info_span!("assert", ?scheme).in_scope(|| match scheme {
                SignatureScheme::RSA_PKCS1_SHA512 => {
                    let verifier = rsa::pkcs1v15::VerifyingKey::<Sha512>::new(public.clone());
                    let sig = rsa::pkcs1v15::Signature::try_from(signature.as_slice()).unwrap();
                    verifier.verify(b"original", &sig).unwrap();
                    assert!(verifier.verify(b"modified", &sig).is_err());
                }
                SignatureScheme::RSA_PSS_SHA512 => {
                    let verifier = rsa::pss::VerifyingKey::<Sha512>::new(public.clone());
                    let sig = rsa::pss::Signature::try_from(signature.as_slice()).unwrap();
                    verifier.verify(b"original", &sig).unwrap();
                    assert!(verifier.verify(b"modified", &sig).is_err());
                }
                _ => {
                    let (_, algorithms) = provider
                        .signature_verification_algorithms
                        .mapping
                        .iter()
                        .find(|(advertised, _)| *advertised == scheme)
                        .unwrap();
                    assert!(!algorithms.is_empty());
                    for verifier in *algorithms {
                        let key = public_der.as_bytes();
                        verifier
                            .verify_signature(key, b"original", &signature)
                            .unwrap();
                        assert!(
                            verifier
                                .verify_signature(key, b"modified", &signature)
                                .is_err()
                        );
                        assert!(
                            verifier
                                .verify_signature(key, b"original", &signature[1..])
                                .is_err()
                        );
                    }
                }
            });
        }
    }
}
