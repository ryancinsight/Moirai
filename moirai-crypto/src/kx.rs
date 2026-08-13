//! Ephemeral key-exchange groups (X25519 and NIST P-256) implemented with
//! RustCrypto primitives.

use rustls::crypto::{ActiveKeyExchange, SharedSecret, SupportedKxGroup};
use rustls::{Error, NamedGroup};

use p256::elliptic_curve::sec1::ToEncodedPoint;
use rand_core::OsRng;

/// Ephemeral ECDH on Curve25519 (RFC 7748).
#[derive(Debug)]
pub struct X25519;

/// Ephemeral ECDH on NIST P-256 (secp256r1).
#[derive(Debug)]
pub struct P256;

impl SupportedKxGroup for X25519 {
    fn start(&self) -> Result<Box<dyn ActiveKeyExchange>, Error> {
        let secret = x25519_dalek::EphemeralSecret::random_from_rng(OsRng);
        let public = x25519_dalek::PublicKey::from(&secret);
        Ok(Box::new(X25519KeyExchange {
            secret,
            pub_key: public.as_bytes().to_vec(),
        }))
    }

    fn name(&self) -> NamedGroup {
        NamedGroup::X25519
    }
}

impl SupportedKxGroup for P256 {
    fn start(&self) -> Result<Box<dyn ActiveKeyExchange>, Error> {
        let secret = p256::ecdh::EphemeralSecret::random(&mut OsRng);
        let pub_key = secret
            .public_key()
            .to_encoded_point(false)
            .as_bytes()
            .to_vec();
        Ok(Box::new(P256KeyExchange { secret, pub_key }))
    }

    fn name(&self) -> NamedGroup {
        NamedGroup::secp256r1
    }
}

struct X25519KeyExchange {
    secret: x25519_dalek::EphemeralSecret,
    pub_key: Vec<u8>,
}

impl ActiveKeyExchange for X25519KeyExchange {
    fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
        let peer: [u8; 32] = peer_pub_key
            .try_into()
            .map_err(|_| Error::from(rustls::PeerMisbehaved::InvalidKeyShare))?;
        let peer = x25519_dalek::PublicKey::from(peer);
        let shared = self.secret.diffie_hellman(&peer);
        // Reject low-order points which yield an all-zero shared secret.
        if !shared.was_contributory() {
            return Err(rustls::PeerMisbehaved::InvalidKeyShare.into());
        }
        Ok(SharedSecret::from(shared.as_bytes().as_slice()))
    }

    fn pub_key(&self) -> &[u8] {
        &self.pub_key
    }

    fn group(&self) -> NamedGroup {
        NamedGroup::X25519
    }
}

struct P256KeyExchange {
    secret: p256::ecdh::EphemeralSecret,
    pub_key: Vec<u8>,
}

impl ActiveKeyExchange for P256KeyExchange {
    fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
        let peer = p256::PublicKey::from_sec1_bytes(peer_pub_key)
            .map_err(|_| Error::from(rustls::PeerMisbehaved::InvalidKeyShare))?;
        let shared = self.secret.diffie_hellman(&peer);
        Ok(SharedSecret::from(shared.raw_secret_bytes().as_slice()))
    }

    fn pub_key(&self) -> &[u8] {
        &self.pub_key
    }

    fn group(&self) -> NamedGroup {
        NamedGroup::secp256r1
    }
}
