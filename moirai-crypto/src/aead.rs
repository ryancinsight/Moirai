//! TLS 1.3 AEAD cipher suites (AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305)
//! implemented with the `aes-gcm` and `chacha20poly1305` crates.

use aes_gcm::aead::generic_array::typenum::Unsigned;
use aes_gcm::aead::generic_array::GenericArray;
use aes_gcm::aead::{AeadCore, AeadInPlace, KeyInit};

use rustls::crypto::cipher::{
    make_tls13_aad, AeadKey, InboundOpaqueMessage, InboundPlainMessage, Iv, MessageDecrypter,
    MessageEncrypter, Nonce, OutboundOpaqueMessage, OutboundPlainMessage, PrefixedPayload,
    Tls13AeadAlgorithm, UnsupportedOperationError,
};
use rustls::{ConnectionTrafficSecrets, ContentType, Error, ProtocolVersion};

/// Length of the TLS record header prefix reserved by [`PrefixedPayload`].
const PREFIX_LEN: usize = 5;

/// AES-128-GCM AEAD.
#[derive(Debug)]
pub struct Aes128Gcm;

/// AES-256-GCM AEAD.
#[derive(Debug)]
pub struct Aes256Gcm;

/// ChaCha20-Poly1305 AEAD.
#[derive(Debug)]
pub struct ChaCha20Poly1305;

impl Tls13AeadAlgorithm for Aes128Gcm {
    fn encrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageEncrypter> {
        Box::new(Encrypter {
            cipher: aes_gcm::Aes128Gcm::new_from_slice(key.as_ref()).expect("valid key length"),
            iv,
        })
    }

    fn decrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageDecrypter> {
        Box::new(Decrypter {
            cipher: aes_gcm::Aes128Gcm::new_from_slice(key.as_ref()).expect("valid key length"),
            iv,
        })
    }

    fn key_len(&self) -> usize {
        16
    }

    fn extract_keys(
        &self,
        key: AeadKey,
        iv: Iv,
    ) -> Result<ConnectionTrafficSecrets, UnsupportedOperationError> {
        Ok(ConnectionTrafficSecrets::Aes128Gcm { key, iv })
    }
}

impl Tls13AeadAlgorithm for Aes256Gcm {
    fn encrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageEncrypter> {
        Box::new(Encrypter {
            cipher: aes_gcm::Aes256Gcm::new_from_slice(key.as_ref()).expect("valid key length"),
            iv,
        })
    }

    fn decrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageDecrypter> {
        Box::new(Decrypter {
            cipher: aes_gcm::Aes256Gcm::new_from_slice(key.as_ref()).expect("valid key length"),
            iv,
        })
    }

    fn key_len(&self) -> usize {
        32
    }

    fn extract_keys(
        &self,
        key: AeadKey,
        iv: Iv,
    ) -> Result<ConnectionTrafficSecrets, UnsupportedOperationError> {
        Ok(ConnectionTrafficSecrets::Aes256Gcm { key, iv })
    }
}

impl Tls13AeadAlgorithm for ChaCha20Poly1305 {
    fn encrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageEncrypter> {
        Box::new(Encrypter {
            cipher: chacha20poly1305::ChaCha20Poly1305::new_from_slice(key.as_ref())
                .expect("valid key length"),
            iv,
        })
    }

    fn decrypter(&self, key: AeadKey, iv: Iv) -> Box<dyn MessageDecrypter> {
        Box::new(Decrypter {
            cipher: chacha20poly1305::ChaCha20Poly1305::new_from_slice(key.as_ref())
                .expect("valid key length"),
            iv,
        })
    }

    fn key_len(&self) -> usize {
        32
    }

    fn extract_keys(
        &self,
        key: AeadKey,
        iv: Iv,
    ) -> Result<ConnectionTrafficSecrets, UnsupportedOperationError> {
        Ok(ConnectionTrafficSecrets::Chacha20Poly1305 { key, iv })
    }
}

struct Encrypter<C> {
    cipher: C,
    iv: Iv,
}

impl<C> MessageEncrypter for Encrypter<C>
where
    C: AeadInPlace + Send + Sync + 'static,
{
    fn encrypt(
        &mut self,
        msg: OutboundPlainMessage<'_>,
        seq: u64,
    ) -> Result<OutboundOpaqueMessage, Error> {
        let total_len = self.encrypted_payload_len(msg.payload.len());
        let mut payload = PrefixedPayload::with_capacity(total_len);
        payload.extend_from_chunks(&msg.payload);
        payload.extend_from_slice(&msg.typ.to_array());

        let nonce_bytes = Nonce::new(&self.iv, seq).0;
        let nonce = GenericArray::from_slice(&nonce_bytes);
        let aad = make_tls13_aad(total_len);

        let tag = self
            .cipher
            .encrypt_in_place_detached(nonce, &aad, &mut payload.as_mut()[PREFIX_LEN..])
            .map_err(|_| Error::EncryptError)?;
        payload.extend_from_slice(&tag);

        Ok(OutboundOpaqueMessage::new(
            ContentType::ApplicationData,
            ProtocolVersion::TLSv1_2,
            payload,
        ))
    }

    fn encrypted_payload_len(&self, payload_len: usize) -> usize {
        payload_len + 1 + <C as AeadCore>::TagSize::to_usize()
    }
}

struct Decrypter<C> {
    cipher: C,
    iv: Iv,
}

impl<C> MessageDecrypter for Decrypter<C>
where
    C: AeadInPlace + Send + Sync + 'static,
{
    fn decrypt<'a>(
        &mut self,
        mut msg: InboundOpaqueMessage<'a>,
        seq: u64,
    ) -> Result<InboundPlainMessage<'a>, Error> {
        let tag_len = <C as AeadCore>::TagSize::to_usize();
        let payload_len = msg.payload.len();
        if payload_len < tag_len {
            return Err(Error::DecryptError);
        }

        let nonce_bytes = Nonce::new(&self.iv, seq).0;
        let nonce = GenericArray::from_slice(&nonce_bytes);
        let aad = make_tls13_aad(payload_len);
        let msg_len = payload_len - tag_len;

        {
            let buffer: &mut [u8] = &mut msg.payload;
            let (ciphertext, tag_bytes) = buffer.split_at_mut(msg_len);
            let tag = GenericArray::from_slice(tag_bytes);
            self.cipher
                .decrypt_in_place_detached(nonce, &aad, ciphertext, tag)
                .map_err(|_| Error::DecryptError)?;
        }

        msg.payload.truncate(msg_len);
        msg.into_tls13_unpadded_message()
    }
}
