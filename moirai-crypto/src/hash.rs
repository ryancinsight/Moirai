//! SHA-256 and SHA-384 hash providers implemented with `sha2`.

use rustls::crypto::hash::{Context, Hash, HashAlgorithm, Output};
use sha2::{Digest, Sha256, Sha384};

/// SHA-256 hash provider.
pub static SHA256: Sha256Hash = Sha256Hash;

/// SHA-384 hash provider.
pub static SHA384: Sha384Hash = Sha384Hash;

/// SHA-256 [`Hash`] implementation.
#[derive(Debug)]
pub struct Sha256Hash;

/// SHA-384 [`Hash`] implementation.
#[derive(Debug)]
pub struct Sha384Hash;

impl Hash for Sha256Hash {
    fn start(&self) -> Box<dyn Context> {
        Box::new(HashContext(Sha256::new()))
    }

    fn hash(&self, data: &[u8]) -> Output {
        Output::new(&Sha256::digest(data)[..])
    }

    fn output_len(&self) -> usize {
        32
    }

    fn algorithm(&self) -> HashAlgorithm {
        HashAlgorithm::SHA256
    }
}

impl Hash for Sha384Hash {
    fn start(&self) -> Box<dyn Context> {
        Box::new(HashContext(Sha384::new()))
    }

    fn hash(&self, data: &[u8]) -> Output {
        Output::new(&Sha384::digest(data)[..])
    }

    fn output_len(&self) -> usize {
        48
    }

    fn algorithm(&self) -> HashAlgorithm {
        HashAlgorithm::SHA384
    }
}

struct HashContext<D>(D);

impl<D> Context for HashContext<D>
where
    D: Digest + Clone + Send + Sync + 'static,
{
    fn fork_finish(&self) -> Output {
        Output::new(&self.0.clone().finalize()[..])
    }

    fn fork(&self) -> Box<dyn Context> {
        Box::new(HashContext(self.0.clone()))
    }

    fn finish(self: Box<Self>) -> Output {
        Output::new(&self.0.finalize()[..])
    }

    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
}
