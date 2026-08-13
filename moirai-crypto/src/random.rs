//! Cryptographically secure randomness backed by [`getrandom`].

use rustls::crypto::{GetRandomFailed, SecureRandom};

/// A [`SecureRandom`] implementation sourcing entropy from the operating
/// system via the pure-Rust `getrandom` crate (no C dependency).
#[derive(Debug)]
pub struct Getrandom;

impl SecureRandom for Getrandom {
    fn fill(&self, buf: &mut [u8]) -> Result<(), GetRandomFailed> {
        getrandom::getrandom(buf).map_err(|_| GetRandomFailed)
    }
}
