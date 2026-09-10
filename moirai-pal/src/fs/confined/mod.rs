//! Directory-handle anchored native file opening.
//!
//! Path canonicalization followed by a second path open leaves a
//! time-of-check/time-of-use race. This module validates the lexical path and
//! resolves every component from an owned directory handle instead.

mod open;

#[cfg(unix)]
mod unix;

#[cfg(windows)]
mod windows;

#[cfg(test)]
mod tests;

pub use open::open_file_within_root;
