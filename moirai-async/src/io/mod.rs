//! Async I/O traits and compatibility utilities for Moirai.
//!
//! This module defines the core asynchronous I/O abstractions matching
//! zero-copy buffer ownership, zero-cost extension futures, and
//! monomorphization goals.

pub mod compat;
pub mod ext;
pub mod traits;

pub use compat::{MoiraiCompat, TokioCompat};
pub use ext::{AsyncReadExt, AsyncWriteExt, Flush, Read, ReadExact, Shutdown, Write, WriteAll};
pub use traits::{AsyncBufRead, AsyncRead, AsyncWrite};

#[cfg(test)]
mod tests;
