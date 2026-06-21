//! Async I/O traits and compatibility utilities for Moirai.
//!
//! This module defines the core asynchronous I/O abstractions matching
//! zero-copy buffer ownership, zero-cost extension futures, and
//! monomorphization goals.

pub mod traits;
pub mod ext;
pub mod compat;

pub use traits::{AsyncRead, AsyncWrite, AsyncBufRead};
pub use ext::{AsyncReadExt, Read, ReadExact, AsyncWriteExt, Write, WriteAll, Flush, Shutdown};
pub use compat::{TokioCompat, MoiraiCompat};

#[cfg(test)]
mod tests;
