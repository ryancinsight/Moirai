//! Async I/O traits and compatibility utilities for Moirai.
//!
//! This module defines the core asynchronous I/O abstractions matching
//! zero-copy buffer ownership, zero-cost extension futures, and
//! monomorphization goals.

/// Adapters bridging Moirai and Tokio I/O trait families.
pub mod compat;
/// Extension futures over the core read/write traits.
pub mod ext;
/// Positioned asynchronous reads and source lengths.
pub mod positioned;
/// Core async read/write/seek trait definitions.
pub mod traits;

pub use compat::{MoiraiCompat, TokioCompat};

pub use ext::{AsyncReadExt, AsyncWriteExt, Flush, Read, ReadExact, Shutdown, Write, WriteAll};
pub use positioned::{AsyncLength, AsyncMemReader, AsyncReadAt};
pub use traits::{AsyncBufRead, AsyncRead, AsyncWrite};

#[cfg(test)]
mod tests;
