//! Async file I/O primitives for Moirai concurrency library.
//!
//! This module provides Moirai-owned async file facade operations without Tokio
//! dependencies.

pub mod file;
pub mod ops;
pub mod options;
pub mod stats;

pub use file::File;
pub use ops::{
    append, append_str, copy, create_dir, create_dir_all, metadata, read, read_to_string,
    remove_dir, remove_dir_all, remove_file, rename, write, write_str,
};
pub use options::FileOpenOptions;
pub use stats::FileStats;

#[cfg(test)]
mod tests;
