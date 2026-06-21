//! Async file I/O primitives for Moirai concurrency library.
//!
//! This module provides Moirai-owned async file facade operations without Tokio
//! dependencies.

pub mod options;
pub mod stats;
pub mod file;
pub mod ops;

pub use options::FileOpenOptions;
pub use stats::FileStats;
pub use file::File;
pub use ops::{
    read_to_string, read, write, write_str, append, append_str, copy, metadata,
    rename, remove_file, create_dir, create_dir_all, remove_dir, remove_dir_all,
};

#[cfg(test)]
mod tests;
