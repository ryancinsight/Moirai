//! Async file I/O primitives for Moirai concurrency library.
//!
//! File-system syscalls have no readiness model: they block the calling thread
//! on every platform this crate serves. Every operation here therefore runs on
//! a bounded file-system blocking pool, never
//! inside `poll`, so a slow disk or network mount stalls a pool worker, not the
//! executor, and a dropped future can be cancelled.

use std::sync::OnceLock;

use crate::blocking::BlockingPool;

/// Async file handle and open/create entry points.
pub mod file;
/// Free-function file operations (read, write, copy, remove).
pub mod ops;
/// File-operation statistics counters.
pub mod stats;

pub use file::File;
pub use moirai_pal::fs::FileOpenOptions;
pub use ops::{
    append, append_str, copy, create_dir, create_dir_all, metadata, read, read_to_string,
    remove_dir, remove_dir_all, remove_file, rename, write, write_str,
};
pub use stats::FileStats;

/// File-system worker threads for the whole process.
///
/// A file syscall holds its worker for one device round trip. That is
/// microseconds on local storage, and up to the server's timeout on a network
/// mount. Four workers bound the threads that a stalled mount can pin, and
/// leave three serving while one hangs. Operations past the bound wait
/// asynchronously for admission, which costs no thread.
pub(crate) const FS_WORKERS: usize = 4;

/// Admitted operations that may wait in the queue behind the running ones:
/// one per worker, so each worker takes its next job without a round trip
/// through admission.
pub(crate) const FS_QUEUE_DEPTH: usize = FS_WORKERS;

/// The process-wide file-system pool.
pub(crate) fn pool() -> &'static BlockingPool {
    static POOL: OnceLock<BlockingPool> = OnceLock::new();
    POOL.get_or_init(|| BlockingPool::new("moirai-fs", FS_WORKERS, FS_QUEUE_DEPTH))
}

#[cfg(test)]
mod tests;
