use std::fs::OpenOptions as StdOpenOptions;
use std::io::{self, Write};
use std::path::Path;

use super::yield_now;

/// Copy a file through the platform file-copy implementation.
pub async fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
    yield_now().await;
    std::fs::copy(from, to)
}

/// Write bytes through the platform file-write implementation.
pub async fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    yield_now().await;
    std::fs::write(path, contents)
}

/// Append bytes through the platform append implementation.
pub async fn append<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    yield_now().await;
    let mut file = StdOpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(contents.as_ref())
}

/// Read file metadata through the platform metadata implementation.
pub async fn metadata<P: AsRef<Path>>(path: P) -> io::Result<std::fs::Metadata> {
    yield_now().await;
    std::fs::metadata(path)
}

/// Rename a path through the platform rename implementation.
pub async fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()> {
    yield_now().await;
    std::fs::rename(from, to)
}

/// Remove a file through the platform remove implementation.
pub async fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()> {
    yield_now().await;
    std::fs::remove_file(path)
}

/// Create one directory through the platform directory-create implementation.
pub async fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    yield_now().await;
    std::fs::create_dir(path)
}

/// Create a directory tree through the platform recursive directory-create implementation.
pub async fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    yield_now().await;
    std::fs::create_dir_all(path)
}

/// Remove one empty directory through the platform directory-remove implementation.
pub async fn remove_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    yield_now().await;
    std::fs::remove_dir(path)
}

/// Remove a directory tree through the platform recursive directory-remove implementation.
pub async fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    yield_now().await;
    std::fs::remove_dir_all(path)
}
