use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::blocking::Abandoned;
use crate::fs::file::File;
use crate::fs::pool;

/// Run one path operation on the file-system pool. A caller that drops the
/// future while the operation is queued cancels it.
async fn on_pool<T: Send + 'static>(
    operation: impl FnOnce() -> io::Result<T> + Send + 'static,
) -> io::Result<T> {
    pool().run(Abandoned::Skip, operation).await?
}

/// Read entire file contents as a string
pub async fn read_to_string<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let mut file = File::open(path).await?;
    file.read_to_string().await
}

/// Read entire file contents as bytes
pub async fn read<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut file = File::open(path).await?;
    file.read_to_end().await
}

/// Write bytes to a file (creates/truncates)
pub async fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    let (path, contents) = (path.as_ref().to_path_buf(), contents.as_ref().to_vec());
    on_pool(move || std::fs::write(path, contents)).await
}

/// Write string to a file (creates/truncates)
pub async fn write_str<P: AsRef<Path>>(path: P, contents: &str) -> io::Result<()> {
    write(path, contents.as_bytes()).await
}

/// Append data to a file
pub async fn append<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    let (path, contents) = (path.as_ref().to_path_buf(), contents.as_ref().to_vec());
    on_pool(move || {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?
            .write_all(&contents)
    })
    .await
}

/// Append string to a file
pub async fn append_str<P: AsRef<Path>>(path: P, contents: &str) -> io::Result<()> {
    append(path, contents.as_bytes()).await
}

/// Copy file from source to destination
pub async fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
    let (from, to) = (from.as_ref().to_path_buf(), to.as_ref().to_path_buf());
    on_pool(move || std::fs::copy(from, to)).await
}

/// Get file metadata
pub async fn metadata<P: AsRef<Path>>(path: P) -> io::Result<std::fs::Metadata> {
    let path = path.as_ref().to_path_buf();
    on_pool(move || std::fs::metadata(path)).await
}

/// Rename a file or directory
pub async fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()> {
    let (from, to) = (from.as_ref().to_path_buf(), to.as_ref().to_path_buf());
    on_pool(move || std::fs::rename(from, to)).await
}

/// Remove a file
pub async fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()> {
    on_path(path, std::fs::remove_file).await
}

/// Create a directory
pub async fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    on_path(path, std::fs::create_dir).await
}

/// Create directories recursively
pub async fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    on_path(path, std::fs::create_dir_all).await
}

/// Remove a directory
pub async fn remove_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    on_path(path, std::fs::remove_dir).await
}

/// Remove a directory and all its contents
pub async fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    on_path(path, std::fs::remove_dir_all).await
}

/// Run a single-path `std::fs` operation on the pool.
async fn on_path<P: AsRef<Path>>(
    path: P,
    operation: fn(PathBuf) -> io::Result<()>,
) -> io::Result<()> {
    let path = path.as_ref().to_path_buf();
    on_pool(move || operation(path)).await
}
