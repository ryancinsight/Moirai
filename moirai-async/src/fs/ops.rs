use std::io;
use std::path::Path;

use crate::fs::file::File;

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
    moirai_pal::fs::write(path, contents).await
}

/// Write string to a file (creates/truncates)
pub async fn write_str<P: AsRef<Path>>(path: P, contents: &str) -> io::Result<()> {
    write(path, contents.as_bytes()).await
}

/// Append data to a file
pub async fn append<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    moirai_pal::fs::append(path, contents).await
}

/// Append string to a file
pub async fn append_str<P: AsRef<Path>>(path: P, contents: &str) -> io::Result<()> {
    append(path, contents.as_bytes()).await
}

/// Copy file from source to destination
pub async fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
    moirai_pal::fs::copy(from, to).await
}

/// Get file metadata
pub async fn metadata<P: AsRef<Path>>(path: P) -> io::Result<std::fs::Metadata> {
    moirai_pal::fs::metadata(path).await
}

/// Rename a file or directory
pub async fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()> {
    moirai_pal::fs::rename(from, to).await
}

/// Remove a file
pub async fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()> {
    moirai_pal::fs::remove_file(path).await
}

/// Create a directory
pub async fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    moirai_pal::fs::create_dir(path).await
}

/// Create directories recursively
pub async fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    moirai_pal::fs::create_dir_all(path).await
}

/// Remove a directory
pub async fn remove_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    moirai_pal::fs::remove_dir(path).await
}

/// Remove a directory and all its contents
pub async fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    moirai_pal::fs::remove_dir_all(path).await
}
