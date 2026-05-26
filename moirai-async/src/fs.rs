//! Async file I/O primitives for Moirai concurrency library.
//!
//! This module provides Moirai-owned async file facade operations without Tokio
//! dependencies. The current facade executes standard-library file operations
//! to completion at the call boundary and returns ready futures; reactor-native
//! file readiness remains a separate PAL responsibility.

use moirai_pal::fs::AsyncFile;
use std::io::{self, SeekFrom};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::io::{AsyncRead, AsyncWrite};

/// Configuration for file operations
#[derive(Debug, Clone)]
pub struct FileOpenOptions {
    /// Open for reading
    pub read: bool,
    /// Open for writing
    pub write: bool,
    /// Create if not exists
    pub create: bool,
    /// Append mode
    pub append: bool,
    /// Truncate existing content
    pub truncate: bool,
    /// File permissions (Unix only)
    pub mode: Option<u32>,
}

impl Default for FileOpenOptions {
    fn default() -> Self {
        Self {
            read: true,
            write: false,
            create: false,
            append: false,
            truncate: false,
            mode: None,
        }
    }
}

impl FileOpenOptions {
    /// Create options for read-only access
    pub fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            create: false,
            append: false,
            truncate: false,
            mode: None,
        }
    }

    /// Create options for write-only access (creates if not exists)
    pub fn write_only() -> Self {
        Self {
            read: false,
            write: true,
            create: true,
            append: false,
            truncate: true,
            mode: None,
        }
    }

    /// Create options for append access
    pub fn append_only() -> Self {
        Self {
            read: false,
            write: true,
            create: true,
            append: true,
            truncate: false,
            mode: None,
        }
    }

    /// Create options for read-write access
    pub fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            create: true,
            append: false,
            truncate: false,
            mode: None,
        }
    }
}

/// High-performance async file handle with native implementation
pub struct File {
    inner: AsyncFile,
    path: PathBuf,
    buffer_size: usize,
    stats: FileStats,
}

/// Statistics for file operations
#[derive(Debug, Default, Clone)]
pub struct FileStats {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_operations: u64,
    pub write_operations: u64,
    pub seek_operations: u64,
}

impl File {
    /// Open a file with default options (read-only)
    pub async fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::open_with_options(path, FileOpenOptions::read_only()).await
    }

    /// Create a new file for writing (truncates if exists)
    pub async fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::open_with_options(path, FileOpenOptions::write_only()).await
    }

    /// Open a file with custom options
    pub async fn open_with_options<P: AsRef<Path>>(
        path: P,
        options: FileOpenOptions,
    ) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let inner = AsyncFile::open_with_options(
            &path_buf,
            options.read,
            options.write,
            options.create,
            options.append,
            options.truncate,
        )
        .await?;

        Ok(Self {
            inner,
            path: path_buf,
            buffer_size: 8192,
            stats: FileStats::default(),
        })
    }

    /// Read entire file contents into a string
    pub async fn read_to_string(&mut self) -> io::Result<String> {
        let mut contents = String::new();
        self.inner.read_to_string(&mut contents).await?;
        self.stats.bytes_read += contents.len() as u64;
        self.stats.read_operations += 1;
        Ok(contents)
    }

    /// Read entire file contents into a byte vector
    pub async fn read_to_end(&mut self) -> io::Result<Vec<u8>> {
        let mut contents = Vec::new();
        self.inner.read_to_end(&mut contents).await?;
        self.stats.bytes_read += contents.len() as u64;
        self.stats.read_operations += 1;
        Ok(contents)
    }

    /// Read data into a buffer
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.inner.read(buf).await?;
        self.stats.bytes_read += bytes_read as u64;
        self.stats.read_operations += 1;
        Ok(bytes_read)
    }

    /// Write data from a buffer
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_written = self.inner.write(buf).await?;
        self.stats.bytes_written += bytes_written as u64;
        self.stats.write_operations += 1;
        Ok(bytes_written)
    }

    /// Write all data from a buffer
    pub async fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let mut written = 0;
        while written < buf.len() {
            let n = self.write(&buf[written..]).await?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ));
            }
            written += n;
        }
        Ok(())
    }

    /// Write a string to the file
    pub async fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.write_all(s.as_bytes()).await
    }

    /// Flush any buffered data to disk
    pub async fn flush(&mut self) -> io::Result<()> {
        self.inner.flush().await?;
        Ok(())
    }

    /// Synchronize all data and metadata to disk
    pub async fn sync_all(&mut self) -> io::Result<()> {
        self.inner.sync_all().await?;
        Ok(())
    }

    /// Synchronize data (but not metadata) to disk
    pub async fn sync_data(&mut self) -> io::Result<()> {
        self.inner.sync_data().await?;
        Ok(())
    }

    /// Seek to a specific position in the file
    pub async fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = self.inner.seek(pos).await?;
        self.stats.seek_operations += 1;
        Ok(new_pos)
    }

    /// Get current position in the file
    pub async fn stream_position(&mut self) -> io::Result<u64> {
        self.inner.seek(SeekFrom::Current(0)).await
    }

    /// Get file metadata
    pub async fn metadata(&self) -> io::Result<std::fs::Metadata> {
        self.inner.metadata().await
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get file statistics
    pub fn stats(&self) -> &FileStats {
        &self.stats
    }

    /// Set buffer size for operations
    pub fn set_buffer_size(&mut self, size: usize) {
        self.buffer_size = size;
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

impl AsyncRead for File {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        match Pin::new(&mut self.inner).poll_read(cx, buf) {
            Poll::Ready(Ok(n)) => {
                self.stats.bytes_read += n as u64;
                self.stats.read_operations += 1;
                Poll::Ready(Ok(n))
            }
            res => res,
        }
    }
}

impl AsyncWrite for File {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match Pin::new(&mut self.inner).poll_write(cx, buf) {
            Poll::Ready(Ok(n)) => {
                self.stats.bytes_written += n as u64;
                self.stats.write_operations += 1;
                Poll::Ready(Ok(n))
            }
            res => res,
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Convenience functions for common file operations
///
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

/// Remove a file
pub async fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()> {
    std::fs::remove_file(path)
}

/// Create a directory
pub async fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    std::fs::create_dir(path)
}

/// Create directories recursively
pub async fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    std::fs::create_dir_all(path)
}

/// Remove a directory
pub async fn remove_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    std::fs::remove_dir(path)
}

/// Remove a directory and all its contents
pub async fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    std::fs::remove_dir_all(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "moirai_async_fs_{name}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn test_file_options() {
        let options = FileOpenOptions::read_only();
        assert!(options.read);
        assert!(!options.write);

        let options = FileOpenOptions::write_only();
        assert!(!options.read);
        assert!(options.write);

        let options = FileOpenOptions::append_only();
        assert!(!options.read);
        assert!(options.write);
        assert!(options.append);
    }

    #[test]
    fn test_file_stats() {
        let stats = FileStats::default();
        assert_eq!(stats.bytes_read, 0);
        assert_eq!(stats.bytes_written, 0);
        assert_eq!(stats.read_operations, 0);
        assert_eq!(stats.write_operations, 0);
        assert_eq!(stats.seek_operations, 0);
    }

    #[test]
    fn test_file_write_read_append_and_stats_values() {
        let path = test_path("roundtrip.txt");
        futures::executor::block_on(async {
            write_str(&path, "alpha")
                .await
                .expect("write_str must succeed");
            append_str(&path, "-beta")
                .await
                .expect("append_str must succeed");

            let contents = read_to_string(&path)
                .await
                .expect("read_to_string must succeed");
            assert_eq!(contents, "alpha-beta");

            let mut file = File::open(&path).await.expect("open must succeed");
            let mut prefix = [0_u8; 5];
            let bytes_read = file.read(&mut prefix).await.expect("read must succeed");
            assert_eq!(bytes_read, 5);
            assert_eq!(&prefix, b"alpha");
            assert_eq!(file.stats().bytes_read, 5);
            assert_eq!(file.stats().read_operations, 1);

            let position = file
                .stream_position()
                .await
                .expect("stream_position must succeed");
            assert_eq!(position, 5);
            let new_position = file
                .seek(SeekFrom::Start(6))
                .await
                .expect("seek must succeed");
            assert_eq!(new_position, 6);
            assert_eq!(file.stats().seek_operations, 1);
        });
        std::fs::remove_file(&path).expect("test file cleanup must succeed");
    }

    #[test]
    fn test_file_copy_and_directory_values() {
        let dir = test_path("dir");
        let source = dir.join("source.bin");
        let dest = dir.join("dest.bin");
        futures::executor::block_on(async {
            create_dir(&dir).await.expect("create_dir must succeed");
            write(&source, b"0123456789")
                .await
                .expect("source write must succeed");
            let copied = copy(&source, &dest).await.expect("copy must succeed");
            assert_eq!(copied, 10);
            let dest_bytes = read(&dest).await.expect("read copied file must succeed");
            assert_eq!(dest_bytes, b"0123456789");
            remove_file(&source)
                .await
                .expect("remove source must succeed");
            remove_file(&dest).await.expect("remove dest must succeed");
            remove_dir(&dir).await.expect("remove dir must succeed");
        });
    }
}
