use moirai_pal::fs::AsyncFile;
use std::io::{self, SeekFrom};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::fs::options::FileOpenOptions;
use crate::fs::stats::FileStats;
use crate::io::{AsyncRead, AsyncWrite};

/// High-performance async file handle with native implementation
pub struct File {
    inner: AsyncFile,
    path: PathBuf,
    stats: FileStats,
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
