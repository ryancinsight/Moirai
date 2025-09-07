//! Async file I/O primitives for Moirai concurrency library.
//!
//! This module provides async file operations with comprehensive error handling
//! and performance monitoring. Following SLAP principle with focused
//! responsibility on file system operations.

use std::io;
use std::path::{Path, PathBuf};
use tokio::fs::{File as TokioFile, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

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

/// High-performance async file handle with buffering and statistics
pub struct File {
    inner: TokioFile,
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
        options: FileOpenOptions
    ) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        
        let mut open_options = OpenOptions::new();
        open_options
            .read(options.read)
            .write(options.write)
            .create(options.create)
            .append(options.append)
            .truncate(options.truncate);

        if let Some(mode) = options.mode {
            #[cfg(unix)]
            {
                #[allow(unused_imports)]
                use std::os::unix::fs::OpenOptionsExt;
                open_options.mode(mode);
            }
        }

        let inner = open_options.open(&path_buf).await?;
        
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
        self.inner.write_all(buf).await?;
        self.stats.bytes_written += buf.len() as u64;
        self.stats.write_operations += 1;
        Ok(())
    }

    /// Write a string to the file
    pub async fn write_str(&mut self, s: &str) -> io::Result<()> {
        self.write_all(s.as_bytes()).await
    }

    /// Flush any buffered data to disk
    pub async fn flush(&mut self) -> io::Result<()> {
        self.inner.flush().await
    }

    /// Synchronize all data and metadata to disk
    pub async fn sync_all(&mut self) -> io::Result<()> {
        self.inner.sync_all().await
    }

    /// Synchronize data (but not metadata) to disk
    pub async fn sync_data(&mut self) -> io::Result<()> {
        self.inner.sync_data().await
    }

    /// Seek to a specific position in the file
    pub async fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let new_pos = self.inner.seek(pos).await?;
        self.stats.seek_operations += 1;
        Ok(new_pos)
    }

    /// Get current position in the file
    pub async fn stream_position(&mut self) -> io::Result<u64> {
        self.inner.stream_position().await
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

/// Convenience functions for common file operations

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

/// Write string to a file (creates/truncates)
pub async fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    let mut file = File::create(path).await?;
    file.write_all(contents.as_ref()).await?;
    file.sync_all().await
}

/// Write string to a file (creates/truncates)
pub async fn write_str<P: AsRef<Path>>(path: P, contents: &str) -> io::Result<()> {
    write(path, contents.as_bytes()).await
}

/// Append data to a file
pub async fn append<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    let mut file = File::open_with_options(path, FileOpenOptions::append_only()).await?;
    file.write_all(contents.as_ref()).await?;
    file.sync_all().await
}

/// Append string to a file
pub async fn append_str<P: AsRef<Path>>(path: P, contents: &str) -> io::Result<()> {
    append(path, contents.as_bytes()).await
}

/// Copy file from source to destination
pub async fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64> {
    let mut source = File::open(from).await?;
    let mut dest = File::create(to).await?;
    
    let mut buffer = vec![0u8; 64 * 1024]; // 64KB buffer
    let mut total_bytes = 0u64;
    
    loop {
        let bytes_read = source.read(&mut buffer).await?;
        if bytes_read == 0 {
            break;
        }
        
        dest.write_all(&buffer[..bytes_read]).await?;
        total_bytes += bytes_read as u64;
    }
    
    dest.sync_all().await?;
    Ok(total_bytes)
}

/// Remove a file
pub async fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()> {
    tokio::fs::remove_file(path).await
}

/// Create a directory
pub async fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    tokio::fs::create_dir(path).await
}

/// Create directories recursively
pub async fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    tokio::fs::create_dir_all(path).await
}

/// Remove a directory
pub async fn remove_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    tokio::fs::remove_dir(path).await
}

/// Remove a directory and all its contents
pub async fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()> {
    tokio::fs::remove_dir_all(path).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_file_read_write() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        
        // Write data
        write_str(&file_path, "Hello, World!").await.unwrap();
        
        // Read data
        let contents = read_to_string(&file_path).await.unwrap();
        assert_eq!(contents, "Hello, World!");
    }

    #[tokio::test]
    async fn test_file_append() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("append_test.txt");
        
        // Write initial data
        write_str(&file_path, "Hello").await.unwrap();
        
        // Append data
        append_str(&file_path, " World!").await.unwrap();
        
        // Read combined data
        let contents = read_to_string(&file_path).await.unwrap();
        assert_eq!(contents, "Hello World!");
    }

    #[tokio::test]
    async fn test_file_statistics() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("stats_test.txt");
        
        let mut file = File::create(&file_path).await.unwrap();
        
        // Write some data
        file.write_str("test data").await.unwrap();
        
        let stats = file.stats();
        assert_eq!(stats.bytes_written, 9);
        assert_eq!(stats.write_operations, 1);
        
        // Read the data back
        let mut read_file = File::open(&file_path).await.unwrap();
        let _contents = read_file.read_to_string().await.unwrap();
        
        let read_stats = read_file.stats();
        assert_eq!(read_stats.bytes_read, 9);
        assert_eq!(read_stats.read_operations, 1);
    }

    #[tokio::test]
    async fn test_file_copy() {
        let dir = tempdir().unwrap();
        let source_path = dir.path().join("source.txt");
        let dest_path = dir.path().join("dest.txt");
        
        // Create source file
        write_str(&source_path, "copy test data").await.unwrap();
        
        // Copy file
        let bytes_copied = copy(&source_path, &dest_path).await.unwrap();
        assert_eq!(bytes_copied, 14);
        
        // Verify destination
        let dest_contents = read_to_string(&dest_path).await.unwrap();
        assert_eq!(dest_contents, "copy test data");
    }

    #[tokio::test]
    async fn test_file_seek() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("seek_test.txt");
        
        let mut file = File::create(&file_path).await.unwrap();
        file.write_str("0123456789").await.unwrap();
        
        // Seek to position 5
        let pos = file.seek(io::SeekFrom::Start(5)).await.unwrap();
        assert_eq!(pos, 5);
        
        // Write at position 5
        file.write_str("ABC").await.unwrap();
        file.sync_all().await.unwrap();
        
        // Read entire file
        let mut read_file = File::open(&file_path).await.unwrap();
        let contents = read_file.read_to_string().await.unwrap();
        assert_eq!(contents, "01234ABC89");
    }

    #[tokio::test]
    async fn test_directory_operations() {
        let dir = tempdir().unwrap();
        let subdir_path = dir.path().join("subdir");
        
        // Create directory
        create_dir(&subdir_path).await.unwrap();
        
        // Verify directory exists
        let metadata = tokio::fs::metadata(&subdir_path).await.unwrap();
        assert!(metadata.is_dir());
        
        // Remove directory
        remove_dir(&subdir_path).await.unwrap();
        
        // Verify directory is removed
        assert!(tokio::fs::metadata(&subdir_path).await.is_err());
    }
}