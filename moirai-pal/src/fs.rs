//! Platform-agnostic async file I/O operations.

use std::fs::{File as StdFile, OpenOptions as StdOpenOptions};
use std::future::Future;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Future that yields to the executor exactly once, then resolves.
pub struct YieldFuture {
    yielded: bool,
}

impl Future for YieldFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Yield to the executor once before resuming (cooperative scheduling point).
pub fn yield_now() -> YieldFuture {
    YieldFuture { yielded: false }
}

/// Declarative open-mode configuration for [`AsyncFile::open_with`].
///
/// Replaces the five positional booleans of
/// [`AsyncFile::open_with_options`] with named fields, so call sites cannot
/// transpose modes silently (boolean-blindness).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileOpenOptions {
    /// Open for reading.
    pub read: bool,
    /// Open for writing.
    pub write: bool,
    /// Create the file if it does not exist.
    pub create: bool,
    /// Append instead of overwriting.
    pub append: bool,
    /// Truncate existing content.
    pub truncate: bool,
}

impl Default for FileOpenOptions {
    fn default() -> Self {
        Self::read_only()
    }
}

impl FileOpenOptions {
    /// Read-only access.
    #[must_use]
    pub const fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            create: false,
            append: false,
            truncate: false,
        }
    }

    /// Write-only access (creates if absent, truncates existing content).
    #[must_use]
    pub const fn write_only() -> Self {
        Self {
            read: false,
            write: true,
            create: true,
            append: false,
            truncate: true,
        }
    }

    /// Append access (creates if absent, preserves existing content).
    #[must_use]
    pub const fn append_only() -> Self {
        Self {
            read: false,
            write: true,
            create: true,
            append: true,
            truncate: false,
        }
    }

    /// Read-write access (creates if absent, preserves existing content).
    #[must_use]
    pub const fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            create: true,
            append: false,
            truncate: false,
        }
    }

    /// Read-write access that truncates existing content (creates if absent).
    #[must_use]
    pub const fn read_write_truncate() -> Self {
        Self {
            read: true,
            write: true,
            create: true,
            append: false,
            truncate: true,
        }
    }
}

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

/// High-performance cooperative async file handle
pub struct AsyncFile {
    inner: StdFile,
}

impl AsyncFile {
    /// Open `path` read-only.
    ///
    /// # Errors
    /// Propagates the underlying open error.
    pub async fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        yield_now().await;
        let inner = StdFile::open(path)?;
        Ok(Self { inner })
    }

    /// Open `path` with the modes described by `options`.
    ///
    /// # Errors
    /// Propagates the underlying open error.
    pub async fn open_with<P: AsRef<Path>>(path: P, options: FileOpenOptions) -> io::Result<Self> {
        yield_now().await;
        let mut opts = StdOpenOptions::new();
        opts.read(options.read)
            .write(options.write)
            .create(options.create)
            .append(options.append)
            .truncate(options.truncate);
        let inner = opts.open(path)?;
        Ok(Self { inner })
    }

    /// Open `path` with positional mode flags.
    ///
    /// Prefer [`AsyncFile::open_with`]; this positional form remains only for
    /// the `moirai-async` caller pending its swap to the struct-taking API.
    ///
    /// # Errors
    /// Propagates the underlying open error.
    pub async fn open_with_options<P: AsRef<Path>>(
        path: P,
        read: bool,
        write: bool,
        create: bool,
        append: bool,
        truncate: bool,
    ) -> io::Result<Self> {
        Self::open_with(
            path,
            FileOpenOptions {
                read,
                write,
                create,
                append,
                truncate,
            },
        )
        .await
    }

    /// Poll one read into `buf` (always ready: file I/O is synchronous here).
    pub fn poll_read(&mut self, _cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<usize>> {
        Poll::Ready((&self.inner).read(buf))
    }

    /// Poll one write of `buf` (always ready: file I/O is synchronous here).
    pub fn poll_write(&mut self, _cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        Poll::Ready((&self.inner).write(buf))
    }

    /// Poll a flush (always ready: file I/O is synchronous here).
    pub fn poll_flush(&mut self, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.inner.flush())
    }

    /// Read into `buf`.
    ///
    /// # Errors
    /// Propagates the underlying read error.
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        yield_now().await;
        (&self.inner).read(buf)
    }

    /// Read the remaining content into `buf` as UTF-8.
    ///
    /// # Errors
    /// Propagates the underlying read error.
    pub async fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        yield_now().await;
        (&self.inner).read_to_string(buf)
    }

    /// Read the remaining content into `buf`.
    ///
    /// # Errors
    /// Propagates the underlying read error.
    pub async fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        yield_now().await;
        (&self.inner).read_to_end(buf)
    }

    /// Write `buf`.
    ///
    /// # Errors
    /// Propagates the underlying write error.
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        yield_now().await;
        (&self.inner).write(buf)
    }

    /// Flush buffered writes.
    ///
    /// # Errors
    /// Propagates the underlying flush error.
    pub async fn flush(&mut self) -> io::Result<()> {
        yield_now().await;
        self.inner.flush()
    }

    /// Seek to `pos`.
    ///
    /// # Errors
    /// Propagates the underlying seek error.
    pub async fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        yield_now().await;
        self.inner.seek(pos)
    }

    /// Synchronize data and metadata to disk.
    ///
    /// # Errors
    /// Propagates the underlying sync error.
    pub async fn sync_all(&mut self) -> io::Result<()> {
        yield_now().await;
        self.inner.sync_all()
    }

    /// Synchronize data (not necessarily metadata) to disk.
    ///
    /// # Errors
    /// Propagates the underlying sync error.
    pub async fn sync_data(&mut self) -> io::Result<()> {
        yield_now().await;
        self.inner.sync_data()
    }

    /// File metadata.
    ///
    /// # Errors
    /// Propagates the underlying metadata error.
    pub async fn metadata(&self) -> io::Result<std::fs::Metadata> {
        yield_now().await;
        self.inner.metadata()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "moirai_pal_async_file_{name}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn async_file_roundtrip_seek_and_metadata_are_value_semantic() {
        let path = test_path("roundtrip.bin");
        block_on(async {
            let mut file = AsyncFile::open_with(&path, FileOpenOptions::read_write_truncate())
                .await
                .expect("file create must succeed");
            let written = file.write(b"alpha-beta").await.expect("write must succeed");
            assert_eq!(written, 10);
            file.flush().await.expect("flush must succeed");

            let position = file
                .seek(SeekFrom::Start(6))
                .await
                .expect("seek must succeed");
            assert_eq!(position, 6);

            let mut suffix = [0_u8; 4];
            let read = file.read(&mut suffix).await.expect("read must succeed");
            assert_eq!(read, 4);
            assert_eq!(&suffix, b"beta");

            let metadata = file.metadata().await.expect("metadata must succeed");
            assert_eq!(metadata.len(), 10);
        });
        std::fs::remove_file(&path).expect("test file cleanup must succeed");
    }

    #[test]
    fn async_file_read_to_end_preserves_source_bytes() {
        let path = test_path("source.bin");
        let expected: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(3)).collect();
        std::fs::write(&path, &expected).expect("source write must succeed");

        block_on(async {
            let mut file = AsyncFile::open(&path).await.expect("open must succeed");
            let mut actual = Vec::new();
            let read = file
                .read_to_end(&mut actual)
                .await
                .expect("read_to_end must succeed");
            assert_eq!(read, expected.len());
            assert_eq!(actual, expected);
        });

        std::fs::remove_file(&path).expect("test file cleanup must succeed");
    }

    #[test]
    fn async_file_copy_preserves_source_bytes() {
        let source = test_path("copy-source.bin");
        let dest = test_path("copy-dest.bin");
        let expected: Vec<u8> = (0_u8..=63).map(|value| value.wrapping_mul(5)).collect();
        std::fs::write(&source, &expected).expect("source write must succeed");

        block_on(async {
            let copied = copy(&source, &dest).await.expect("copy must succeed");
            assert_eq!(copied, expected.len() as u64);
            let actual = std::fs::read(&dest).expect("dest read must succeed");
            assert_eq!(actual, expected);
        });

        std::fs::remove_file(&source).expect("source cleanup must succeed");
        std::fs::remove_file(&dest).expect("dest cleanup must succeed");
    }

    #[test]
    fn async_file_write_preserves_source_bytes() {
        let path = test_path("write.bin");
        let expected: Vec<u8> = (0_u8..=127).map(|value| value.wrapping_mul(7)).collect();

        block_on(async {
            write(&path, &expected).await.expect("write must succeed");
            let actual = std::fs::read(&path).expect("written file must be readable");
            assert_eq!(actual, expected);
        });

        std::fs::remove_file(&path).expect("written file cleanup must succeed");
    }

    #[test]
    fn async_file_append_preserves_prefix_and_appended_bytes() {
        let path = test_path("append.bin");
        let prefix: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(3)).collect();
        let suffix: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(11)).collect();
        std::fs::write(&path, &prefix).expect("prefix write must succeed");

        block_on(async {
            append(&path, &suffix).await.expect("append must succeed");
            let actual = std::fs::read(&path).expect("appended file must be readable");
            assert_eq!(&actual[..prefix.len()], prefix.as_slice());
            assert_eq!(&actual[prefix.len()..], suffix.as_slice());
        });

        std::fs::remove_file(&path).expect("appended file cleanup must succeed");
    }

    #[test]
    fn async_file_metadata_preserves_file_type_and_length() {
        let path = test_path("metadata.bin");
        let expected: Vec<u8> = (0_u8..=95).map(|value| value.wrapping_mul(13)).collect();
        std::fs::write(&path, &expected).expect("metadata source write must succeed");

        block_on(async {
            let actual = metadata(&path).await.expect("metadata must succeed");
            assert!(actual.is_file());
            assert_eq!(actual.len(), expected.len() as u64);
        });

        std::fs::remove_file(&path).expect("metadata file cleanup must succeed");
    }

    #[test]
    fn async_file_rename_preserves_source_bytes_at_destination() {
        let source = test_path("rename-source.bin");
        let dest = test_path("rename-dest.bin");
        let expected: Vec<u8> = (0_u8..=79).map(|value| value.wrapping_mul(17)).collect();
        std::fs::write(&source, &expected).expect("rename source write must succeed");

        block_on(async {
            rename(&source, &dest).await.expect("rename must succeed");
            assert!(!source.exists());
            let actual = std::fs::read(&dest).expect("renamed dest read must succeed");
            assert_eq!(actual, expected);
        });

        std::fs::remove_file(&dest).expect("renamed file cleanup must succeed");
    }

    #[test]
    fn async_file_remove_file_deletes_expected_path() {
        let path = test_path("remove.bin");
        let expected: Vec<u8> = (0_u8..=47).map(|value| value.wrapping_mul(19)).collect();
        std::fs::write(&path, &expected).expect("remove source write must succeed");

        block_on(async {
            let actual = std::fs::read(&path).expect("remove source read must succeed");
            assert_eq!(actual, expected);
            remove_file(&path).await.expect("remove_file must succeed");
            assert!(!path.exists());
        });
    }

    #[test]
    fn async_dir_create_and_remove_preserves_directory_state() {
        let dir = test_path("dir");

        block_on(async {
            create_dir(&dir).await.expect("create_dir must succeed");
            let metadata = std::fs::metadata(&dir).expect("created dir metadata must exist");
            assert!(metadata.is_dir());
            remove_dir(&dir).await.expect("remove_dir must succeed");
            assert!(!dir.exists());
        });
    }

    #[test]
    fn async_dir_all_create_and_remove_deletes_nested_tree() {
        let root = test_path("dir-all");
        let leaf = root.join("alpha").join("beta");
        let marker = leaf.join("marker.bin");
        let expected: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(23)).collect();

        block_on(async {
            create_dir_all(&leaf)
                .await
                .expect("create_dir_all must succeed");
            assert!(leaf.is_dir());
            std::fs::write(&marker, &expected).expect("nested marker write must succeed");
            let actual = std::fs::read(&marker).expect("nested marker read must succeed");
            assert_eq!(actual, expected);
            remove_dir_all(&root)
                .await
                .expect("remove_dir_all must succeed");
            assert!(!root.exists());
        });
    }
}
