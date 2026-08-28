use std::fs::{File as StdFile, OpenOptions as StdOpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::task::{Context, Poll};

#[cfg(windows)]
use std::sync::Mutex;

use super::yield_now;

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

/// Cooperative file handle with stream and native positioned-read operations.
pub struct AsyncFile {
    inner: StdFile,
    // `seek_read` mutates the Windows file cursor. Stateful methods require
    // `&mut self`, so safe callers can overlap only positioned reads; this lock
    // serializes exactly those save/read/restore sequences.
    #[cfg(windows)]
    positioned_read_lock: Mutex<()>,
}

impl AsyncFile {
    /// Open `path` read-only.
    ///
    /// # Errors
    /// Propagates the underlying open error.
    pub async fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        yield_now().await;
        let inner = StdFile::open(path)?;
        Ok(Self {
            inner,
            #[cfg(windows)]
            positioned_read_lock: Mutex::new(()),
        })
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
        Ok(Self {
            inner,
            #[cfg(windows)]
            positioned_read_lock: Mutex::new(()),
        })
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

    /// Read bytes at an absolute offset without changing the stream cursor.
    ///
    /// A successful call may read fewer bytes than requested. Unix provides a
    /// cursor-independent primitive. Windows `seek_read` changes the cursor,
    /// so concurrent positioned reads serialize only their save/read/restore
    /// sequence; ordinary stream operations retain their existing lock-free
    /// representation. Targets without a native primitive report
    /// [`io::ErrorKind::Unsupported`].
    ///
    /// # Errors
    ///
    /// Propagates the platform read error. On Windows, also reports a poisoned
    /// positioned-read lock or failure to restore the original cursor.
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;

            self.inner.read_at(buf, offset)
        }

        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;

            let _positioned_read = self
                .positioned_read_lock
                .lock()
                .map_err(|_| io::Error::other("positioned-read lock was poisoned"))?;
            let mut file = &self.inner;
            let original = file.stream_position()?;
            let read = self.inner.seek_read(buf, offset);
            file.seek(SeekFrom::Start(original))?;
            read
        }

        #[cfg(not(any(unix, windows)))]
        {
            let _ = (buf, offset);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "positioned file reads are unsupported on this target",
            ))
        }
    }
}
