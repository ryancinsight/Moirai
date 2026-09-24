use std::fs::{File as StdFile, OpenOptions as StdOpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

#[cfg(windows)]
use std::sync::Mutex;

/// Declarative open-mode configuration for [`File::open_with`].
///
/// Named fields keep call sites from transposing modes silently.
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

/// Platform file handle whose stream and positioned operations all take
/// `&self`, so one handle can be shared with the threads that run blocking
/// file I/O.
///
/// Every call is a blocking syscall. Async callers run them on a blocking
/// pool (`moirai_async::fs`), never inside `poll`.
pub struct File {
    inner: StdFile,
    // Windows `seek_read` moves the file cursor, so a positioned read saves and
    // restores it. Stream operations take the same lock, so no read, write, or
    // seek observes a positioned read's temporary cursor. Unix `pread` leaves
    // the cursor alone and needs no lock.
    #[cfg(windows)]
    cursor_lock: Mutex<()>,
}

impl File {
    /// Open `path` with the modes described by `options`.
    ///
    /// # Errors
    /// Propagates the underlying open error.
    pub fn open_with<P: AsRef<Path>>(path: P, options: FileOpenOptions) -> io::Result<Self> {
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
            cursor_lock: Mutex::new(()),
        })
    }

    /// Run a cursor-moving stream operation, excluding positioned reads on
    /// Windows.
    fn with_cursor<T>(&self, operation: impl FnOnce(&StdFile) -> io::Result<T>) -> io::Result<T> {
        #[cfg(windows)]
        let _cursor = self
            .cursor_lock
            .lock()
            .map_err(|_| io::Error::other("file cursor lock was poisoned"))?;
        operation(&self.inner)
    }

    /// Read into `buf` at the cursor.
    ///
    /// # Errors
    /// Propagates the underlying read error.
    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.with_cursor(|mut file| file.read(buf))
    }

    /// Read from the cursor to the end of the file, appending to `buf`.
    ///
    /// # Errors
    /// Propagates the underlying read error.
    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.with_cursor(|mut file| file.read_to_end(buf))
    }

    /// Write `buf` at the cursor (or the end, in append mode).
    ///
    /// # Errors
    /// Propagates the underlying write error.
    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.with_cursor(|mut file| file.write(buf))
    }

    /// Write all of `buf` at the cursor (or the end, in append mode).
    ///
    /// # Errors
    /// Propagates the underlying seek or write error.
    pub fn write_all(&self, buf: &[u8]) -> io::Result<()> {
        self.with_cursor(|mut file| file.write_all(buf))
    }

    /// Move the cursor to `pos`.
    ///
    /// # Errors
    /// Propagates the underlying seek error.
    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        self.with_cursor(|mut file| file.seek(pos))
    }

    /// Synchronize data and metadata to disk.
    ///
    /// # Errors
    /// Propagates the underlying sync error.
    pub fn sync_all(&self) -> io::Result<()> {
        self.inner.sync_all()
    }

    /// Synchronize data (not necessarily metadata) to disk.
    ///
    /// # Errors
    /// Propagates the underlying sync error.
    pub fn sync_data(&self) -> io::Result<()> {
        self.inner.sync_data()
    }

    /// File metadata.
    ///
    /// # Errors
    /// Propagates the underlying metadata error.
    pub fn metadata(&self) -> io::Result<std::fs::Metadata> {
        self.inner.metadata()
    }

    /// Read bytes at an absolute offset without changing the stream cursor.
    ///
    /// A successful call may read fewer bytes than requested. Unix provides a
    /// cursor-independent primitive. Windows `seek_read` changes the cursor,
    /// so the save/read/restore sequence holds the cursor lock that stream
    /// operations also take. Targets without a native primitive report
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

            let _cursor = self
                .cursor_lock
                .lock()
                .map_err(|_| io::Error::other("file cursor lock was poisoned"))?;
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
