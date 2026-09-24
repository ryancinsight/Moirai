//! Async file handle over the file-system blocking pool.
//!
//! Every syscall runs on the file-system blocking pool, never inside `poll`. A handle has
//! at most one stream operation in flight, and every operation first settles
//! the one before it, so stream operations take effect in call order even when
//! a caller drops a future mid-flight.
//!
//! Cancellation keeps stream semantics:
//! - A read whose future is dropped still consumes its bytes. They are kept
//!   and delivered, in order, to the next read.
//! - A write, seek, or sync submitted before its future is dropped still takes
//!   effect. Writes run even if the handle itself is dropped.
//! - An error from an operation whose caller has gone is returned by the next
//!   operation on the handle, which then does not run.
//!
//! `AsyncWrite::poll_write` queues the write and reports it complete. Its
//! failure, if any, is returned by the next operation, as for a dropped write.
//! [`File::write`] and [`File::write_all`] wait for the write itself.

use std::future::{Future, poll_fn};
use std::io::{self, SeekFrom};
use std::mem;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use moirai_pal::fs::{File as Handle, FileOpenOptions};

use crate::blocking::{Abandoned, Admission, Completion};
use crate::fs::pool;
use crate::fs::stats::FileStats;
use crate::io::{AsyncLength, AsyncRead, AsyncReadAt, AsyncWrite};

mod request;

use request::{Outcome, Request};

/// A pending admission, stored so repeated polls continue the same wait.
type Admitting = Pin<Box<dyn Future<Output = io::Result<Admission>> + Send + Sync>>;

enum State {
    Idle,
    /// Waiting for a pool slot for the next operation.
    Admitting(Admitting),
    /// A stream operation is queued or running on the pool.
    Busy(Completion<Outcome>),
}

/// What settling a finished operation left behind.
enum Settled {
    /// Nothing was in flight, or its outcome was absorbed.
    Absorbed,
    /// A read reached the end of the file.
    EndOfFile,
}

/// Async file handle supporting stateful streams and positioned reads.
pub struct File {
    handle: Arc<Handle>,
    state: State,
    /// Bytes read from the file that no caller has taken yet. The OS cursor is
    /// past them, so cursor-relative operations rewind over them first.
    unread: Vec<u8>,
    /// The error of an operation that finished after its caller left.
    deferred: Option<io::Error>,
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
        let path = path.as_ref().to_path_buf();
        let target = path.clone();
        let handle = pool()
            .run(Abandoned::Skip, move || Handle::open_with(target, options))
            .await??;
        Ok(Self {
            handle: Arc::new(handle),
            state: State::Idle,
            unread: Vec::new(),
            deferred: None,
            path,
            stats: FileStats::default(),
        })
    }

    /// Read entire file contents into a string
    pub async fn read_to_string(&mut self) -> io::Result<String> {
        let contents = String::from_utf8(self.read_all().await?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        Ok(contents)
    }

    /// Read entire file contents into a byte vector
    pub async fn read_to_end(&mut self) -> io::Result<Vec<u8>> {
        self.read_all().await
    }

    async fn read_all(&mut self) -> io::Result<Vec<u8>> {
        let Outcome::Read(contents) = self
            .perform(Request::ReadToEnd { prefix: Vec::new() })
            .await?
        else {
            unreachable!("invariant: a read-to-end request yields a read outcome");
        };
        let contents = contents?;
        self.stats.bytes_read += contents.len() as u64;
        self.stats.read_operations += 1;
        Ok(contents)
    }

    /// Read data into a buffer
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = poll_fn(|cx| self.poll_read_into(cx, buf)).await?;
        self.stats.bytes_read += bytes_read as u64;
        self.stats.read_operations += 1;
        Ok(bytes_read)
    }

    /// Write data from a buffer
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let Outcome::Wrote(written) = self
            .perform(Request::Write {
                data: buf.to_vec(),
                rewind: 0,
            })
            .await?
        else {
            unreachable!("invariant: a write request yields a write outcome");
        };
        let bytes_written = written?;
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
        poll_fn(|cx| self.poll_settled(cx)).await
    }

    /// Synchronize all data and metadata to disk
    pub async fn sync_all(&mut self) -> io::Result<()> {
        self.perform_unit(Request::SyncAll).await
    }

    /// Synchronize data (but not metadata) to disk
    pub async fn sync_data(&mut self) -> io::Result<()> {
        self.perform_unit(Request::SyncData).await
    }

    async fn perform_unit(&mut self, request: Request) -> io::Result<()> {
        let Outcome::Done(done) = self.perform(request).await? else {
            unreachable!("invariant: a sync request yields a done outcome");
        };
        done
    }

    /// Seek to a specific position in the file
    pub async fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let Outcome::Sought(position) = self.perform(Request::Seek(pos)).await? else {
            unreachable!("invariant: a seek request yields a seek outcome");
        };
        let new_pos = position?;
        self.stats.seek_operations += 1;
        Ok(new_pos)
    }

    /// Get current position in the file
    pub async fn stream_position(&mut self) -> io::Result<u64> {
        let Outcome::Sought(position) = self.perform(Request::Seek(SeekFrom::Current(0))).await?
        else {
            unreachable!("invariant: a seek request yields a seek outcome");
        };
        position
    }

    /// Get file metadata
    pub async fn metadata(&self) -> io::Result<std::fs::Metadata> {
        let handle = Arc::clone(&self.handle);
        pool()
            .run(Abandoned::Skip, move || handle.metadata())
            .await?
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get statistics for stateful stream operations.
    ///
    /// Positioned reads do not advance the stream and are not included.
    pub fn stats(&self) -> &FileStats {
        &self.stats
    }

    /// Settle the previous operation, then run `request` and return its
    /// outcome. `Err` means the request did not run: a deferred error, an
    /// admission failure, or a panicked job.
    async fn perform(&mut self, request: Request) -> io::Result<Outcome> {
        let mut request = Some(request);
        poll_fn(|cx| {
            if request.is_some() {
                ready!(self.poll_settle(cx));
                if let Some(error) = self.deferred.take() {
                    request = None;
                    return Poll::Ready(Err(error));
                }
                let admission = ready!(self.poll_admission(cx))?;
                let pending = request
                    .take()
                    .expect("invariant: the request is submitted at most once, checked above");
                self.submit(admission, pending)?;
            }
            let State::Busy(completion) = &mut self.state else {
                unreachable!(
                    "invariant: this call's request is in flight while it borrows the file"
                );
            };
            let outcome = ready!(Pin::new(completion).poll(cx));
            self.state = State::Idle;
            Poll::Ready(outcome)
        })
        .await
    }

    /// Queue `request`, first adjusting it for bytes read but not delivered.
    fn submit(&mut self, admission: Admission, request: Request) -> io::Result<()> {
        let request = match request {
            Request::ReadToEnd { .. } => Request::ReadToEnd {
                prefix: mem::take(&mut self.unread),
            },
            Request::Write { data, .. } => Request::Write {
                data,
                rewind: self.take_unread_len(),
            },
            Request::WriteAll { data, .. } => Request::WriteAll {
                data,
                rewind: self.take_unread_len(),
            },
            Request::Seek(SeekFrom::Current(offset)) => {
                let rewind = i64::try_from(self.unread.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "unread bytes exceed a seek offset",
                    )
                })?;
                let offset = offset.checked_sub(rewind).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "seek offset overflows i64")
                })?;
                self.unread.clear();
                Request::Seek(SeekFrom::Current(offset))
            }
            Request::Seek(absolute) => {
                self.unread.clear();
                Request::Seek(absolute)
            }
            other => other,
        };
        let handle = Arc::clone(&self.handle);
        let abandoned = request.abandoned();
        let completion = admission.submit(abandoned, move || request.run(&handle))?;
        self.state = State::Busy(completion);
        Ok(())
    }

    fn take_unread_len(&mut self) -> u64 {
        let rewind = self.unread.len() as u64;
        self.unread.clear();
        rewind
    }

    /// Wait for any operation in flight and absorb its outcome: read bytes
    /// join the unread buffer, and an error is deferred to the next caller.
    fn poll_settle(&mut self, cx: &mut Context<'_>) -> Poll<Settled> {
        let State::Busy(completion) = &mut self.state else {
            return Poll::Ready(Settled::Absorbed);
        };
        let outcome = ready!(Pin::new(completion).poll(cx));
        self.state = State::Idle;
        Poll::Ready(match outcome {
            Ok(Outcome::Read(Ok(data))) if data.is_empty() => Settled::EndOfFile,
            Ok(Outcome::Read(Ok(data))) => {
                self.unread.extend_from_slice(&data);
                Settled::Absorbed
            }
            Ok(outcome) => {
                if let Some(error) = outcome.into_error() {
                    self.defer(error);
                }
                Settled::Absorbed
            }
            Err(error) => {
                self.defer(error);
                Settled::Absorbed
            }
        })
    }

    /// Settle the previous operation and report its deferred error, if any.
    fn poll_settled(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        ready!(self.poll_settle(cx));
        Poll::Ready(self.deferred.take().map_or(Ok(()), Err))
    }

    fn defer(&mut self, error: io::Error) {
        if self.deferred.is_none() {
            self.deferred = Some(error);
        }
    }

    fn poll_admission(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<Admission>> {
        if matches!(self.state, State::Idle) {
            self.state = State::Admitting(Box::pin(pool().admit()));
        }
        let State::Admitting(admitting) = &mut self.state else {
            unreachable!("invariant: callers settle a busy state before admission");
        };
        let admitted = ready!(admitting.as_mut().poll(cx));
        self.state = State::Idle;
        Poll::Ready(admitted)
    }

    fn poll_read_into(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<usize>> {
        loop {
            if !self.unread.is_empty() {
                let delivered = self.unread.len().min(buf.len());
                buf[..delivered].copy_from_slice(&self.unread[..delivered]);
                self.unread.drain(..delivered);
                return Poll::Ready(Ok(delivered));
            }
            if buf.is_empty() {
                return Poll::Ready(Ok(0));
            }
            if matches!(self.state, State::Busy(_)) {
                if let Settled::EndOfFile = ready!(self.poll_settle(cx)) {
                    return Poll::Ready(Ok(0));
                }
                continue;
            }
            if let Some(error) = self.deferred.take() {
                return Poll::Ready(Err(error));
            }
            let admission = ready!(self.poll_admission(cx))?;
            self.submit(admission, Request::Read { len: buf.len() })?;
        }
    }

    fn poll_write_behind(&mut self, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        ready!(self.poll_settled(cx))?;
        let admission = ready!(self.poll_admission(cx))?;
        self.submit(
            admission,
            Request::WriteAll {
                data: buf.to_vec(),
                rewind: 0,
            },
        )?;
        Poll::Ready(Ok(buf.len()))
    }
}

impl AsyncReadAt for File {
    async fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }

        let requested = u64::try_from(buf.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "read length does not fit u64")
        })?;
        offset.checked_add(requested).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "read range overflows u64")
        })?;

        let handle = Arc::clone(&self.handle);
        let len = buf.len();
        let (data, filled) = pool()
            .run(Abandoned::Skip, move || read_exact_at(&handle, len, offset))
            .await?;
        // Bytes read before a failure still reach the caller, as with a
        // positioned read loop run in place.
        buf[..data.len()].copy_from_slice(&data);
        filled
    }
}

/// Fill `len` bytes from `offset` with positioned reads, which leave the
/// stream cursor alone. Returns the bytes read, and whether all `len` were.
fn read_exact_at(handle: &Handle, len: usize, offset: u64) -> (Vec<u8>, io::Result<()>) {
    let mut data = vec![0; len];
    let mut filled = 0;
    let result = loop {
        if filled == len {
            break Ok(());
        }
        let Some(position) = u64::try_from(filled)
            .ok()
            .and_then(|filled| offset.checked_add(filled))
        else {
            break Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read range overflows u64",
            ));
        };
        match handle.read_at(&mut data[filled..], position) {
            Ok(0) => {
                break Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "positioned read reached end of file before filling buffer",
                ));
            }
            Ok(count) => filled += count,
            Err(error) => break Err(error),
        }
    };
    data.truncate(filled);
    (data, result)
}

impl AsyncLength for File {
    async fn len(&self) -> io::Result<u64> {
        Ok(self.metadata().await?.len())
    }
}

impl AsyncRead for File {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let n = ready!(self.poll_read_into(cx, buf))?;
        self.stats.bytes_read += n as u64;
        self.stats.read_operations += 1;
        Poll::Ready(Ok(n))
    }
}

impl AsyncWrite for File {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let n = ready!(self.poll_write_behind(cx, buf))?;
        self.stats.bytes_written += n as u64;
        self.stats.write_operations += 1;
        Poll::Ready(Ok(n))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_settled(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_settled(cx)
    }
}
