use std::{
    future::{poll_fn, Future},
    io,
    pin::Pin,
    task::{ready, Context, Poll},
};

use bytes::Bytes;
use moirai_async::io::ReadBuffer;
use proto::{Chunk, Chunks, ClosedStream, ConnectionError, ReadableError, StreamId};
use thiserror::Error;

use crate::{connection::ConnectionRef, VarInt};

/// A stream that can only be used to receive data
///
/// `stop(0)` is implicitly called on drop unless:
/// - A variant of [`ReadError`] has been yielded by a read call
/// - [`stop()`] was called explicitly
///
/// # Cancellation
///
/// A `read` method is said to be *cancel-safe* when dropping its future before the future becomes
/// ready cannot lead to loss of stream data. This is true of methods which succeed immediately when
/// any progress is made, and is not true of methods which might need to perform multiple reads
/// internally before succeeding. Each `read` method documents whether it is cancel-safe.
///
/// # Common issues
///
/// ## Data never received on a locally-opened stream
///
/// Peers are not notified of streams until they or a later-numbered stream are used to send
/// data. If a bidirectional stream is locally opened but never used to send, then the peer may
/// never see it. Application protocols should always arrange for the endpoint which will first
/// transmit on a stream to be the endpoint responsible for opening it.
///
/// ## Data never received on a remotely-opened stream
///
/// Verify that the stream you are receiving is the same one that the server is sending on, e.g. by
/// logging the [`id`] of each. Streams are always accepted in the same order as they are created,
/// i.e. ascending order by [`StreamId`]. For example, even if a sender first transmits on
/// bidirectional stream 1, the first stream yielded by [`Connection::accept_bi`] on the receiver
/// will be bidirectional stream 0.
///
/// [`ReadError`]: crate::ReadError
/// [`stop()`]: RecvStream::stop
/// [`SendStream::finish`]: crate::SendStream::finish
/// [`WriteError::Stopped`]: crate::WriteError::Stopped
/// [`id`]: RecvStream::id
/// [`Connection::accept_bi`]: crate::Connection::accept_bi
#[derive(Debug)]
pub struct RecvStream {
    conn: ConnectionRef,
    stream: StreamId,
    is_0rtt: bool,
    all_data_read: bool,
    reset: Option<VarInt>,
}

enum ReadStatus<T> {
    Readable(T),
    Finished(Option<T>),
    Failed(Option<T>, proto::ReadError),
}

impl<T> From<(Option<T>, Option<proto::ReadError>)> for ReadStatus<T> {
    fn from(status: (Option<T>, Option<proto::ReadError>)) -> Self {
        match status {
            (read, None) => Self::Finished(read),
            (read, Some(e)) => Self::Failed(read, e),
        }
    }
}

/// Future produced by [`RecvStream::read_to_end()`].
///
/// [`RecvStream::read_to_end()`]: crate::RecvStream::read_to_end
struct ReadToEnd<'a> {
    stream: &'a mut RecvStream,
    buffer: ReadToEndBuffer,
}

impl Future for ReadToEnd<'_> {
    type Output = Result<Vec<u8>, ReadToEndError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        loop {
            match ready!(self.stream.poll_read_chunk(cx, usize::MAX, false))? {
                Some(chunk) => self.buffer.push(chunk)?,
                None => return Poll::Ready(Ok(self.buffer.finish())),
            }
        }
    }
}

struct ReadToEndBuffer {
    read: Vec<(Bytes, u64)>,
    start: u64,
    end: u64,
    size_limit: usize,
}

impl ReadToEndBuffer {
    fn new(size_limit: usize) -> Self {
        Self {
            read: Vec::new(),
            start: u64::MAX,
            end: 0,
            size_limit,
        }
    }

    fn push(&mut self, chunk: Chunk) -> Result<(), ReadToEndError> {
        self.start = self.start.min(chunk.offset);
        self.end = self.end.max(chunk.bytes.len() as u64 + chunk.offset);
        if (self.end - self.start) > self.size_limit as u64 {
            return Err(ReadToEndError::TooLong);
        }
        self.read.push((chunk.bytes, chunk.offset));
        Ok(())
    }

    fn finish(&mut self) -> Vec<u8> {
        if self.end == 0 {
            return Vec::new();
        }
        let mut buffer = vec![0; (self.end - self.start) as usize];
        for (data, offset) in self.read.drain(..) {
            let offset = (offset - self.start) as usize;
            buffer[offset..offset + data.len()].copy_from_slice(&data);
        }
        buffer
    }
}

/// Errors from [`RecvStream::read_to_end`]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReadToEndError {
    /// An error occurred during reading
    #[error("read error: {0}")]
    Read(#[from] ReadError),
    /// The stream is larger than the user-supplied limit
    #[error("stream too long")]
    TooLong,
}

#[cfg(feature = "futures-io")]
impl futures_io::AsyncRead for RecvStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let mut buf = ReadBuffer::new(buf);
        ready!(Self::poll_read_buf(self.get_mut(), cx, &mut buf))?;
        Poll::Ready(Ok(buf.filled().len()))
    }
}

#[cfg(any(test, feature = "tokio-compat"))]
impl tokio::io::AsyncRead for RecvStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let count = ready!(Self::poll_read(
            self.get_mut(),
            cx,
            buf.initialize_unfilled()
        ))?;
        buf.advance(count);
        Poll::Ready(Ok(()))
    }
}

impl Drop for RecvStream {
    fn drop(&mut self) {
        if self.all_data_read {
            debug_assert!(
                !self
                    .conn
                    .state
                    .lock("RecvStream:drop")
                    .blocked_readers
                    .contains_key(&self.stream),
                "Stream {} should not have a blocked reader when all data read is true",
                self.stream
            );
            return;
        }
        let mut conn = self.conn.state.lock("RecvStream::drop");

        // clean up any previously registered wakers
        conn.blocked_readers.remove(&self.stream);

        if conn.error.is_some() || (self.is_0rtt && conn.check_0rtt().is_err()) {
            return;
        }

        // Ignore ClosedStream errors
        let _ = conn.inner.recv_stream(self.stream).stop(0u32.into());
        conn.wake();
    }
}

/// Errors that arise from reading from a stream.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReadError {
    /// The peer abandoned transmitting data on this stream
    ///
    /// Carries an application-defined error code.
    #[error("stream reset by peer: error {0}")]
    Reset(VarInt),
    /// The connection was lost
    #[error("connection lost")]
    ConnectionLost(#[from] ConnectionError),
    /// The stream has already been stopped, finished, or reset
    #[error("closed stream")]
    ClosedStream,
    /// Attempted an ordered read following an unordered read
    ///
    /// Performing an unordered read allows discontinuities to arise in the receive buffer of a
    /// stream which cannot be recovered, making further ordered reads impossible.
    #[error("ordered read after unordered read")]
    IllegalOrderedRead,
    /// This was a 0-RTT stream and the server rejected it
    ///
    /// Can only occur on clients for 0-RTT streams, which can be opened using
    /// [`Connecting::into_0rtt()`].
    ///
    /// [`Connecting::into_0rtt()`]: crate::Connecting::into_0rtt()
    #[error("0-RTT rejected")]
    ZeroRttRejected,
}

impl From<ReadableError> for ReadError {
    fn from(e: ReadableError) -> Self {
        match e {
            ReadableError::ClosedStream => Self::ClosedStream,
            ReadableError::IllegalOrderedRead => Self::IllegalOrderedRead,
        }
    }
}

impl From<ResetError> for ReadError {
    fn from(e: ResetError) -> Self {
        match e {
            ResetError::ConnectionLost(e) => Self::ConnectionLost(e),
            ResetError::ZeroRttRejected => Self::ZeroRttRejected,
        }
    }
}

impl From<ReadError> for io::Error {
    fn from(x: ReadError) -> Self {
        use ReadError::*;
        let kind = match x {
            Reset { .. } | ZeroRttRejected => io::ErrorKind::ConnectionReset,
            ConnectionLost(_) | ClosedStream => io::ErrorKind::NotConnected,
            IllegalOrderedRead => io::ErrorKind::InvalidInput,
        };
        Self::new(kind, x)
    }
}

/// Errors that arise while waiting for a stream to be reset
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ResetError {
    /// The connection was lost
    #[error("connection lost")]
    ConnectionLost(#[from] ConnectionError),
    /// This was a 0-RTT stream and the server rejected it
    ///
    /// Can only occur on clients for 0-RTT streams, which can be opened using
    /// [`Connecting::into_0rtt()`].
    ///
    /// [`Connecting::into_0rtt()`]: crate::Connecting::into_0rtt()
    #[error("0-RTT rejected")]
    ZeroRttRejected,
}

impl From<ResetError> for io::Error {
    fn from(x: ResetError) -> Self {
        use ResetError::*;
        let kind = match x {
            ZeroRttRejected => io::ErrorKind::ConnectionReset,
            ConnectionLost(_) => io::ErrorKind::NotConnected,
        };
        Self::new(kind, x)
    }
}

/// Future produced by [`RecvStream::read()`].
///
/// [`RecvStream::read()`]: crate::RecvStream::read
struct Read<'a> {
    stream: &'a mut RecvStream,
    buf: ReadBuffer<'a>,
}

impl Future for Read<'_> {
    type Output = Result<Option<usize>, ReadError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let this = self.get_mut();
        ready!(this.stream.poll_read_buf(cx, &mut this.buf))?;
        match this.buf.filled().len() {
            0 if this.buf.capacity() != 0 => Poll::Ready(Ok(None)),
            n => Poll::Ready(Ok(Some(n))),
        }
    }
}

/// Future produced by [`RecvStream::read_exact()`].
///
/// [`RecvStream::read_exact()`]: crate::RecvStream::read_exact
struct ReadExact<'a> {
    stream: &'a mut RecvStream,
    buf: ReadBuffer<'a>,
}

impl Future for ReadExact<'_> {
    type Output = Result<(), ReadExactError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut remaining = this.buf.remaining();
        while remaining > 0 {
            ready!(this.stream.poll_read_buf(cx, &mut this.buf))?;
            let new = this.buf.remaining();
            if new == remaining {
                return Poll::Ready(Err(ReadExactError::FinishedEarly(this.buf.filled().len())));
            }
            remaining = new;
        }
        Poll::Ready(Ok(()))
    }
}

/// Errors that arise from reading from a stream.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReadExactError {
    /// The stream finished before all bytes were read
    #[error("stream finished early ({0} bytes read)")]
    FinishedEarly(usize),
    /// A read error occurred
    #[error(transparent)]
    ReadError(#[from] ReadError),
}

/// Future produced by [`RecvStream::read_chunk()`].
///
/// [`RecvStream::read_chunk()`]: crate::RecvStream::read_chunk
struct ReadChunk<'a> {
    stream: &'a mut RecvStream,
    max_length: usize,
    ordered: bool,
}

impl Future for ReadChunk<'_> {
    type Output = Result<Option<Chunk>, ReadError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let (max_length, ordered) = (self.max_length, self.ordered);
        self.stream.poll_read_chunk(cx, max_length, ordered)
    }
}

/// Future produced by [`RecvStream::read_chunks()`].
///
/// [`RecvStream::read_chunks()`]: crate::RecvStream::read_chunks
struct ReadChunks<'a> {
    stream: &'a mut RecvStream,
    bufs: &'a mut [Bytes],
}

impl Future for ReadChunks<'_> {
    type Output = Result<Option<usize>, ReadError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.stream.poll_read_chunks(cx, this.bufs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_to_end_reordered_limit() {
        let mut buffer = ReadToEndBuffer::new(4);
        buffer
            .push(Chunk {
                offset: 4,
                bytes: Bytes::from_static(b"efgh"),
            })
            .unwrap();
        assert_eq!(
            buffer.push(Chunk {
                offset: 0,
                bytes: Bytes::from_static(b"abcd")
            }),
            Err(ReadToEndError::TooLong)
        );
        assert_eq!(buffer.read.len(), 1);
    }

    #[test]
    fn read_to_end_reordered_exact_limit_after_prefix() {
        let mut buffer = ReadToEndBuffer::new(8);
        // A previously consumed prefix must not count against the remaining-data limit.
        buffer
            .push(Chunk {
                offset: 104,
                bytes: Bytes::from_static(b"efgh"),
            })
            .unwrap();
        buffer
            .push(Chunk {
                offset: 100,
                bytes: Bytes::from_static(b"abcd"),
            })
            .unwrap();
        assert_eq!(buffer.finish(), b"abcdefgh");
    }

    #[test]
    fn read_to_end_gaps_and_empty() {
        assert!(ReadToEndBuffer::new(0).finish().is_empty());
        let mut buffer = ReadToEndBuffer::new(5);
        buffer
            .push(Chunk {
                offset: 14,
                bytes: Bytes::from_static(b"e"),
            })
            .unwrap();
        buffer
            .push(Chunk {
                offset: 10,
                bytes: Bytes::from_static(b"a"),
            })
            .unwrap();
        assert_eq!(buffer.finish(), b"a\0\0\0e");
        let mut buffer = ReadToEndBuffer::new(4);
        buffer
            .push(Chunk {
                offset: 10,
                bytes: Bytes::from_static(b"a"),
            })
            .unwrap();
        assert_eq!(
            buffer.push(Chunk {
                offset: 14,
                bytes: Bytes::from_static(b"e")
            }),
            Err(ReadToEndError::TooLong)
        );
    }
}

impl moirai_async::io::AsyncRead for RecvStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Self::poll_read(self.get_mut(), cx, buffer).map_err(Into::into)
    }
}

mod read;
