//! Async I/O traits and compatibility utilities for Moirai.
//!
//! This module defines the core asynchronous I/O abstractions matching
//! zero-copy buffer ownership, zero-cost extension futures, and
//! monomorphization goals. Feature-gated Tokio compatibility wrappers are
//! excluded from the default build so the core runtime has no Tokio dependency.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

#[cfg(feature = "tokio-compat")]
use tokio_dep as tokio;

/// Read bytes asynchronously.
pub trait AsyncRead {
    /// Attempt to read from the source into the provided buffer.
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>>;
}

/// Write bytes asynchronously.
pub trait AsyncWrite {
    /// Attempt to write bytes from the buffer to the destination.
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>>;

    /// Attempt to flush pending writes.
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>>;

    /// Attempt to close/shutdown the write side of the stream.
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>>;
}

/// Read bytes from a buffered source asynchronously.
pub trait AsyncBufRead: AsyncRead {
    /// Returns the contents of the internal buffer, filling it with more data
    /// from the inner reader if necessary.
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<&[u8]>>;

    /// Consumes the specified amount of bytes from the internal buffer.
    fn consume(self: Pin<&mut Self>, amt: usize);
}

/// Extension methods for types implementing [`AsyncRead`].
pub trait AsyncReadExt: AsyncRead {
    /// Read some bytes asynchronously from the reader.
    fn read<'a>(&'a mut self, buf: &'a mut [u8]) -> Read<'a, Self>
    where
        Self: Unpin,
    {
        Read { reader: self, buf }
    }

    /// Read exactly enough bytes to fill the provided buffer.
    ///
    /// The future borrows the caller-provided buffer directly and stores only a
    /// byte offset, so partial progress remains visible if the future is
    /// cancelled before completion.
    fn read_exact<'a>(&'a mut self, buf: &'a mut [u8]) -> ReadExact<'a, Self>
    where
        Self: Unpin,
    {
        ReadExact {
            reader: self,
            buf,
            filled: 0,
        }
    }
}

impl<R: AsyncRead + ?Sized> AsyncReadExt for R {}

/// Future returned by [`AsyncReadExt::read`].
pub struct Read<'a, R: ?Sized> {
    reader: &'a mut R,
    buf: &'a mut [u8],
}

impl<'a, R: AsyncRead + ?Sized + Unpin> Future for Read<'a, R> {
    type Output = io::Result<usize>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        Pin::new(&mut *this.reader).poll_read(cx, this.buf)
    }
}

/// Future returned by [`AsyncReadExt::read_exact`].
pub struct ReadExact<'a, R: ?Sized> {
    reader: &'a mut R,
    buf: &'a mut [u8],
    filled: usize,
}

impl<'a, R: AsyncRead + ?Sized + Unpin> Future for ReadExact<'a, R> {
    type Output = io::Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        while this.filled < this.buf.len() {
            let filled = this.filled;
            match Pin::new(&mut *this.reader).poll_read(cx, &mut this.buf[filled..]) {
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    )));
                }
                Poll::Ready(Ok(n)) => {
                    this.filled += n;
                }
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(Ok(()))
    }
}

/// Extension methods for types implementing [`AsyncWrite`].
pub trait AsyncWriteExt: AsyncWrite {
    /// Write some bytes asynchronously.
    fn write<'a>(&'a mut self, buf: &'a [u8]) -> Write<'a, Self>
    where
        Self: Unpin,
    {
        Write { writer: self, buf }
    }

    /// Write all bytes asynchronously.
    fn write_all<'a>(&'a mut self, buf: &'a [u8]) -> WriteAll<'a, Self>
    where
        Self: Unpin,
    {
        WriteAll {
            writer: self,
            buf,
            written: 0,
        }
    }

    /// Flush pending writes asynchronously.
    fn flush(&mut self) -> Flush<'_, Self>
    where
        Self: Unpin,
    {
        Flush { writer: self }
    }

    /// Shutdown the write side of the stream.
    fn shutdown(&mut self) -> Shutdown<'_, Self>
    where
        Self: Unpin,
    {
        Shutdown { writer: self }
    }
}

impl<W: AsyncWrite + ?Sized> AsyncWriteExt for W {}

/// Future returned by [`AsyncWriteExt::write`].
pub struct Write<'a, W: ?Sized> {
    writer: &'a mut W,
    buf: &'a [u8],
}

impl<'a, W: AsyncWrite + ?Sized + Unpin> Future for Write<'a, W> {
    type Output = io::Result<usize>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        Pin::new(&mut *this.writer).poll_write(cx, this.buf)
    }
}

/// Future returned by [`AsyncWriteExt::write_all`].
pub struct WriteAll<'a, W: ?Sized> {
    writer: &'a mut W,
    buf: &'a [u8],
    written: usize,
}

impl<'a, W: AsyncWrite + ?Sized + Unpin> Future for WriteAll<'a, W> {
    type Output = io::Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        while this.written < this.buf.len() {
            match Pin::new(&mut *this.writer).poll_write(cx, &this.buf[this.written..]) {
                Poll::Ready(Ok(0)) => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    )));
                }
                Poll::Ready(Ok(n)) => {
                    this.written += n;
                }
                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(Ok(()))
    }
}

/// Future returned by [`AsyncWriteExt::flush`].
pub struct Flush<'a, W: ?Sized> {
    writer: &'a mut W,
}

impl<'a, W: AsyncWrite + ?Sized + Unpin> Future for Flush<'a, W> {
    type Output = io::Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        Pin::new(&mut *this.writer).poll_flush(cx)
    }
}

/// Future returned by [`AsyncWriteExt::shutdown`].
pub struct Shutdown<'a, W: ?Sized> {
    writer: &'a mut W,
}

impl<'a, W: AsyncWrite + ?Sized + Unpin> Future for Shutdown<'a, W> {
    type Output = io::Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        Pin::new(&mut *this.writer).poll_shutdown(cx)
    }
}

/// Wrapper providing Tokio's I/O traits compatibility.
#[repr(transparent)]
pub struct TokioCompat<T> {
    inner: T,
}

impl<T> TokioCompat<T> {
    /// Create a new Tokio compatibility wrapper.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Extract the inner type.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> From<T> for TokioCompat<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

/// Wrapper providing Moirai's native I/O traits compatibility for Tokio types.
#[repr(transparent)]
pub struct MoiraiCompat<T> {
    inner: T,
}

impl<T> MoiraiCompat<T> {
    /// Create a new Moirai compatibility wrapper.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Extract the inner type.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> From<T> for MoiraiCompat<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

#[cfg(feature = "tokio-compat")]
impl<T: AsyncRead + Unpin> tokio::io::AsyncRead for TokioCompat<T> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let unfilled = buf.initialize_unfilled();
        match Pin::new(&mut self.inner).poll_read(cx, unfilled) {
            Poll::Ready(Ok(n)) => {
                buf.advance(n);
                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(feature = "tokio-compat")]
impl<T: AsyncWrite + Unpin> tokio::io::AsyncWrite for TokioCompat<T> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

#[cfg(feature = "tokio-compat")]
impl<T: tokio::io::AsyncRead + Unpin> AsyncRead for MoiraiCompat<T> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let mut read_buf = tokio::io::ReadBuf::new(buf);
        match Pin::new(&mut self.inner).poll_read(cx, &mut read_buf) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(read_buf.filled().len())),
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(feature = "tokio-compat")]
impl<T: tokio::io::AsyncWrite + Unpin> AsyncWrite for MoiraiCompat<T> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
#[path = "io/tests.rs"]
mod tests;
