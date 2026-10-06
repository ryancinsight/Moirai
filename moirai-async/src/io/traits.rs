use std::io::{self, IoSlice};
use std::pin::Pin;
use std::task::{Context, Poll};

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

    /// Attempt to write bytes from several buffers in one operation.
    ///
    /// The default writes the first non-empty buffer through
    /// [`poll_write`](Self::poll_write) and reports `Ok(0)` when every buffer
    /// is empty. A writer with a scatter-gather primitive overrides this and
    /// [`is_write_vectored`](Self::is_write_vectored) together.
    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let buf = bufs
            .iter()
            .find(|buf| !buf.is_empty())
            .map_or(&[][..], |buf| &**buf);
        self.poll_write(cx, buf)
    }

    /// Whether [`poll_write_vectored`](Self::poll_write_vectored) is cheaper
    /// than one [`poll_write`](Self::poll_write) per buffer.
    fn is_write_vectored(&self) -> bool {
        false
    }

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
