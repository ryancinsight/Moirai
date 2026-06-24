// These are used only by the `tokio-compat` trait-bridge impls below.
#[cfg(feature = "tokio-compat")]
use std::io;
#[cfg(feature = "tokio-compat")]
use std::pin::Pin;
#[cfg(feature = "tokio-compat")]
use std::task::{Context, Poll};

#[cfg(feature = "tokio-compat")]
use crate::io::traits::{AsyncRead, AsyncWrite};

#[cfg(feature = "tokio-compat")]
use tokio_dep as tokio;

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
