use moirai_pal::net::AsyncTcpStream;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::io::{AsyncRead, AsyncWrite};
use crate::net::types::{ConnectionId, ConnectionPool, ServerStats};

/// Native async TCP stream with statistics tracking
pub struct TcpStream {
    inner: AsyncTcpStream,
    stats: Arc<ServerStats>,
    connection_pool: Arc<ConnectionPool>,
    /// Pool tracking id assigned at accept time, or `None` for client-side
    /// streams (`connect`/`from_std`) that are not pool-tracked. `Drop` removes
    /// by this id rather than re-querying the socket, so a connection is
    /// untracked exactly once even if the peer has already reset.
    connection_id: Option<ConnectionId>,
}

impl TcpStream {
    pub(super) fn new(
        inner: AsyncTcpStream,
        stats: Arc<ServerStats>,
        connection_pool: Arc<ConnectionPool>,
        connection_id: Option<ConnectionId>,
    ) -> Self {
        Self {
            inner,
            stats,
            connection_pool,
            connection_id,
        }
    }

    /// Connect to a remote address asynchronously
    pub async fn connect(addr: &str) -> io::Result<Self> {
        use std::net::ToSocketAddrs;
        let addr_parsed = addr.to_socket_addrs()?.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Could not resolve address")
        })?;
        let inner = AsyncTcpStream::connect(addr_parsed).await?;
        let stats = Arc::new(ServerStats::default());
        let connection_pool = Arc::new(ConnectionPool::new(None));

        Ok(Self::new(inner, stats, connection_pool, None))
    }

    /// Wrap an existing TCP stream in the Moirai TCP facade.
    pub fn from_std(stream: std::net::TcpStream) -> io::Result<Self> {
        let inner = AsyncTcpStream::from_std(stream)?;
        let stats = Arc::new(ServerStats::default());
        let connection_pool = Arc::new(ConnectionPool::new(None));
        Ok(Self::new(inner, stats, connection_pool, None))
    }

    /// Update per-connection pool tracking (byte counters, `last_activity`)
    /// for pool-tracked (accept-side) streams; no-op for client-side streams.
    fn record_io(&self, bytes_received: u64, bytes_sent: u64) {
        if let Some(id) = self.connection_id {
            self.connection_pool
                .record_io(id, bytes_received, bytes_sent);
        }
    }

    /// Read data from the stream
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.inner.read(buf).await?;
        self.stats
            .bytes_received
            .fetch_add(bytes_read as u64, std::sync::atomic::Ordering::Relaxed);
        self.record_io(bytes_read as u64, 0);
        Ok(bytes_read)
    }

    /// Write data to the stream
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_written = self.inner.write(buf).await?;
        self.stats
            .bytes_sent
            .fetch_add(bytes_written as u64, std::sync::atomic::Ordering::Relaxed);
        self.record_io(0, bytes_written as u64);
        Ok(bytes_written)
    }

    /// Flush the stream
    pub async fn flush(&mut self) -> io::Result<()> {
        self.inner.flush().await
    }

    /// Shutdown the write side of the stream.
    pub async fn shutdown(&mut self) -> io::Result<()> {
        self.inner.shutdown_write()
    }

    /// Get the peer address
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    /// Get the local address
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Configure TCP_NODELAY on the stream.
    pub fn set_nodelay(&self, on: bool) -> io::Result<()> {
        self.inner.set_nodelay(on)
    }
}

impl Drop for TcpStream {
    fn drop(&mut self) {
        // Untrack by the id captured at accept time. Re-querying `peer_addr()`
        // here would fail on an already-reset socket and leak the pool slot plus
        // the `active_connections` counter for the process lifetime.
        let Some(id) = self.connection_id else {
            return;
        };

        if self.connection_pool.remove_connection(id) {
            self.stats
                .active_connections
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl AsyncRead for TcpStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        match self.inner.poll_read(cx, buf) {
            Poll::Ready(Ok(n)) => {
                self.stats
                    .bytes_received
                    .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                self.record_io(n as u64, 0);
                Poll::Ready(Ok(n))
            }
            res => res,
        }
    }
}

impl AsyncWrite for TcpStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.inner.poll_write(cx, buf) {
            Poll::Ready(Ok(n)) => {
                self.stats
                    .bytes_sent
                    .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
                self.record_io(0, n as u64);
                Poll::Ready(Ok(n))
            }
            res => res,
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.inner.poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.inner.shutdown_write())
    }
}
