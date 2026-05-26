//! Async networking primitives for Moirai concurrency library.
//!
//! This module provides Moirai-owned async networking facades for TCP and UDP
//! sockets without Tokio dependencies.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::io::{AsyncRead, AsyncWrite};

#[path = "net/types.rs"]
pub mod types;

pub use types::{
    ConnectionInfo, ConnectionPool, ConnectionStats, ServerStats, TcpServerConfig, TcpServerStats,
    UdpConfig, UdpSocketStats, UdpStats,
};

use moirai_pal::net::{AsyncTcpListener, AsyncTcpStream, AsyncUdpSocket};

/// Native async TCP listener with connection management
pub struct TcpListener {
    inner: AsyncTcpListener,
    config: TcpServerConfig,
    stats: Arc<ServerStats>,
    connection_pool: Arc<ConnectionPool>,
}

impl TcpListener {
    /// Bind to an address with default configuration
    pub async fn bind(addr: &str) -> io::Result<Self> {
        Self::bind_with_config(addr, TcpServerConfig::default()).await
    }

    /// Bind to an address with custom configuration
    pub async fn bind_with_config(addr: &str, config: TcpServerConfig) -> io::Result<Self> {
        use std::net::ToSocketAddrs;
        let addr_parsed = addr.to_socket_addrs()?.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Could not resolve address")
        })?;
        let inner = AsyncTcpListener::bind(addr_parsed).await?;
        let stats = Arc::new(ServerStats::default());
        let connection_pool = Arc::new(ConnectionPool::new(config.max_connections));

        Ok(Self {
            inner,
            config,
            stats,
            connection_pool,
        })
    }

    /// Accept the next incoming connection
    pub async fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        if !self.connection_pool.has_capacity() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "Connection limit reached",
            ));
        }

        if let Some(max) = self.config.max_connections {
            let current = self
                .stats
                .active_connections
                .load(std::sync::atomic::Ordering::Relaxed);
            if current >= max as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "Connection limit reached",
                ));
            }
        }

        let (stream, addr) = self.inner.accept().await?;

        // Update statistics
        self.stats
            .total_connections
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .active_connections
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Track connection
        self.connection_pool.add_connection(addr);

        Ok((
            TcpStream::new(stream, self.stats.clone(), self.connection_pool.clone()),
            addr,
        ))
    }

    /// Get server statistics
    pub fn stats(&self) -> TcpServerStats {
        TcpServerStats {
            total_connections: self
                .stats
                .total_connections
                .load(std::sync::atomic::Ordering::Relaxed),
            active_connections: self
                .stats
                .active_connections
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_received: self
                .stats
                .bytes_received
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_sent: self
                .stats
                .bytes_sent
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Return the local socket address this listener is bound to.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

/// Native async TCP stream with statistics tracking
pub struct TcpStream {
    inner: AsyncTcpStream,
    stats: Arc<ServerStats>,
    connection_pool: Arc<ConnectionPool>,
}

impl TcpStream {
    fn new(
        inner: AsyncTcpStream,
        stats: Arc<ServerStats>,
        connection_pool: Arc<ConnectionPool>,
    ) -> Self {
        Self {
            inner,
            stats,
            connection_pool,
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

        Ok(Self::new(inner, stats, connection_pool))
    }

    /// Wrap an existing TCP stream in the Moirai TCP facade.
    pub fn from_std(stream: std::net::TcpStream) -> io::Result<Self> {
        let inner = AsyncTcpStream::from_std(stream)?;
        let stats = Arc::new(ServerStats::default());
        let connection_pool = Arc::new(ConnectionPool::new(None));
        Ok(Self::new(inner, stats, connection_pool))
    }

    /// Read data from the stream
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.inner.read(buf).await?;
        self.stats
            .bytes_received
            .fetch_add(bytes_read as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes_read)
    }

    /// Write data to the stream
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_written = self.inner.write(buf).await?;
        self.stats
            .bytes_sent
            .fetch_add(bytes_written as u64, std::sync::atomic::Ordering::Relaxed);
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
        let Ok(addr) = self.inner.peer_addr() else {
            return;
        };

        if self.connection_pool.remove_connection(&addr) {
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

/// Native async UDP socket
pub struct UdpSocket {
    inner: AsyncUdpSocket,
    stats: Arc<UdpStats>,
    config: UdpConfig,
}

impl UdpSocket {
    /// Bind UDP socket to an address
    pub async fn bind(addr: &str) -> io::Result<Self> {
        Self::bind_with_config(addr, UdpConfig::default()).await
    }

    /// Bind UDP socket with custom configuration
    pub async fn bind_with_config(addr: &str, config: UdpConfig) -> io::Result<Self> {
        use std::net::ToSocketAddrs;
        let addr_parsed = addr.to_socket_addrs()?.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Could not resolve address")
        })?;
        let inner = AsyncUdpSocket::bind(addr_parsed).await?;
        if config.broadcast {
            inner.set_broadcast(true)?;
        }

        Ok(Self {
            inner,
            stats: Arc::new(UdpStats::default()),
            config,
        })
    }

    /// Send data to a specific address
    pub async fn send_to(&self, buf: &[u8], target: SocketAddr) -> io::Result<usize> {
        let bytes_sent = self.inner.send_to(buf, target).await?;
        self.stats
            .packets_sent
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .bytes_sent
            .fetch_add(bytes_sent as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes_sent)
    }

    /// Receive data from any address
    pub async fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        let (bytes_received, addr) = self.inner.recv_from(buf).await?;
        self.stats
            .packets_received
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .bytes_received
            .fetch_add(bytes_received as u64, std::sync::atomic::Ordering::Relaxed);
        Ok((bytes_received, addr))
    }

    /// Get UDP socket statistics
    pub fn stats(&self) -> UdpSocketStats {
        UdpSocketStats {
            packets_sent: self
                .stats
                .packets_sent
                .load(std::sync::atomic::Ordering::Relaxed),
            packets_received: self
                .stats
                .packets_received
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_sent: self
                .stats
                .bytes_sent
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_received: self
                .stats
                .bytes_received
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Return the local socket address this UDP socket is bound to.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Return the socket configuration used at bind time.
    pub fn config(&self) -> &UdpConfig {
        &self.config
    }
}

#[cfg(test)]
#[path = "net/tests.rs"]
mod tests;
