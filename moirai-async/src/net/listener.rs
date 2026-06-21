use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use moirai_pal::net::AsyncTcpListener;

use crate::net::types::{TcpServerConfig, ServerStats, ConnectionPool, TcpServerStats};
use crate::net::stream::TcpStream;

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
