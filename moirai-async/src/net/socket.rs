use moirai_pal::net::AsyncUdpSocket;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::net::types::{UdpConfig, UdpSocketStats, UdpStats};

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
