//! Async networking primitives for Moirai concurrency library.
//!
//! This module provides native async networking support including TCP and UDP sockets,
//! without tokio dependencies, following SLAP principle with focused responsibility 
//! on network I/O operations.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream, UdpSocket as StdUdpSocket, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for TCP server behavior
#[derive(Debug, Clone)]
pub struct TcpServerConfig {
    /// Maximum number of concurrent connections
    pub max_connections: Option<usize>,
    /// Socket TCP_NODELAY setting
    pub nodelay: bool,
    /// TCP keep-alive duration
    pub keep_alive: Option<Duration>,
    /// Connection timeout
    pub timeout: Option<Duration>,
}

impl Default for TcpServerConfig {
    fn default() -> Self {
        Self {
            max_connections: Some(1000),
            nodelay: true,
            keep_alive: Some(Duration::from_secs(300)),
            timeout: Some(Duration::from_secs(30)),
        }
    }
}

/// TCP server statistics for monitoring
#[derive(Debug, Default)]
pub struct ServerStats {
    total_connections: std::sync::atomic::AtomicU64,
    active_connections: std::sync::atomic::AtomicU64,
    bytes_received: std::sync::atomic::AtomicU64,
    bytes_sent: std::sync::atomic::AtomicU64,
}

/// Connection information tracking
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub connected_at: Instant,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub last_activity: Instant,
}

/// Connection pool for managing active connections
#[derive(Debug)]
#[allow(dead_code)] // Fields used for future connection management per ADR requirements
pub struct ConnectionPool {
    active_connections: Mutex<HashMap<SocketAddr, ConnectionInfo>>,
    max_connections: Option<usize>,
}

impl ConnectionPool {
    pub fn new(max_connections: Option<usize>) -> Self {
        Self {
            active_connections: Mutex::new(HashMap::new()),
            max_connections,
        }
    }

    pub fn add_connection(&self, addr: SocketAddr) {
        let mut connections = self.active_connections.lock().unwrap();
        connections.insert(addr, ConnectionInfo {
            connected_at: Instant::now(),
            bytes_received: 0,
            bytes_sent: 0,
            last_activity: Instant::now(),
        });
    }

    pub fn remove_connection(&self, addr: &SocketAddr) {
        self.active_connections.lock().unwrap().remove(addr);
    }

    pub fn get_active_connections(&self) -> HashMap<SocketAddr, ConnectionInfo> {
        self.active_connections.lock().unwrap().clone()
    }

    pub fn connection_count(&self) -> usize {
        self.active_connections.lock().unwrap().len()
    }
}

/// Native async TCP listener with connection management
pub struct TcpListener {
    inner: StdTcpListener,
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
        let inner = StdTcpListener::bind(addr)?;
        inner.set_nonblocking(true)?;
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
    /// 
    /// Note: This is a simplified implementation that wraps blocking I/O.
    /// A full implementation would use proper async I/O with epoll/kqueue/iocp.
    pub async fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        // Check connection limit
        if let Some(max) = self.config.max_connections {
            let current = self.stats.active_connections.load(std::sync::atomic::Ordering::Relaxed);
            if current >= max as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "Connection limit reached",
                ));
            }
        }

        // For now, use blocking accept wrapped in async
        // TODO: Implement proper async I/O reactor
        let (stream, addr) = self.inner.accept()?;
        
        // Configure the accepted socket
        stream.set_nodelay(self.config.nodelay)?;
        
        if let Some(keep_alive) = self.config.keep_alive {
            let socket = socket2::Socket::from(stream);
            let keep_alive_config = socket2::TcpKeepalive::new()
                .with_time(keep_alive)
                .with_interval(Duration::from_secs(60));
            socket.set_tcp_keepalive(&keep_alive_config)?;
            let stream: StdTcpStream = socket.into();
            
            // Update statistics
            self.stats.total_connections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.stats.active_connections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Track connection
            self.connection_pool.add_connection(addr);

            return Ok((
                TcpStream::new(stream, self.stats.clone(), self.connection_pool.clone()),
                addr,
            ));
        }

        // Update statistics
        self.stats.total_connections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.active_connections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

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
            total_connections: self.stats.total_connections.load(std::sync::atomic::Ordering::Relaxed),
            active_connections: self.stats.active_connections.load(std::sync::atomic::Ordering::Relaxed),
            bytes_received: self.stats.bytes_received.load(std::sync::atomic::Ordering::Relaxed),
            bytes_sent: self.stats.bytes_sent.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// Native async TCP stream with statistics tracking
#[allow(dead_code)] // Fields used for future connection tracking per ADR requirements
pub struct TcpStream {
    inner: StdTcpStream,
    stats: Arc<ServerStats>,
    connection_pool: Arc<ConnectionPool>,
}

/// Statistics for individual TCP connections
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_ops: u64,
    pub write_ops: u64,
}

impl TcpStream {
    fn new(
        inner: StdTcpStream,
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
        let stream = StdTcpStream::connect(addr)?;
        let _peer_addr = stream.peer_addr()?;
        
        // Configure socket for optimal performance
        stream.set_nodelay(true)?;

        let stats = Arc::new(ServerStats::default());
        let connection_pool = Arc::new(ConnectionPool::new(None));

        Ok(Self::new(stream, stats, connection_pool))
    }

    /// Read data from the stream
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let bytes_read = self.inner.read(buf)?;
        self.stats.bytes_received.fetch_add(bytes_read as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes_read)
    }

    /// Write data to the stream
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let bytes_written = self.inner.write(buf)?;
        self.stats.bytes_sent.fetch_add(bytes_written as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes_written)
    }

    /// Flush the stream
    pub async fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    /// Get the peer address
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    /// Get the local address
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

/// Configuration for UDP socket behavior
#[derive(Debug, Clone)]
pub struct UdpConfig {
    /// Socket buffer size
    pub buffer_size: usize,
    /// Broadcast support
    pub broadcast: bool,
    /// Multicast support
    pub multicast: bool,
}

impl Default for UdpConfig {
    fn default() -> Self {
        Self {
            buffer_size: 65536,
            broadcast: false,
            multicast: false,
        }
    }
}

/// Native async UDP socket
#[allow(dead_code)] // Fields used for future UDP configuration per ADR requirements
pub struct UdpSocket {
    inner: StdUdpSocket,
    stats: Arc<UdpStats>,
    config: UdpConfig,
}

/// Statistics for UDP socket operations
#[derive(Debug, Default)]
pub struct UdpStats {
    packets_sent: std::sync::atomic::AtomicU64,
    packets_received: std::sync::atomic::AtomicU64,
    bytes_sent: std::sync::atomic::AtomicU64,
    bytes_received: std::sync::atomic::AtomicU64,
}

impl UdpSocket {
    /// Bind UDP socket to an address
    pub async fn bind(addr: &str) -> io::Result<Self> {
        Self::bind_with_config(addr, UdpConfig::default()).await
    }

    /// Bind UDP socket with custom configuration
    pub async fn bind_with_config(addr: &str, config: UdpConfig) -> io::Result<Self> {
        let inner = StdUdpSocket::bind(addr)?;
        inner.set_nonblocking(true)?;
        
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
        let bytes_sent = self.inner.send_to(buf, target)?;
        self.stats.packets_sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.bytes_sent.fetch_add(bytes_sent as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes_sent)
    }

    /// Receive data from any address
    pub async fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        let (bytes_received, addr) = self.inner.recv_from(buf)?;
        self.stats.packets_received.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.bytes_received.fetch_add(bytes_received as u64, std::sync::atomic::Ordering::Relaxed);
        Ok((bytes_received, addr))
    }

    /// Get UDP socket statistics
    pub fn stats(&self) -> UdpSocketStats {
        UdpSocketStats {
            packets_sent: self.stats.packets_sent.load(std::sync::atomic::Ordering::Relaxed),
            packets_received: self.stats.packets_received.load(std::sync::atomic::Ordering::Relaxed),
            bytes_sent: self.stats.bytes_sent.load(std::sync::atomic::Ordering::Relaxed),
            bytes_received: self.stats.bytes_received.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// Public TCP server statistics
#[derive(Debug, Clone)]
pub struct TcpServerStats {
    pub total_connections: u64,
    pub active_connections: u64,
    pub bytes_received: u64,
    pub bytes_sent: u64,
}

/// Public UDP socket statistics  
#[derive(Debug, Clone)]
pub struct UdpSocketStats {
    pub packets_sent: u64,
    pub packets_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_server_config() {
        let config = TcpServerConfig::default();
        assert_eq!(config.max_connections, Some(1000));
        assert!(config.nodelay);
        assert_eq!(config.keep_alive, Some(Duration::from_secs(300)));
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_udp_config() {
        let config = UdpConfig::default();
        assert_eq!(config.buffer_size, 65536);
        assert!(!config.broadcast);
        assert!(!config.multicast);
    }

    #[test]
    fn test_server_stats() {
        let stats = ServerStats::default();
        assert_eq!(stats.total_connections.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(stats.active_connections.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(stats.bytes_received.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(stats.bytes_sent.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn test_udp_stats() {
        let stats = UdpStats::default();
        assert_eq!(stats.packets_sent.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(stats.packets_received.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(stats.bytes_sent.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(stats.bytes_received.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    // TODO: Add proper async network tests once Moirai's async runtime is integrated
}