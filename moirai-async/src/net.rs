//! Async networking primitives for Moirai concurrency library.
//!
//! This module provides async networking support including TCP and UDP sockets,
//! following SLAP principle with focused responsibility on network I/O operations.

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::{TcpListener as TokioTcpListener, TcpStream as TokioTcpStream, UdpSocket as TokioUdpSocket};

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

/// High-performance async TCP listener with connection management
pub struct TcpListener {
    inner: TokioTcpListener,
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
        let inner = TokioTcpListener::bind(addr).await?;
        let stats = Arc::new(ServerStats::default());
        let connection_pool = Arc::new(ConnectionPool::new(config.max_connections));

        Ok(Self {
            inner,
            config,
            stats,
            connection_pool,
        })
    }

    /// Accept the next incoming connection with comprehensive error handling
    ///
    /// # Behavior Guarantees
    /// - Returns when a connection is available
    /// - Respects connection limits if set
    /// - Properly handles network errors
    /// - Updates connection tracking and statistics
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

        let (mut stream, addr) = self.inner.accept().await?;
        
        // Configure the accepted socket
        stream.set_nodelay(self.config.nodelay)?;
        
        if let Some(keep_alive) = self.config.keep_alive {
            // Convert tokio TcpStream to std TcpStream for socket2 compatibility
            let std_stream = stream.into_std()?;
            let socket = socket2::Socket::from(std_stream);
            let keep_alive = socket2::TcpKeepalive::new()
                .with_time(keep_alive)
                .with_interval(Duration::from_secs(60));
            socket.set_tcp_keepalive(&keep_alive)?;
            // Convert back to tokio TcpStream
            stream = TokioTcpStream::from_std(socket.into())?;
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

/// High-performance async TCP stream with statistics tracking
pub struct TcpStream {
    inner: TokioTcpStream,
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
        inner: TokioTcpStream,
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
        let stream = TokioTcpStream::connect(addr).await?;
        let _peer_addr = stream.peer_addr()?;
        
        // Configure socket for optimal performance
        stream.set_nodelay(true)?;

        let stats = Arc::new(ServerStats::default());
        let connection_pool = Arc::new(ConnectionPool::new(None));

        Ok(Self::new(stream, stats, connection_pool))
    }

    /// Read data from the stream
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        use tokio::io::AsyncReadExt;
        let bytes_read = self.inner.read(buf).await?;
        self.stats.bytes_received.fetch_add(bytes_read as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes_read)
    }

    /// Write data to the stream
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        use tokio::io::AsyncWriteExt;
        let bytes_written = self.inner.write(buf).await?;
        self.stats.bytes_sent.fetch_add(bytes_written as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes_written)
    }

    /// Flush the stream
    pub async fn flush(&mut self) -> io::Result<()> {
        use tokio::io::AsyncWriteExt;
        self.inner.flush().await
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

/// High-performance async UDP socket
pub struct UdpSocket {
    inner: TokioUdpSocket,
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
        let inner = TokioUdpSocket::bind(addr).await?;
        
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
        self.stats.packets_sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.bytes_sent.fetch_add(bytes_sent as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(bytes_sent)
    }

    /// Receive data from any address
    pub async fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        let (bytes_received, addr) = self.inner.recv_from(buf).await?;
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
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_tcp_server_basic() {
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = server.inner.local_addr().unwrap();

        // Spawn a client connection
        let client_handle = tokio::spawn(async move {
            TcpStream::connect(&addr.to_string()).await.unwrap()
        });

        // Accept the connection
        let (stream, _addr) = server.accept().await.unwrap();
        let _client_stream = client_handle.await.unwrap();

        let stats = server.stats();
        assert_eq!(stats.total_connections, 1);
        assert_eq!(stats.active_connections, 1);
    }

    #[tokio::test]
    async fn test_udp_socket_basic() {
        let socket1 = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let socket2 = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        
        let addr1 = socket1.inner.local_addr().unwrap();
        let addr2 = socket2.inner.local_addr().unwrap();

        // Send data from socket1 to socket2
        let sent = socket1.send_to(b"hello", addr2).await.unwrap();
        assert_eq!(sent, 5);

        // Receive data on socket2
        let mut buf = [0u8; 1024];
        let (received, from_addr) = socket2.recv_from(&mut buf).await.unwrap();
        assert_eq!(received, 5);
        assert_eq!(from_addr, addr1);
        assert_eq!(&buf[..received], b"hello");

        let stats1 = socket1.stats();
        let stats2 = socket2.stats();
        assert_eq!(stats1.packets_sent, 1);
        assert_eq!(stats2.packets_received, 1);
    }

    #[tokio::test]
    async fn test_connection_limits() {
        let config = TcpServerConfig {
            max_connections: Some(1),
            ..Default::default()
        };
        let server = TcpListener::bind_with_config("127.0.0.1:0", config).await.unwrap();
        let addr = server.inner.local_addr().unwrap();

        // First connection should succeed
        let _client1 = TcpStream::connect(&addr.to_string()).await.unwrap();
        let (_stream1, _addr1) = server.accept().await.unwrap();

        // Second connection should be rejected due to limit
        let _client2 = TcpStream::connect(&addr.to_string()).await.unwrap();
        let result = timeout(Duration::from_millis(100), server.accept()).await;
        
        // Should timeout or return would block error
        assert!(result.is_err() || result.unwrap().is_err());
    }
}