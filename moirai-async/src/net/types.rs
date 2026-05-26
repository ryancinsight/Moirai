//! Network configuration and statistics types.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Mutex;
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
    pub total_connections: AtomicU64,
    pub active_connections: AtomicU64,
    pub bytes_received: AtomicU64,
    pub bytes_sent: AtomicU64,
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
    active_connections: Mutex<HashMap<std::net::SocketAddr, ConnectionInfo>>,
    max_connections: Option<usize>,
}

impl ConnectionPool {
    pub fn new(max_connections: Option<usize>) -> Self {
        Self {
            active_connections: Mutex::new(HashMap::new()),
            max_connections,
        }
    }

    pub fn add_connection(&self, addr: std::net::SocketAddr) {
        let mut connections = self.active_connections.lock().unwrap();
        connections.insert(
            addr,
            ConnectionInfo {
                connected_at: Instant::now(),
                bytes_received: 0,
                bytes_sent: 0,
                last_activity: Instant::now(),
            },
        );
    }

    pub fn remove_connection(&self, addr: &std::net::SocketAddr) -> bool {
        self.active_connections
            .lock()
            .unwrap()
            .remove(addr)
            .is_some()
    }

    pub fn has_capacity(&self) -> bool {
        match self.max_connections {
            Some(max) => self.connection_count() < max,
            None => true,
        }
    }

    pub fn get_active_connections(&self) -> HashMap<std::net::SocketAddr, ConnectionInfo> {
        self.active_connections.lock().unwrap().clone()
    }

    pub fn connection_count(&self) -> usize {
        self.active_connections.lock().unwrap().len()
    }
}

/// Statistics for individual TCP connections
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_ops: u64,
    pub write_ops: u64,
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

/// Statistics for UDP socket operations
#[derive(Debug, Default)]
pub struct UdpStats {
    pub packets_sent: AtomicU64,
    pub packets_received: AtomicU64,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,
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
