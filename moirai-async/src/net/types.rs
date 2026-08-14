//! Network configuration and statistics types.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Configuration for TCP server behavior
#[derive(Debug, Clone)]
pub struct TcpServerConfig {
    /// Maximum number of concurrent connections
    pub max_connections: Option<usize>,
    /// Socket TCP_NODELAY setting, applied to every accepted stream.
    pub nodelay: bool,
    /// TCP keep-alive duration.
    ///
    /// Currently not applied: neither `std::net::TcpStream` nor
    /// `moirai_pal::net::AsyncTcpStream` exposes a keep-alive setter (it
    /// requires `SO_KEEPALIVE`/`TCP_KEEPIDLE` support in the PAL). Pending PAL
    /// wiring; until then the value is configuration-only.
    pub keep_alive: Option<Duration>,
    /// Connection timeout.
    ///
    /// Currently not applied: accepted sockets are non-blocking (reactor
    /// driven), where `SO_RCVTIMEO`/`SO_SNDTIMEO` have no effect. Async
    /// deadlines are expressed by wrapping operations in
    /// [`crate::timer::timeout()`]; automatic application of this value is
    /// pending.
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
    /// Connections accepted over the server's lifetime.
    pub total_connections: AtomicU64,
    /// Connections currently tracked as open.
    pub active_connections: AtomicU64,
    /// Bytes received across all connections.
    pub bytes_received: AtomicU64,
    /// Bytes sent across all connections.
    pub bytes_sent: AtomicU64,
}

/// Unique, monotonically-assigned identifier for a tracked connection.
///
/// Connections are keyed by id rather than by peer [`std::net::SocketAddr`] so that
/// (a) a stream can be removed from the pool at drop time without re-querying
/// the (possibly already-reset) socket, and (b) two connections sharing a peer
/// address (NAT, rapid address reuse) cannot collide in the tracking map.
pub type ConnectionId = u64;

/// Connection information tracking
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Peer address captured at accept/connect time (never re-queried).
    pub peer_addr: std::net::SocketAddr,
    /// Instant the connection entered the pool.
    pub connected_at: Instant,
    /// Bytes received on this connection.
    pub bytes_received: u64,
    /// Bytes sent on this connection.
    pub bytes_sent: u64,
    /// Instant of the most recent recorded activity.
    pub last_activity: Instant,
}

/// Connection pool for managing active connections
#[derive(Debug)]
pub struct ConnectionPool {
    active_connections: Mutex<HashMap<ConnectionId, ConnectionInfo>>,
    reserved_connections: AtomicUsize,
    next_connection_id: AtomicU64,
    max_connections: Option<usize>,
}

impl ConnectionPool {
    /// Create a pool bounded to `max_connections`, or unbounded on `None`.
    #[must_use]
    pub fn new(max_connections: Option<usize>) -> Self {
        Self {
            active_connections: Mutex::new(HashMap::new()),
            reserved_connections: AtomicUsize::new(0),
            next_connection_id: AtomicU64::new(0),
            max_connections,
        }
    }

    /// Reserve one connection slot ahead of an accept.
    ///
    /// Returns false when the pool (active plus reserved) is at capacity.
    pub fn try_reserve(&self) -> bool {
        let max = match self.max_connections {
            Some(m) => m,
            None => return true,
        };

        let connections = self.active_connections.lock().unwrap();
        let current = connections.len();
        // The mutex orders all admission increments. Releases may race this
        // snapshot, but they only reduce the reservation count; a stale release
        // can conservatively reject an admission, never over-admit one. The
        // counter carries no payload, so Relaxed is sufficient for its atomic
        // accounting.
        let reserved = self.reserved_connections.load(Ordering::Relaxed);
        if current + reserved < max {
            self.reserved_connections.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Release a slot taken by [`Self::try_reserve`] without admitting a
    /// connection.
    pub fn cancel_reservation(&self) {
        if self.max_connections.is_some() {
            self.reserved_connections.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Register a connection and return its unique id. The id is what the owning
    /// stream stores and later passes to [`Self::remove_connection`].
    pub fn add_connection(&self, addr: std::net::SocketAddr) -> ConnectionId {
        let id = self.next_connection_id.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();
        self.active_connections.lock().unwrap().insert(
            id,
            ConnectionInfo {
                peer_addr: addr,
                connected_at: now,
                bytes_received: 0,
                bytes_sent: 0,
                last_activity: now,
            },
        );
        id
    }

    /// Convert a successful reservation into a tracked connection, returning the
    /// new connection id. Releases exactly the one reservation taken by
    /// [`Self::try_reserve`].
    pub fn add_connection_reserved(&self, addr: std::net::SocketAddr) -> ConnectionId {
        let id = self.add_connection(addr);
        if self.max_connections.is_some() {
            self.reserved_connections.fetch_sub(1, Ordering::Relaxed);
        }
        id
    }

    /// Record I/O activity on a tracked connection, updating its byte counters
    /// and `last_activity` timestamp. No-op when the connection is no longer
    /// tracked (already removed or never pool-tracked).
    pub fn record_io(&self, id: ConnectionId, bytes_received: u64, bytes_sent: u64) {
        let mut connections = self.active_connections.lock().unwrap();
        if let Some(info) = connections.get_mut(&id) {
            info.bytes_received += bytes_received;
            info.bytes_sent += bytes_sent;
            info.last_activity = Instant::now();
        }
    }

    /// Remove a tracked connection; returns whether it was present.
    pub fn remove_connection(&self, id: ConnectionId) -> bool {
        self.active_connections
            .lock()
            .unwrap()
            .remove(&id)
            .is_some()
    }

    /// Return whether the pool can admit another connection.
    pub fn has_capacity(&self) -> bool {
        match self.max_connections {
            Some(max) => {
                let current = self.connection_count();
                let reserved = self.reserved_connections.load(Ordering::Relaxed);
                current + reserved < max
            }
            None => true,
        }
    }

    /// Snapshot the tracked connections.
    pub fn get_active_connections(&self) -> HashMap<ConnectionId, ConnectionInfo> {
        self.active_connections.lock().unwrap().clone()
    }

    /// Count of connections currently tracked as open.
    pub fn connection_count(&self) -> usize {
        self.active_connections.lock().unwrap().len()
    }
}

/// Statistics for individual TCP connections
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    /// Bytes read on the connection.
    pub bytes_read: u64,
    /// Bytes written on the connection.
    pub bytes_written: u64,
    /// Completed read operations.
    pub read_ops: u64,
    /// Completed write operations.
    pub write_ops: u64,
}

/// Configuration for UDP socket behavior
#[derive(Debug, Clone)]
pub struct UdpConfig {
    /// Socket buffer size.
    ///
    /// Currently not applied: `std::net::UdpSocket` exposes no
    /// `SO_RCVBUF`/`SO_SNDBUF` setter (requires PAL/socket2-level support).
    /// Pending PAL wiring; until then the value is configuration-only.
    pub buffer_size: usize,
    /// Broadcast support, applied at bind time via `SO_BROADCAST`.
    pub broadcast: bool,
    /// Multicast support.
    ///
    /// Currently not applied: joining a multicast group requires a group
    /// address, which this flag cannot carry, and the PAL exposes no
    /// `join_multicast` surface. Pending a typed multicast configuration
    /// (group + interface) in place of this flag.
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
    /// Datagrams sent.
    pub packets_sent: AtomicU64,
    /// Datagrams received.
    pub packets_received: AtomicU64,
    /// Bytes sent.
    pub bytes_sent: AtomicU64,
    /// Bytes received.
    pub bytes_received: AtomicU64,
}

/// Public TCP server statistics
#[derive(Debug, Clone)]
pub struct TcpServerStats {
    /// Connections accepted over the server's lifetime.
    pub total_connections: u64,
    /// Connections currently open.
    pub active_connections: u64,
    /// Bytes received across all connections.
    pub bytes_received: u64,
    /// Bytes sent across all connections.
    pub bytes_sent: u64,
}

/// Public UDP socket statistics  
#[derive(Debug, Clone)]
pub struct UdpSocketStats {
    /// Datagrams sent.
    pub packets_sent: u64,
    /// Datagrams received.
    pub packets_received: u64,
    /// Bytes sent.
    pub bytes_sent: u64,
    /// Bytes received.
    pub bytes_received: u64,
}
