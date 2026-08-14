#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use crate::{router::RemoteAddress, NetworkTransport, Transport, TransportError, TransportResult};
use moirai_core::channel::{mpmc, MpmcReceiver, MpmcSender};
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// Default MPMC channel capacity
const DEFAULT_MPMC_CAPACITY: usize = 1024;

/// Poison-recovering read lock.
fn read_rwlock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// Poison-recovering write lock.
fn write_rwlock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// Address for identifying communication endpoints
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Address {
    /// Local in-process address
    Local(String),
    /// Remote network address
    Remote(RemoteAddress),
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Address::Local(id) => write!(f, "local://{}", id),
            Address::Remote(addr) => write!(f, "{}", addr),
        }
    }
}

/// A local in-memory channel: the sender/receiver pair for one `Address::Local` id.
type LocalChannel = (MpmcSender<Vec<u8>>, MpmcReceiver<Vec<u8>>);

/// In-memory transport for local communication
pub struct InMemoryTransport {
    /// One `RwLock`-guarded map of `id -> (sender, receiver)`. Steady-state
    /// `send`/`recv` resolve an existing channel under a *concurrent read* lock
    /// and clone the cloned handle (the MPMC channel itself is lock-free), so
    /// they no longer serialize through a global mutex per message. The write
    /// lock is taken only to create a new channel.
    channels: Arc<RwLock<HashMap<String, LocalChannel>>>,
}

impl InMemoryTransport {
    /// Create an empty in-memory transport with no registered channels.
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn get_or_create_channel(&self, id: &str) -> LocalChannel {
        // Fast path: an existing channel — the steady-state case after the first
        // message — is resolved under a concurrent read lock.
        if let Some(pair) = read_rwlock(&self.channels).get(id) {
            return pair.clone();
        }

        // Slow path: create under the write lock, re-checking in case another
        // thread created the same id while we waited for the lock.
        let mut channels = write_rwlock(&self.channels);
        if let Some(pair) = channels.get(id) {
            return pair.clone();
        }
        let pair = mpmc(DEFAULT_MPMC_CAPACITY);
        channels.insert(id.to_string(), pair.clone());
        pair
    }
}

impl Transport for InMemoryTransport {
    fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()> {
        match target {
            Address::Local(id) => {
                let (tx, _) = self.get_or_create_channel(id);
                tx.send(data)
            }
            _ => Err(TransportError::Closed),
        }
    }

    fn recv(&self, source: &Address) -> TransportResult<Vec<u8>> {
        match source {
            Address::Local(id) => {
                let (_, rx) = self.get_or_create_channel(id);
                rx.recv()
            }
            _ => Err(TransportError::Closed),
        }
    }

    fn supports(&self, address: &Address) -> bool {
        matches!(address, Address::Local(_))
    }
}

/// Transport manager that routes messages to appropriate transport
pub struct TransportManager {
    transports: Vec<Box<dyn Transport>>,
}

impl TransportManager {
    /// Create a manager routing local addresses in-memory and remote addresses
    /// over the network transport.
    pub fn new() -> Self {
        Self {
            transports: vec![
                Box::new(InMemoryTransport::new()),
                Box::new(NetworkTransport {}),
            ],
        }
    }

    /// Send `data` via the first registered transport supporting `target`.
    ///
    /// # Errors
    /// Returns [`TransportError::Closed`] when no transport supports `target`;
    /// otherwise propagates the selected transport's send error.
    pub fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()> {
        for transport in &self.transports {
            if transport.supports(target) {
                return transport.send(target, data);
            }
        }
        Err(TransportError::Closed)
    }

    /// Receive from the first registered transport supporting `source`.
    ///
    /// # Errors
    /// Returns [`TransportError::Closed`] when no transport supports `source`;
    /// otherwise propagates the selected transport's receive error.
    pub fn recv(&self, source: &Address) -> TransportResult<Vec<u8>> {
        for transport in &self.transports {
            if transport.supports(source) {
                return transport.recv(source);
            }
        }
        Err(TransportError::Closed)
    }
}

/// Tracks the connection state of remote/local endpoints.
pub struct ConnectionManager {
    connections: Arc<Mutex<HashMap<Address, ConnectionState>>>,
}

/// Observable state of a tracked connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// The endpoint is currently connected.
    Connected,
    /// The endpoint was connected and has since disconnected.
    Disconnected,
}

impl ConnectionManager {
    /// Create a manager tracking no endpoints.
    pub fn new() -> Self {
        Self {
            connections: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Mark `address` as connected.
    pub fn connect(&self, address: &Address) -> TransportResult<()> {
        let mut conns = crate::lock_mutex(&self.connections);
        conns.insert(address.clone(), ConnectionState::Connected);
        Ok(())
    }

    /// Mark `address` as disconnected.
    pub fn disconnect(&self, address: &Address) -> TransportResult<()> {
        let mut conns = crate::lock_mutex(&self.connections);
        conns.insert(address.clone(), ConnectionState::Disconnected);
        Ok(())
    }

    /// Current tracked state of `address`, or `None` if never seen.
    #[must_use]
    pub fn state(&self, address: &Address) -> Option<ConnectionState> {
        crate::lock_mutex(&self.connections).get(address).copied()
    }

    /// Whether `address` is currently connected.
    #[must_use]
    pub fn is_connected(&self, address: &Address) -> bool {
        self.state(address) == Some(ConnectionState::Connected)
    }

    /// All currently-connected addresses.
    #[must_use]
    pub fn connected_addresses(&self) -> Vec<Address> {
        crate::lock_mutex(&self.connections)
            .iter()
            .filter(|(_, state)| **state == ConnectionState::Connected)
            .map(|(addr, _)| addr.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_compatibility() {
        let (tx, rx) = mpmc::<i32>(10);

        assert!(tx.send(42).is_ok());
        assert_eq!(rx.recv().unwrap(), 42);
    }

    #[test]
    fn test_in_memory_transport() {
        let transport1 = InMemoryTransport::new();
        let transport2 = InMemoryTransport::new();

        // Register transports with each other for routing
        // This would require a more robust mechanism for inter-transport communication
        // For now, we'll just check if they can send/recv to/from themselves
        assert!(transport1
            .send(&Address::Local("t1".to_string()), vec![1])
            .is_ok());
        assert_eq!(
            transport1.recv(&Address::Local("t1".to_string())).unwrap(),
            vec![1]
        );

        assert!(transport2
            .send(&Address::Local("t2".to_string()), vec![2])
            .is_ok());
        assert_eq!(
            transport2.recv(&Address::Local("t2".to_string())).unwrap(),
            vec![2]
        );
    }

    #[test]
    fn network_transport_transfers_length_prefixed_remote_bytes() {
        let transport = NetworkTransport {};
        let address = loopback_remote_address();
        let payload = b"server route payload".to_vec();
        let expected = payload.clone();
        let receiver_address = Address::Remote(address.clone());
        let receiver = std::thread::spawn(move || transport.recv(&receiver_address).unwrap());

        std::thread::sleep(std::time::Duration::from_millis(10));
        NetworkTransport {}
            .send(&Address::Remote(address), payload)
            .unwrap();

        assert_eq!(receiver.join().unwrap(), expected);
    }

    #[test]
    fn transport_manager_routes_remote_bytes_through_network_transport() {
        let manager = TransportManager::new();
        let address = loopback_remote_address();
        let payload = b"transport manager remote payload".to_vec();
        let expected = payload.clone();
        let receiver_address = Address::Remote(address.clone());
        let receiver =
            std::thread::spawn(move || TransportManager::new().recv(&receiver_address).unwrap());

        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.send(&Address::Remote(address), payload).unwrap();

        assert_eq!(receiver.join().unwrap(), expected);
    }

    #[test]
    fn connection_manager_tracks_and_reports_state() {
        let mgr = ConnectionManager::new();
        let addr = Address::Local("node".to_string());

        assert_eq!(mgr.state(&addr), None);
        assert!(!mgr.is_connected(&addr));
        assert!(mgr.connected_addresses().is_empty());

        mgr.connect(&addr).unwrap();
        assert!(mgr.is_connected(&addr));
        assert_eq!(mgr.state(&addr), Some(ConnectionState::Connected));
        assert_eq!(mgr.connected_addresses(), vec![addr.clone()]);

        mgr.disconnect(&addr).unwrap();
        assert!(!mgr.is_connected(&addr));
        assert_eq!(mgr.state(&addr), Some(ConnectionState::Disconnected));
        assert!(mgr.connected_addresses().is_empty());
    }

    fn loopback_remote_address() -> RemoteAddress {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        RemoteAddress {
            host: "127.0.0.1".to_string(),
            port,
            service: "moirai-test".to_string(),
        }
    }
}
