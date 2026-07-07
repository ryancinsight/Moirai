//! Unified transport layer for Moirai concurrency library.
//!
//! This module provides transport abstractions that work across different
//! communication boundaries: threads, processes, and machines. It builds on
//! top of the core channel primitives to provide location-transparent messaging.
//!
//! # Design Principles
//! - Location transparency: same API for local and remote communication
//! - Zero-copy optimization for local transport
//! - Pluggable transport backends (in-memory, IPC, network)

#![allow(clippy::new_without_default)]
#![allow(clippy::unwrap_or_default)]
#![deny(missing_docs)]
//! - Integration with Moirai scheduler for optimal performance

#[cfg(any(unix, windows))]
mod ipc;
mod network;
pub mod payload;
pub mod process;
pub mod remote_task;
#[cfg(feature = "scheduler-routes")]
pub mod route;
pub mod safe_channel;

use moirai_core::channel::{mpmc, MpmcReceiver, MpmcSender};
use moirai_core::constants::DEFAULT_MPMC_CAPACITY;
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// Crate-wide lock policy: recover from poisoning instead of propagating the
/// panic. Guarded state here (channel maps, subscription lists, connection
/// states) stays structurally valid under a poisoned lock — a writer that
/// panicked mid-critical-section cannot leave a torn invariant in these maps —
/// so continuing with the recovered guard is sound. Matches the pal reactor
/// backends' `lock_mutex` helpers.
pub(crate) fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Poison-recovering read lock; see [`lock_mutex`] for the policy rationale.
fn read_rwlock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// Poison-recovering write lock; see [`lock_mutex`] for the policy rationale.
fn write_rwlock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

// Re-export core channel types for compatibility
/// Shared-memory same-machine IPC transport (Unix/Windows only).
#[cfg(any(unix, windows))]
pub use ipc::IpcTransport;
pub use moirai_core::channel::{
    ChannelError as TransportError, MpmcReceiver as Receiver, MpmcSender as Sender,
};
pub use network::NetworkTransport;
#[cfg(feature = "network")]
pub use network::TcpTransport;
pub(crate) use network::{read_network_frame_from_stream, NETWORK_IO_TIMEOUT};
// The canonical typed cross-boundary channel: rkyv-style archive serialization
// over a transport (zero-copy borrowed views on receive).
pub use safe_channel::{
    ArchiveSerialize, ArchiveView, ArchivedMessage, ArchivedUniversalReceiver,
    ArchivedUniversalSender,
};

/// Result type for transport operations
pub type TransportResult<T> = Result<T, TransportError>;

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

/// Transport trait for different communication mechanisms
pub trait Transport: Send + Sync {
    /// Send a message to the specified address
    fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()>;

    /// Receive a message from the specified address
    fn recv(&self, source: &Address) -> TransportResult<Vec<u8>>;

    /// Check if the transport supports the given address
    fn supports(&self, address: &Address) -> bool;
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

// A typed cross-boundary channel over a transport is provided by the rkyv-style
// archive channels in `safe_channel` (`ArchivedUniversalSender<T: ArchiveSerialize>`
// / `ArchivedUniversalReceiver<T: ArchiveView>`), re-exported below. The previous
// `UniversalChannel<T: Send>` / `UniversalSender` / `UniversalReceiver` were
// non-functional placeholders (their `send`/`recv` ignored their argument and
// returned `Closed`): a channel generic over an arbitrary `Send` `T` cannot
// serialize the value for transport without a serialization bound, which is
// exactly what the archive traits add. They were removed in favor of the working
// archive channels rather than left as mocks.

/// Remote address for cross-machine communication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteAddress {
    /// Remote host name or IP address.
    pub host: String,
    /// Remote TCP port.
    pub port: u16,
    /// Service label carried in the address display form.
    pub service: String,
}

impl fmt::Display for RemoteAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}://{}:{}", self.service, self.host, self.port)
    }
}

/// Topic-based pub/sub router that delivers published messages to every
/// subscribed [`Address`] over a shared transport.
///
/// The router is generic over the backing [`Transport`] so delivery is
/// monomorphized and zero-cost; the transport must be the *same instance* the
/// subscribers receive from (e.g. one `Arc<InMemoryTransport>`), since in-memory
/// channels are keyed by address within a single transport instance. The prior
/// implementation constructed a throwaway `InMemoryTransport` per send and so
/// silently discarded every message.
pub struct MessageRouter<T: Transport> {
    transport: Arc<T>,
    subscriptions: Mutex<HashMap<String, Vec<Address>>>,
}

impl<T: Transport> MessageRouter<T> {
    /// Create a router that delivers over `transport`.
    pub fn new(transport: Arc<T>) -> Self {
        Self {
            transport,
            subscriptions: Mutex::new(HashMap::new()),
        }
    }

    /// Subscribe `address` to `topic`. Duplicate (topic, address) pairs are
    /// ignored so a message is delivered to each subscriber exactly once.
    pub fn subscribe(&self, topic: &str, address: Address) {
        let mut subs = lock_mutex(&self.subscriptions);
        let entry = subs.entry(topic.to_string()).or_default();
        if !entry.contains(&address) {
            entry.push(address);
        }
    }

    /// Remove `address` from `topic`. Returns `true` if a subscription was
    /// removed.
    pub fn unsubscribe(&self, topic: &str, address: &Address) -> bool {
        let mut subs = lock_mutex(&self.subscriptions);
        if let Some(entry) = subs.get_mut(topic) {
            let before = entry.len();
            entry.retain(|a| a != address);
            let removed = entry.len() != before;
            if entry.is_empty() {
                subs.remove(topic);
            }
            return removed;
        }
        false
    }

    /// Publish `data` to every subscriber of `topic` via the shared transport.
    ///
    /// Returns the number of subscribers the message was delivered to. Delivery
    /// is fail-fast: the first transport error is propagated (after the
    /// subscribers ahead of it have already received the message).
    ///
    /// # Errors
    /// Propagates the first per-subscriber transport send error.
    pub fn publish(&self, topic: &str, data: Vec<u8>) -> TransportResult<usize> {
        // Snapshot the subscriber list so the transport sends happen without the
        // subscriptions lock held (a subscriber's send must not block resubscribe).
        let targets: Vec<Address> = {
            let subs = lock_mutex(&self.subscriptions);
            match subs.get(topic) {
                Some(addresses) => addresses.clone(),
                None => return Ok(0),
            }
        };

        // `Transport::send` takes ownership (each in-memory subscriber channel
        // stores its own `Vec<u8>`), so N subscribers need N owned buffers —
        // but only N-1 copies: the caller's original buffer is moved to the
        // final subscriber instead of being cloned and dropped.
        let mut delivered = 0;
        let Some(last) = targets.len().checked_sub(1) else {
            return Ok(0);
        };
        let mut data = Some(data);
        for (index, addr) in targets.iter().enumerate() {
            let payload = if index == last {
                data.take()
                    .expect("invariant: original buffer moved exactly once, at the last subscriber")
            } else {
                data.as_ref()
                    .expect("invariant: original buffer present until the last subscriber")
                    .clone()
            };
            self.transport.send(addr, payload)?;
            delivered += 1;
        }
        Ok(delivered)
    }

    /// Number of distinct addresses subscribed to `topic`.
    pub fn subscriber_count(&self, topic: &str) -> usize {
        lock_mutex(&self.subscriptions)
            .get(topic)
            .map_or(0, Vec::len)
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
        let mut conns = lock_mutex(&self.connections);
        conns.insert(address.clone(), ConnectionState::Connected);
        Ok(())
    }

    /// Mark `address` as disconnected.
    pub fn disconnect(&self, address: &Address) -> TransportResult<()> {
        let mut conns = lock_mutex(&self.connections);
        conns.insert(address.clone(), ConnectionState::Disconnected);
        Ok(())
    }

    /// Current tracked state of `address`, or `None` if never seen.
    #[must_use]
    pub fn state(&self, address: &Address) -> Option<ConnectionState> {
        lock_mutex(&self.connections).get(address).copied()
    }

    /// Whether `address` is currently connected.
    #[must_use]
    pub fn is_connected(&self, address: &Address) -> bool {
        self.state(address) == Some(ConnectionState::Connected)
    }

    /// All currently-connected addresses.
    #[must_use]
    pub fn connected_addresses(&self) -> Vec<Address> {
        lock_mutex(&self.connections)
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
        let (tx, rx) = moirai_core::channel::mpmc::<i32>(10);

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

    #[test]
    fn message_router_delivers_to_each_subscriber_once() {
        let transport = Arc::new(InMemoryTransport::new());
        let router = MessageRouter::new(Arc::clone(&transport));
        let sub_a = Address::Local("sub_a".to_string());
        let sub_b = Address::Local("sub_b".to_string());

        router.subscribe("topic", sub_a.clone());
        router.subscribe("topic", sub_b.clone());
        router.subscribe("topic", sub_a.clone()); // duplicate ignored
        assert_eq!(router.subscriber_count("topic"), 2);

        let delivered = router.publish("topic", vec![1, 2, 3]).unwrap();
        assert_eq!(delivered, 2);

        // Both subscribers actually receive the message through the shared
        // transport (the prior throwaway-transport implementation delivered none).
        assert_eq!(transport.recv(&sub_a).unwrap(), vec![1, 2, 3]);
        assert_eq!(transport.recv(&sub_b).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn message_router_single_subscriber_receives_moved_buffer() {
        // N = 1 exercises the zero-clone path: the caller's buffer is moved to
        // the sole subscriber without an intermediate copy.
        let transport = Arc::new(InMemoryTransport::new());
        let router = MessageRouter::new(Arc::clone(&transport));
        let sub = Address::Local("solo".to_string());
        router.subscribe("t", sub.clone());

        let delivered = router.publish("t", vec![7, 8, 9]).unwrap();
        assert_eq!(delivered, 1);
        assert_eq!(transport.recv(&sub).unwrap(), vec![7, 8, 9]);
    }

    #[test]
    fn message_router_unknown_topic_delivers_nothing() {
        let transport = Arc::new(InMemoryTransport::new());
        let router = MessageRouter::new(transport);
        assert_eq!(router.publish("absent", vec![0]).unwrap(), 0);
    }

    #[test]
    fn message_router_unsubscribe_stops_delivery() {
        let transport = Arc::new(InMemoryTransport::new());
        let router = MessageRouter::new(Arc::clone(&transport));
        let sub = Address::Local("s".to_string());

        router.subscribe("t", sub.clone());
        assert!(router.unsubscribe("t", &sub));
        assert!(
            !router.unsubscribe("t", &sub),
            "second unsubscribe is a no-op"
        );
        assert_eq!(router.subscriber_count("t"), 0);
        assert_eq!(router.publish("t", vec![9]).unwrap(), 0);
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
}
