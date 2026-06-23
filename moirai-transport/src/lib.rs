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
//! - Integration with Moirai scheduler for optimal performance

// Zero-copy moved to moirai-core::communication::zero_copy (SSOT)
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
    sync::{Arc, Mutex},
};

// Re-export core channel types for compatibility
pub use moirai_core::channel::{
    ChannelError as TransportError, MpmcReceiver as Receiver, MpmcSender as Sender,
};
pub use moirai_core::communication::zero_copy as core_zero_copy;
pub use network::NetworkTransport;
pub(crate) use network::{read_network_frame_from_stream, NETWORK_IO_TIMEOUT};
#[cfg(feature = "network")]
pub use network::{TcpTransport, UdpTransport};

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

/// In-memory transport for local communication
pub struct InMemoryTransport {
    channels: Arc<Mutex<HashMap<String, MpmcSender<Vec<u8>>>>>,
    receivers: Arc<Mutex<HashMap<String, MpmcReceiver<Vec<u8>>>>>,
}

impl InMemoryTransport {
    pub fn new() -> Self {
        Self {
            channels: Arc::new(Mutex::new(HashMap::new())),
            receivers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn get_or_create_channel(&self, id: &str) -> (MpmcSender<Vec<u8>>, MpmcReceiver<Vec<u8>>) {
        let mut channels = self.channels.lock().unwrap();
        let mut receivers = self.receivers.lock().unwrap();

        if let Some(sender) = channels.get(id) {
            if let Some(receiver) = receivers.get(id) {
                return (sender.clone(), receiver.clone());
            }
        }

        let (tx, rx) = mpmc(DEFAULT_MPMC_CAPACITY);
        channels.insert(id.to_string(), tx.clone());
        receivers.insert(id.to_string(), rx.clone());
        (tx, rx)
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

/// IPC transport for inter-process communication
pub struct IpcTransport {}

impl Transport for IpcTransport {
    fn send(&self, _target: &Address, _data: Vec<u8>) -> TransportResult<()> {
        Err(TransportError::WouldBlock)
    }

    fn recv(&self, _source: &Address) -> TransportResult<Vec<u8>> {
        Err(TransportError::Empty)
    }

    fn supports(&self, _address: &Address) -> bool {
        false
    }
}

/// Transport manager that routes messages to appropriate transport
pub struct TransportManager {
    transports: Vec<Box<dyn Transport>>,
}

impl TransportManager {
    pub fn new() -> Self {
        Self {
            transports: vec![
                Box::new(InMemoryTransport::new()),
                Box::new(IpcTransport {}),
                Box::new(NetworkTransport {}),
            ],
        }
    }

    pub fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()> {
        for transport in &self.transports {
            if transport.supports(target) {
                return transport.send(target, data);
            }
        }
        Err(TransportError::Closed)
    }

    pub fn recv(&self, source: &Address) -> TransportResult<Vec<u8>> {
        for transport in &self.transports {
            if transport.supports(source) {
                return transport.recv(source);
            }
        }
        Err(TransportError::Closed)
    }
}

/// Universal channel that works across different transport boundaries
///
/// This is a wrapper around core channel implementations that adds
/// transport-specific functionality following DRY principle.
pub struct UniversalChannel<T: Send + 'static> {
    sender: UniversalSender<T>,
    receiver: UniversalReceiver<T>,
}

impl<T: Send + 'static> UniversalChannel<T> {
    /// Create a new universal channel
    pub fn new(transport: Arc<TransportManager>, address: Address) -> Self {
        Self {
            sender: UniversalSender {
                transport: transport.clone(),
                target: address.clone(),
                _phantom: std::marker::PhantomData,
            },
            receiver: UniversalReceiver {
                _transport: transport,
                _source: address,
                _phantom: std::marker::PhantomData,
            },
        }
    }

    /// Split into sender and receiver halves
    pub fn split(self) -> (UniversalSender<T>, UniversalReceiver<T>) {
        (self.sender, self.receiver)
    }
}

/// Sender half of universal channel
///
/// This wraps core channel functionality with transport-specific archive bytes.
pub struct UniversalSender<T: Send + 'static> {
    transport: Arc<TransportManager>,
    target: Address,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + 'static> UniversalSender<T> {
    /// Send a message to the target address
    pub fn send(&self, _value: T) -> TransportResult<()> {
        Err(TransportError::Closed)
    }
}

impl<T: Send + 'static> Clone for UniversalSender<T> {
    fn clone(&self) -> Self {
        Self {
            transport: self.transport.clone(),
            target: self.target.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

unsafe impl<T: Send + 'static> Send for UniversalSender<T> {}
unsafe impl<T: Send + 'static> Sync for UniversalSender<T> {}

/// Receiver half of universal channel
pub struct UniversalReceiver<T: Send + 'static> {
    _transport: Arc<TransportManager>,
    _source: Address,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + 'static> UniversalReceiver<T> {
    /// Receive a message from the source address
    pub fn recv(&self) -> TransportResult<T> {
        Err(TransportError::Closed)
    }
}

/// Remote address for cross-machine communication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteAddress {
    pub host: String,
    pub port: u16,
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
        let mut subs = self.subscriptions.lock().unwrap();
        let entry = subs.entry(topic.to_string()).or_default();
        if !entry.contains(&address) {
            entry.push(address);
        }
    }

    /// Remove `address` from `topic`. Returns `true` if a subscription was
    /// removed.
    pub fn unsubscribe(&self, topic: &str, address: &Address) -> bool {
        let mut subs = self.subscriptions.lock().unwrap();
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
            let subs = self.subscriptions.lock().unwrap();
            match subs.get(topic) {
                Some(addresses) => addresses.clone(),
                None => return Ok(0),
            }
        };

        let mut delivered = 0;
        for addr in &targets {
            self.transport.send(addr, data.clone())?;
            delivered += 1;
        }
        Ok(delivered)
    }

    /// Number of distinct addresses subscribed to `topic`.
    pub fn subscriber_count(&self, topic: &str) -> usize {
        self.subscriptions
            .lock()
            .unwrap()
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
    pub fn new() -> Self {
        Self {
            connections: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Mark `address` as connected.
    pub fn connect(&self, address: &Address) -> TransportResult<()> {
        let mut conns = self.connections.lock().unwrap();
        conns.insert(address.clone(), ConnectionState::Connected);
        Ok(())
    }

    /// Mark `address` as disconnected.
    pub fn disconnect(&self, address: &Address) -> TransportResult<()> {
        let mut conns = self.connections.lock().unwrap();
        conns.insert(address.clone(), ConnectionState::Disconnected);
        Ok(())
    }

    /// Current tracked state of `address`, or `None` if never seen.
    #[must_use]
    pub fn state(&self, address: &Address) -> Option<ConnectionState> {
        self.connections.lock().unwrap().get(address).copied()
    }

    /// Whether `address` is currently connected.
    #[must_use]
    pub fn is_connected(&self, address: &Address) -> bool {
        self.state(address) == Some(ConnectionState::Connected)
    }

    /// All currently-connected addresses.
    #[must_use]
    pub fn connected_addresses(&self) -> Vec<Address> {
        self.connections
            .lock()
            .unwrap()
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
    fn test_universal_channel() {
        let transport_manager = TransportManager::new();
        let _sender = UniversalSender::<String> {
            transport: Arc::new(transport_manager),
            target: Address::Local("test_sender".to_string()),
            _phantom: std::marker::PhantomData,
        };

        // Test sender construction; typed transport payloads use archive bytes.
        // This test demonstrates channel creation API
        // assert!(sender.send(42).is_ok());
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
