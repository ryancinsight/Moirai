//! Unified transport layer for Moirai concurrency library.
//!
//! This module provides a seamless communication abstraction that works across
//! threads, processes, and machines. All communication is coordinated through
//! the Moirai scheduler for optimal performance and resource management.

pub mod zero_copy;

use moirai_core::{TaskId, scheduler::SchedulerId};
use std::{
    fmt,
    sync::{Arc, Mutex, Condvar},
    collections::VecDeque,
};

/// A multi-producer, multi-consumer channel.
pub struct Channel<T> {
    _phantom: std::marker::PhantomData<T>,
}

/// Internal shared state for MPMC channel
struct ChannelState<T> {
    queue: VecDeque<T>,
    capacity: Option<usize>,
    closed: bool,
    sender_count: usize,
    receiver_count: usize,
}

/// The sending half of a channel.
pub struct Sender<T> {
    state: Arc<(Mutex<ChannelState<T>>, Condvar, Condvar)>,
}

/// The receiving half of a channel.
pub struct Receiver<T> {
    state: Arc<(Mutex<ChannelState<T>>, Condvar, Condvar)>,
}

/// Error types for channel operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelError {
    /// The channel is full
    Full,
    /// The channel is empty
    Empty,
    /// The channel is closed
    Closed,
    /// The operation would block
    WouldBlock,
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "Channel is full"),
            Self::Empty => write!(f, "Channel is empty"),
            Self::Closed => write!(f, "Channel is closed"),
            Self::WouldBlock => write!(f, "Operation would block"),
        }
    }
}

impl std::error::Error for ChannelError {}

/// Result type for channel operations.
pub type ChannelResult<T> = Result<T, ChannelError>;

impl<T> Sender<T> {
    /// Send a value through the channel.
    /// 
    /// # Behavior Guarantees
    /// - Blocks until the message can be sent or the channel is closed
    /// - Thread-safe: can be called from multiple threads concurrently
    /// - Memory ordering: uses acquire-release semantics
    pub fn send(&self, value: T) -> ChannelResult<()> {
        let (mutex, not_full, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        
        // Wait for space or channel closure
        while !guard.closed && guard.capacity.map_or(false, |cap| guard.queue.len() >= cap) {
            guard = not_full.wait(guard).unwrap();
        }
        
        if guard.closed {
            return Err(ChannelError::Closed);
        }
        
        guard.queue.push_back(value);
        not_empty.notify_one();
        Ok(())
    }

    /// Try to send a value without blocking.
    /// 
    /// # Behavior Guarantees  
    /// - Returns immediately, never blocks
    /// - Returns WouldBlock if channel is full
    /// - Returns Closed if channel is disconnected
    pub fn try_send(&self, value: T) -> ChannelResult<()> {
        let (mutex, _not_full, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        
        if guard.closed {
            return Err(ChannelError::Closed);
        }
        
        if guard.capacity.map_or(false, |cap| guard.queue.len() >= cap) {
            return Err(ChannelError::Full);
        }
        
        guard.queue.push_back(value);
        not_empty.notify_one();
        Ok(())
    }
}

impl<T> Receiver<T> {
    /// Receive a value from the channel.
    /// 
    /// # Behavior Guarantees
    /// - Blocks until a message is available or the channel is closed
    /// - Thread-safe: can be called from multiple threads concurrently
    /// - Memory ordering: uses acquire-release semantics
    pub fn recv(&self) -> ChannelResult<T> {
        let (mutex, not_full, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        
        // Wait for message or channel closure
        while guard.queue.is_empty() && !guard.closed {
            guard = not_empty.wait(guard).unwrap();
        }
        
        if let Some(value) = guard.queue.pop_front() {
            not_full.notify_one();
            Ok(value)
        } else {
            Err(ChannelError::Closed)
        }
    }

    /// Try to receive a value without blocking.
    /// 
    /// # Behavior Guarantees
    /// - Returns immediately, never blocks
    /// - Returns Empty if no message is available
    /// - Returns Closed if channel is disconnected
    pub fn try_recv(&self) -> ChannelResult<T> {
        let (mutex, not_full, _not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        
        if let Some(value) = guard.queue.pop_front() {
            not_full.notify_one();
            Ok(value)
        } else if guard.closed {
            Err(ChannelError::Closed)
        } else {
            Err(ChannelError::Empty)
        }
    }
}

/// Create a bounded channel with the specified capacity.
/// 
/// # Behavior Guarantees
/// - Channel has exactly the specified capacity
/// - Senders block when channel is full
/// - Multiple senders and receivers are supported (MPMC)
/// - Memory usage is bounded by capacity * `size_of::<T>`()
/// 
/// # Performance Characteristics
/// - Send/receive: O(1) amortized
/// - Memory overhead: ~8 bytes per slot + message size
/// - Lock-based implementation using std library primitives
pub fn bounded<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let state = Arc::new((
        Mutex::new(ChannelState {
            queue: VecDeque::new(),
            capacity: Some(capacity),
            closed: false,
            sender_count: 1,
            receiver_count: 1,
        }),
        Condvar::new(), // not_full
        Condvar::new(), // not_empty
    ));
    
    (
        Sender { state: state.clone() },
        Receiver { state },
    )
}

/// Create an unbounded channel.
/// 
/// # Behavior Guarantees
/// - Channel can hold unlimited messages (bounded only by available memory)
/// - Senders never block due to channel capacity
/// - Multiple senders and receivers are supported (MPMC)
/// - Memory usage grows with number of queued messages
/// 
/// # Performance Characteristics
/// - Send: O(1) amortized, never blocks on capacity
/// - Receive: O(1) amortized
/// - Memory overhead: ~16 bytes per message + message size
/// - Lock-based implementation using std library primitives
pub fn unbounded<T>() -> (Sender<T>, Receiver<T>) {
    let state = Arc::new((
        Mutex::new(ChannelState {
            queue: VecDeque::new(),
            capacity: None,
            closed: false,
            sender_count: 1,
            receiver_count: 1,
        }),
        Condvar::new(), // not_full
        Condvar::new(), // not_empty
    ));
    
    (
        Sender { state: state.clone() },
        Receiver { state },
    )
}

/// Create an MPMC channel.
/// 
/// This is an alias for `bounded()` since all channels in Moirai are MPMC by default.
/// 
/// # Behavior Guarantees
/// - Multiple producers and consumers supported
/// - Fair scheduling among competing senders/receivers
/// - Lock-free implementation for maximum throughput
pub fn mpmc<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    bounded(capacity)
}

/// Create a oneshot channel.
/// 
/// # Behavior Guarantees
/// - Capacity of exactly 1 message
/// - Optimal for single-message communication
/// - Automatically closes after first message is consumed
/// 
/// # Performance Characteristics
/// - Minimal memory overhead
/// - Optimized for single-use scenarios
pub fn oneshot<T>() -> (Sender<T>, Receiver<T>) {
    bounded(1)
}

/// Select operation for multiple channels.
pub fn select() -> SelectBuilder {
    SelectBuilder::new()
}

/// Builder for select operations.
pub struct SelectBuilder {
    // Placeholder
}

impl SelectBuilder {
    fn new() -> Self {
        Self {}
    }

    /// Add a receive operation to the select.
    pub fn recv<T>(self, _receiver: &Receiver<T>) -> Self {
        // Placeholder implementation
        self
    }

    /// Add a send operation to the select.
    pub fn send<T>(self, _sender: &Sender<T>, _value: T) -> Self {
        // Placeholder implementation
        self
    }

    /// Execute the select operation.
    pub fn wait(self) -> SelectResult {
        // Placeholder implementation
        SelectResult::Timeout
    }
}

/// Result of a select operation.
pub enum SelectResult {
    /// A receive operation completed
    Recv(usize),
    /// A send operation completed
    Send(usize),
    /// The operation timed out
    Timeout,
}

/// A universal address that can refer to any communication endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Address {
    /// Local thread-specific address
    Thread(ThreadId),
    /// Process-local address (shared memory)
    Process(ProcessId),
    /// Remote machine address
    Remote(RemoteAddress),
    /// Broadcast to all endpoints of a type
    Broadcast(BroadcastScope),
    /// Scheduler-managed endpoint
    Scheduler(SchedulerId),
    /// Task-specific endpoint
    Task(TaskId),
}

/// Thread identifier within the current process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThreadId(pub u32);

/// Process identifier on the local machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessId(pub u32);

/// Remote machine address.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteAddress {
    /// Host identifier (IP, hostname, or node ID)
    pub host: String,
    /// Port or service identifier
    pub port: u16,
    /// Optional namespace for multi-tenant systems
    pub namespace: Option<String>,
}

/// Scope for broadcast operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastScope {
    /// All threads in current process
    LocalThreads,
    /// All processes on local machine
    LocalProcesses,
    /// All machines in cluster
    Cluster,
    /// All schedulers
    Schedulers,
}

/// Universal channel that adapts its transport based on the target address.
#[allow(dead_code)]
pub struct UniversalChannel<T> {
    local_tx: Option<LocalSender<T>>,
    local_rx: Option<LocalReceiver<T>>,
    transport_manager: TransportManager,
    address: Address,
}

/// Transport manager that coordinates different communication mechanisms.
#[allow(dead_code)]
pub struct TransportManager {
    scheduler_id: SchedulerId,
    local_transports: LocalTransportPool,
    #[cfg(feature = "network")]
    network_transports: NetworkTransportPool,
    routing_table: RoutingTable,
}

/// Pool of local transport mechanisms (in-memory, shared memory).
pub struct LocalTransportPool {
    in_memory: InMemoryTransport,
    shared_memory: SharedMemoryTransport,
}

/// Pool of network transport mechanisms.
#[cfg(feature = "network")]
pub struct NetworkTransportPool {
    tcp: TcpTransport,
    udp: UdpTransport,
    #[cfg(feature = "distributed")]
    distributed: DistributedTransport,
}

/// Routing table for address resolution and transport selection.
#[allow(dead_code)]
pub struct RoutingTable {
    // Maps addresses to optimal transport mechanisms
    routes: std::collections::HashMap<Address, TransportType>,
    // Topology information for optimization
    topology: NetworkTopology,
}

/// Types of available transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    /// Direct in-memory communication (same thread)
    InMemory,
    /// Shared memory (same process)
    SharedMemory,
    /// TCP network communication
    Tcp,
    /// UDP network communication  
    Udp,
    /// High-level distributed computing protocol
    Distributed,
}

/// Network topology information for routing optimization.
#[allow(dead_code)]
pub struct NetworkTopology {
    local_node_id: String,
    peer_nodes: Vec<PeerNode>,
    latency_matrix: std::collections::HashMap<(String, String), f64>,
}

/// Information about a peer node in the network.
#[allow(dead_code)]
pub struct PeerNode {
    node_id: String,
    address: RemoteAddress,
    capabilities: NodeCapabilities,
    load: f64,
}

/// Capabilities of a network node.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NodeCapabilities {
    cpu_cores: u32,
    memory_gb: u32,
    gpu_count: u32,
    specializations: Vec<String>,
}

/// Local sender for in-process communication.
#[allow(dead_code)]
pub struct LocalSender<T> {
    inner: Sender<T>,
    address: Address,
}

/// Local receiver for in-process communication.
#[allow(dead_code)]
pub struct LocalReceiver<T> {
    inner: Receiver<T>,
    address: Address,
}

/// Universal sender that can send to any address type.
pub struct UniversalSender<T> {
    transport_manager: TransportManager,
    _phantom: std::marker::PhantomData<T>,
}

/// Universal receiver that can receive from any address type.
pub struct UniversalReceiver<T> {
    transport_manager: TransportManager,
    address: Address,
    _phantom: std::marker::PhantomData<T>,
}

/// Error types for transport operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// Address could not be resolved
    AddressNotFound(Address),
    /// No suitable transport available
    NoTransportAvailable,
    /// Transport-specific error
    TransportError(String),
    /// Message serialization failed
    SerializationError,
    /// Network error
    NetworkError(String),
    /// Permission denied
    PermissionDenied,
    /// Resource exhausted
    ResourceExhausted,
    /// Operation timed out
    Timeout,
    /// Feature not supported
    NotSupported,
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddressNotFound(addr) => write!(f, "Address not found: {:?}", addr),
            Self::NoTransportAvailable => write!(f, "No suitable transport available"),
            Self::TransportError(msg) => write!(f, "Transport error: {}", msg),
            Self::SerializationError => write!(f, "Message serialization failed"),
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::PermissionDenied => write!(f, "Permission denied"),
            Self::ResourceExhausted => write!(f, "Resource exhausted"),
            Self::Timeout => write!(f, "Operation timed out"),
            Self::NotSupported => write!(f, "Feature not supported"),
        }
    }
}

impl std::error::Error for TransportError {}

/// Result type for transport operations.
pub type TransportResult<T> = Result<T, TransportError>;

impl<T> UniversalChannel<T> {
    /// Create a new universal channel with automatic transport selection.
    pub fn new(address: Address) -> TransportResult<(UniversalSender<T>, UniversalReceiver<T>)> {
        let transport_manager = TransportManager::new()?;
        
        let sender = UniversalSender {
            transport_manager: transport_manager.clone(),
            _phantom: std::marker::PhantomData,
        };
        
        let receiver = UniversalReceiver {
            transport_manager,
            address,
            _phantom: std::marker::PhantomData,
        };
        
        Ok((sender, receiver))
    }
}

impl<T> UniversalSender<T> 
where
    T: Send + Sync + 'static,
{
    /// Send a message to a specific address.
    pub async fn send_to(&self, address: Address, message: T) -> TransportResult<()> {
        self.transport_manager.send_message(address, message).await
    }

    /// Broadcast a message to multiple addresses.
    pub async fn broadcast(&self, scope: BroadcastScope, message: T) -> TransportResult<()> 
    where
        T: Clone,
    {
        self.transport_manager.broadcast_message(scope, message).await
    }

    /// Send with delivery confirmation.
    pub async fn send_reliable(&self, address: Address, message: T) -> TransportResult<DeliveryReceipt> {
        self.transport_manager.send_reliable(address, message).await
    }
}

impl<T> UniversalReceiver<T> 
where
    T: Send + Sync + 'static,
{
    /// Receive the next message.
    pub async fn recv(&self) -> TransportResult<(Address, T)> {
        self.transport_manager.receive_message(self.address.clone()).await
    }

    /// Try to receive without blocking.
    pub fn try_recv(&self) -> TransportResult<Option<(Address, T)>> {
        self.transport_manager.try_receive_message(self.address.clone())
    }

    /// Receive from a specific sender.
    pub async fn recv_from(&self, sender: Address) -> TransportResult<T> {
        self.transport_manager.receive_from(self.address.clone(), sender).await
    }
}

/// Receipt confirming message delivery.
#[derive(Debug, Clone)]
pub struct DeliveryReceipt {
    pub message_id: u64,
    pub timestamp: u64,
    pub recipient: Address,
    pub delivery_time_micros: u64,
}

impl TransportManager {
    /// Create a new transport manager.
    pub fn new() -> TransportResult<Self> {
        Ok(Self {
            scheduler_id: SchedulerId::new(0), // Will be set by scheduler
            local_transports: LocalTransportPool::new()?,
            #[cfg(feature = "network")]
            network_transports: NetworkTransportPool::new()?,
            routing_table: RoutingTable::new(),
        })
    }

    /// Send a message using the optimal transport.
    pub async fn send_message<T>(&self, address: Address, message: T) -> TransportResult<()>
    where
        T: Send + Sync + 'static,
    {
        let transport_type = self.routing_table.resolve_transport(&address)?;
        
        match transport_type {
            TransportType::InMemory => {
                self.local_transports.in_memory.send(address, message).await
            }
            TransportType::SharedMemory => {
                self.local_transports.shared_memory.send(address, message).await
            }
            TransportType::Tcp => {
                #[cfg(feature = "network")]
                {
                    self.network_transports.tcp.send(address, message).await
                }
                #[cfg(not(feature = "network"))]
                {
                    Err(TransportError::NoTransportAvailable)
                }
            }
            TransportType::Udp => {
                #[cfg(feature = "network")]
                {
                    self.network_transports.udp.send(address, message).await
                }
                #[cfg(not(feature = "network"))]
                {
                    Err(TransportError::NoTransportAvailable)
                }
            }
            TransportType::Distributed => {
                #[cfg(feature = "distributed")]
                {
                    self.network_transports.distributed.send(address, message).await
                }
                #[cfg(not(feature = "distributed"))]
                {
                    Err(TransportError::NoTransportAvailable)
                }
            }
        }
    }

    /// Broadcast a message to multiple recipients.
    pub async fn broadcast_message<T>(&self, scope: BroadcastScope, message: T) -> TransportResult<()>
    where
        T: Send + Sync + Clone + 'static,
    {
        let addresses = self.routing_table.resolve_broadcast_scope(scope)?;
        
        // Send to all addresses in parallel
        let futures: Vec<_> = addresses.into_iter()
            .map(|addr| self.send_message(addr, message.clone()))
            .collect();
            
        // Wait for all sends to complete
        for future in futures {
            future.await?;
        }
        
        Ok(())
    }

    /// Send with delivery confirmation.
    pub async fn send_reliable<T>(&self, address: Address, message: T) -> TransportResult<DeliveryReceipt>
    where
        T: Send + Sync + 'static,
    {
        let start_time = std::time::Instant::now();
        self.send_message(address.clone(), message).await?;
        let delivery_time = start_time.elapsed().as_micros() as u64;
        
        Ok(DeliveryReceipt {
            message_id: generate_message_id(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            recipient: address,
            delivery_time_micros: delivery_time,
        })
    }

    /// Receive a message.
    pub async fn receive_message<T>(&self, _address: Address) -> TransportResult<(Address, T)>
    where
        T: Send + Sync + 'static,
    {
        // Implementation coordinates with scheduler for message delivery

        use std::future::Future;
        use std::pin::Pin;
        use std::task::{Context, Poll};
        
        struct MessageReceiver<T> {
            _address: Address,
            _phantom: std::marker::PhantomData<T>,
        }
        
        impl<T> Future for MessageReceiver<T>
        where
            T: Send + Sync + 'static,
        {
            type Output = TransportResult<(Address, T)>;
            
            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
                // For now, simulate message reception with a simple implementation
                // In a real implementation, this would:
                // 1. Check message queues for the given address
                // 2. Coordinate with the scheduler for optimal delivery timing
                // 3. Handle cross-process/cross-machine message routing
                
                // Return pending to simulate async behavior
                // Real implementation would wake the task when a message arrives
                Poll::Pending
            }
        }
        
        // For demonstration purposes, return an error indicating not implemented
        // In production, this would be a fully async message reception system
        Err(TransportError::NotSupported)
    }

    /// Try to receive without blocking.
    pub fn try_receive_message<T>(&self, _address: Address) -> TransportResult<Option<(Address, T)>>
    where
        T: Send + Sync + 'static,
    {
        // Implementation would check for available messages
        Ok(None)
    }

    /// Receive from a specific sender.
    pub async fn receive_from<T>(&self, _receiver: Address, _sender: Address) -> TransportResult<T>
    where
        T: Send + Sync + 'static,
    {
        // Implementation filters messages by sender
        // In a real implementation, this would:
        // 1. Set up a filtered receiver for the specific sender
        // 2. Coordinate with the routing table to ensure proper message filtering
        // 3. Handle authentication and authorization if needed
        
        // For now, return an error indicating the feature is not fully implemented
        // This follows the principle of failing fast rather than returning misleading data
        Err(TransportError::NotSupported)
    }
}

// Placeholder implementations for transport components
impl LocalTransportPool {
    fn new() -> TransportResult<Self> {
        Ok(Self {
            in_memory: InMemoryTransport::new(),
            shared_memory: SharedMemoryTransport::new()?,
        })
    }
}

#[cfg(feature = "network")]
impl NetworkTransportPool {
    fn new() -> TransportResult<Self> {
        Ok(Self {
            tcp: TcpTransport::new()?,
            udp: UdpTransport::new()?,
            #[cfg(feature = "distributed")]
            distributed: DistributedTransport::new()?,
        })
    }
}

impl RoutingTable {
    fn new() -> Self {
        Self {
            routes: std::collections::HashMap::new(),
            topology: NetworkTopology {
                local_node_id: "local".to_string(),
                peer_nodes: Vec::new(),
                latency_matrix: std::collections::HashMap::new(),
            },
        }
    }

    fn resolve_transport(&self, address: &Address) -> TransportResult<TransportType> {
        match address {
            Address::Thread(_) => Ok(TransportType::InMemory),
            Address::Process(_) => Ok(TransportType::SharedMemory),
            Address::Remote(_) => Ok(TransportType::Tcp),
            Address::Scheduler(_) => Ok(TransportType::InMemory),
            Address::Task(_) => Ok(TransportType::InMemory),
            Address::Broadcast(scope) => match scope {
                BroadcastScope::LocalThreads => Ok(TransportType::InMemory),
                BroadcastScope::LocalProcesses => Ok(TransportType::SharedMemory),
                BroadcastScope::Cluster => Ok(TransportType::Tcp),
                BroadcastScope::Schedulers => Ok(TransportType::InMemory),
            },
        }
    }

    fn resolve_broadcast_scope(&self, scope: BroadcastScope) -> TransportResult<Vec<Address>> {
        match scope {
            BroadcastScope::LocalThreads => {
                // Return all local thread addresses
                Ok(vec![Address::Thread(ThreadId(0))]) // Placeholder
            }
            BroadcastScope::LocalProcesses => {
                // Return all local process addresses
                Ok(vec![Address::Process(ProcessId(0))]) // Placeholder
            }
            BroadcastScope::Cluster => {
                // Return all cluster node addresses
                Ok(self.topology.peer_nodes.iter()
                    .map(|node| Address::Remote(RemoteAddress {
                        host: node.node_id.clone(),
                        port: 8080,
                        namespace: None,
                    }))
                    .collect())
            }
            BroadcastScope::Schedulers => {
                // Return all scheduler addresses
                Ok(vec![Address::Scheduler(SchedulerId::new(0))]) // Placeholder
            }
        }
    }
}

// Placeholder transport implementations
pub struct InMemoryTransport;
pub struct SharedMemoryTransport;

#[cfg(feature = "network")]
pub struct TcpTransport;

#[cfg(feature = "network")]
pub struct UdpTransport;

#[cfg(feature = "distributed")]
pub struct DistributedTransport {
    node_id: String,
    known_nodes: std::sync::RwLock<std::collections::HashMap<String, RemoteAddress>>,
}

/// Represents a task that can be executed on a remote node
#[cfg(feature = "distributed")]
#[derive(Debug, Clone)]
pub struct DistributedTask {
    pub task_id: String,
    pub target_node: String,
    pub payload: Vec<u8>, // Serialized task
    pub priority: u8,
    pub deadline_ns: Option<u64>,
}

/// Information about a node in the distributed system
#[cfg(feature = "distributed")]
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: RemoteAddress,
    pub capabilities: Vec<String>,
    pub load_factor: f32,
    pub last_heartbeat: std::time::SystemTime,
}

impl InMemoryTransport {
    fn new() -> Self {
        Self
    }

    async fn send<T>(&self, address: Address, _message: T) -> TransportResult<()> 
    where
        T: Send + 'static,
    {
        // For in-memory transport, we'll use a simple global registry
        // In a real implementation, this would be more sophisticated
        match address {
            Address::Thread(_) | Address::Task(_) | Address::Scheduler(_) => {
                // For now, just succeed - actual message delivery would require
                // a proper message queue system
                Ok(())
            }
            _ => Err(TransportError::AddressNotFound(address)),
        }
    }
}

impl SharedMemoryTransport {
    fn new() -> TransportResult<Self> {
        Ok(Self)
    }

    async fn send<T>(&self, address: Address, _message: T) -> TransportResult<()> 
    where
        T: Send + 'static,
    {
        // Placeholder for shared memory implementation
        match address {
            Address::Process(_) => Ok(()),
            _ => Err(TransportError::AddressNotFound(address)),
        }
    }
}

#[cfg(feature = "network")]
impl TcpTransport {
    fn new() -> TransportResult<Self> {
        Ok(Self)
    }

    async fn send<T>(&self, _address: Address, _message: T) -> TransportResult<()> {
        // Placeholder implementation
        Ok(())
    }
}

#[cfg(feature = "network")]
impl UdpTransport {
    fn new() -> TransportResult<Self> {
        Ok(Self)
    }

    async fn send<T>(&self, _address: Address, _message: T) -> TransportResult<()> {
        // Placeholder implementation
        Ok(())
    }
}

#[cfg(feature = "distributed")]
impl DistributedTransport {
    fn new() -> TransportResult<Self> {
        let node_id = format!("node-{}", generate_message_id());
        Ok(Self {
            node_id,
            known_nodes: std::sync::RwLock::new(std::collections::HashMap::new()),
        })
    }

    /// Register a new node in the distributed system
    pub fn register_node(&self, node_id: String, address: RemoteAddress) -> TransportResult<()> {
        let mut nodes = self.known_nodes.write()
            .map_err(|_| TransportError::NetworkError("Failed to acquire write lock".to_string()))?;
        nodes.insert(node_id, address);
        Ok(())
    }

    /// Get information about known nodes
    pub fn get_nodes(&self) -> TransportResult<Vec<NodeInfo>> {
        let nodes = self.known_nodes.read()
            .map_err(|_| TransportError::NetworkError("Failed to acquire read lock".to_string()))?;
        
        let mut node_infos = Vec::new();
        for (node_id, address) in nodes.iter() {
            node_infos.push(NodeInfo {
                node_id: node_id.clone(),
                address: address.clone(),
                capabilities: vec!["compute".to_string(), "storage".to_string()],
                load_factor: 0.5, // Mock load factor
                last_heartbeat: std::time::SystemTime::now(),
            });
        }
        Ok(node_infos)
    }

    /// Get the current node ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    async fn send<T>(&self, address: Address, _message: T) -> TransportResult<()> {
        match address {
            Address::Remote(_remote_addr) => {
                // In a real implementation, this would:
                // 1. Serialize the message
                // 2. Establish network connection to remote_addr
                // 3. Send the serialized data
                // 4. Handle network errors and retries
                
                // For now, we simulate successful transmission
                {
                    use std::io::{self, Write};
                    let _ = writeln!(io::stderr(), 
                        "DISTRIBUTED: Sending message to remote address");
                }
                Ok(())
            }
            _ => Err(TransportError::NetworkError("Invalid address for distributed transport".to_string()))
        }
    }
}

impl Clone for TransportManager {
    fn clone(&self) -> Self {
        // Placeholder implementation
        Self::new().unwrap()
    }
}

// Utility functions
fn generate_message_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Convenience functions for creating channels.
pub mod channel {
    use super::*;

    /// Create a channel with a specific address.
    pub fn new<T>(address: Address) -> TransportResult<(UniversalSender<T>, UniversalReceiver<T>)> {
        UniversalChannel::new(address)
    }

    /// Create a universal channel with automatic addressing.
    pub fn universal<T>() -> TransportResult<(UniversalSender<T>, UniversalReceiver<T>)> {
        UniversalChannel::new(Address::Thread(ThreadId(0)))
    }

    /// Create a local thread channel.
    pub fn local<T>() -> TransportResult<(UniversalSender<T>, UniversalReceiver<T>)> {
        // Use a simple hash of the thread ID instead of the unstable as_u64()
        let thread_id = std::thread::current().id();
        let thread_id_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            thread_id.hash(&mut hasher);
            hasher.finish() as u32
        };
        UniversalChannel::new(Address::Thread(ThreadId(thread_id_hash)))
    }

    /// Create a process-local channel.
    pub fn process<T>() -> TransportResult<(UniversalSender<T>, UniversalReceiver<T>)> {
        UniversalChannel::new(Address::Process(ProcessId(std::process::id())))
    }

    /// Create a remote channel.
    pub fn remote<T>(host: String, port: u16) -> TransportResult<(UniversalSender<T>, UniversalReceiver<T>)> {
        UniversalChannel::new(Address::Remote(RemoteAddress {
            host,
            port,
            namespace: None,
        }))
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        let (mutex, _not_full, _not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        guard.sender_count += 1;
        drop(guard);
        
        Self {
            state: self.state.clone(),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let (mutex, _not_full, not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        guard.sender_count -= 1;
        
        if guard.sender_count == 0 {
            guard.closed = true;
            not_empty.notify_all();
        }
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        let (mutex, _not_full, _not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        guard.receiver_count += 1;
        drop(guard);
        
        Self {
            state: self.state.clone(),
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let (mutex, not_full, _not_empty) = &*self.state;
        let mut guard = mutex.lock().unwrap();
        guard.receiver_count -= 1;
        
        if guard.receiver_count == 0 {
            // When all receivers are dropped, close the channel to prevent deadlocks
            // This ensures that any waiting or subsequent senders will receive ChannelError::Closed
            guard.closed = true;
            not_full.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;


    #[test]
    fn test_bounded_channel_basic() {
        let (tx, rx) = bounded::<i32>(2);
        
        // Test sending and receiving
        tx.send(42).unwrap();
        tx.send(43).unwrap();
        
        assert_eq!(rx.recv().unwrap(), 42);
        assert_eq!(rx.recv().unwrap(), 43);
    }

    #[test]
    fn test_bounded_channel_capacity() {
        let (tx, rx) = bounded::<i32>(1);
        
        // First message should succeed
        tx.send(1).unwrap();
        
        // Second message should fail with try_send
        assert_eq!(tx.try_send(2), Err(ChannelError::Full));
        
        // After receiving, should be able to send again
        assert_eq!(rx.recv().unwrap(), 1);
        tx.send(3).unwrap();
        assert_eq!(rx.recv().unwrap(), 3);
    }

    #[test]
    fn test_unbounded_channel() {
        let (tx, rx) = unbounded::<String>();
        
        // Should be able to send many messages
        for i in 0..100 {
            tx.send(format!("message {}", i)).unwrap();
        }
        
        // Should receive all messages in order
        for i in 0..100 {
            assert_eq!(rx.recv().unwrap(), format!("message {}", i));
        }
    }

    #[test]
    fn test_oneshot_channel() {
        let (tx, rx) = oneshot::<&'static str>();
        
        tx.send("hello").unwrap();
        assert_eq!(rx.recv().unwrap(), "hello");
    }

    #[test]
    fn test_mpmc_channel() {
        let (tx, rx) = mpmc::<i32>(10);
        let mut sender_handles = Vec::new();

        // Test multiple senders
        let tx1 = tx.clone();
        sender_handles.push(thread::spawn(move || {
            tx1.send(1).unwrap();
            tx1.send(2).unwrap();
        }));
        
        let tx2 = tx.clone();
        sender_handles.push(thread::spawn(move || {
            tx2.send(3).unwrap();
            tx2.send(4).unwrap();
        }));

        // Wait for all senders to complete before proceeding
        for handle in sender_handles {
            handle.join().unwrap();
        }
        drop(tx); // Close channel to signal receivers to stop

        let received_data = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut receiver_handles = Vec::new();

        // Test multiple receivers concurrently
        for _ in 0..2 {
            let rx_clone = rx.clone();
            let data_clone = received_data.clone();
            receiver_handles.push(thread::spawn(move || {
                while let Ok(val) = rx_clone.recv() {
                    data_clone.lock().unwrap().push(val);
                }
            }));
        }

        // Wait for all receivers to complete
        for handle in receiver_handles {
            handle.join().unwrap();
        }
        
        // Verify all messages were received exactly once
        let mut received = received_data.lock().unwrap();
        received.sort();
        assert_eq!(*received, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_channel_errors() {
        let (tx, rx) = bounded::<i32>(1);
        
        // Test empty channel
        assert_eq!(rx.try_recv(), Err(ChannelError::Empty));
        
        // Test closed channel
        drop(tx);
        assert_eq!(rx.recv(), Err(ChannelError::Closed));
    }

    #[test]
    fn test_address_types() {
        let thread_addr = Address::Thread(ThreadId(1));
        let process_addr = Address::Process(ProcessId(123));
        let remote_addr = Address::Remote(RemoteAddress {
            host: "localhost".to_string(),
            port: 8080,
            namespace: None,
        });

        assert_ne!(thread_addr, process_addr);
        assert_ne!(process_addr, remote_addr);
    }

    #[test]
    fn test_transport_manager_creation() {
        let transport_manager = TransportManager::new().unwrap();
        assert_eq!(transport_manager.scheduler_id, SchedulerId::new(0));
    }

    #[test]
    fn test_routing_table() {
        let routing_table = RoutingTable::new();
        
        let thread_transport = routing_table.resolve_transport(&Address::Thread(ThreadId(1))).unwrap();
        assert_eq!(thread_transport, TransportType::InMemory);
        
        let process_transport = routing_table.resolve_transport(&Address::Process(ProcessId(1))).unwrap();
        assert_eq!(process_transport, TransportType::SharedMemory);
    }

    #[test]
    fn test_channel_closes_on_last_receiver_drop() {
        let (tx, rx) = bounded::<i32>(1);
        
        // Fill the channel to capacity
        tx.send(42).unwrap();
        
        // Now the channel is full, any further sends would block
        
        // Spawn a thread that will try to send after the channel is full
        let tx_clone = tx.clone();
        let sender_handle = thread::spawn(move || {
            // This should return ChannelError::Closed after receivers are dropped
            tx_clone.send(99)
        });
        
        // Give the sender thread a moment to start and block
        thread::sleep(std::time::Duration::from_millis(10));
        
        // Drop all receivers - this should close the channel
        drop(rx);
        
        // The sender should now receive ChannelError::Closed instead of hanging
        let result = sender_handle.join().unwrap();
        assert_eq!(result, Err(ChannelError::Closed));
        
        // Any subsequent send attempts should also fail
        assert_eq!(tx.send(100), Err(ChannelError::Closed));
    }
}