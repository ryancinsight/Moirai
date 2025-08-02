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
//! - Integration with Moirai scheduler for optimal performance

pub mod zero_copy;

use moirai_core::{
    TaskId, scheduler::SchedulerId,
    channel::{MpmcSender, MpmcReceiver, mpmc, unbounded, ChannelError}
};
use std::{
    fmt,
    sync::{Arc, Mutex},
    collections::HashMap,
    any::Any,
};

// Re-export core channel types for compatibility
pub use moirai_core::channel::{
    ChannelError as TransportError,
    MpmcSender as Sender,
    MpmcReceiver as Receiver,
};

/// Result type for transport operations
pub type TransportResult<T> = Result<T, TransportError>;

/// Create a bounded channel (re-export for compatibility)
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    mpmc(capacity)
}

/// Create an unbounded channel (re-export for compatibility)
pub fn unbounded_channel<T>() -> (Sender<T>, Receiver<T>) {
    unbounded()
}

/// Address for location-transparent communication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Address {
    /// Local thread address
    Thread(String),
    /// Process-local address
    Process(String),
    /// Remote machine address
    Remote(String, String), // (host, port)
    /// Broadcast to all addresses
    Broadcast,
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Thread(id) => write!(f, "thread://{}", id),
            Self::Process(id) => write!(f, "process://{}", id),
            Self::Remote(host, port) => write!(f, "remote://{}:{}", host, port),
            Self::Broadcast => write!(f, "broadcast://"),
        }
    }
}

/// Transport backend trait for different communication mechanisms
pub trait Transport: Send + Sync {
    /// Send a message to an address
    fn send(&self, addr: &Address, message: Vec<u8>) -> TransportResult<()>;
    
    /// Receive a message from any address
    fn recv(&self) -> TransportResult<(Address, Vec<u8>)>;
    
    /// Try to receive without blocking
    fn try_recv(&self) -> TransportResult<(Address, Vec<u8>)>;
    
    /// Get the local address for this transport
    fn local_address(&self) -> Address;
}

/// In-memory transport for thread-local communication
pub struct InMemoryTransport {
    address: Address,
    inbox: Arc<Mutex<HashMap<Address, MpmcSender<(Address, Vec<u8>)>>>>,
    receiver: MpmcReceiver<(Address, Vec<u8>)>,
}

impl InMemoryTransport {
    /// Create a new in-memory transport
    pub fn new(id: String) -> Self {
        let address = Address::Thread(id);
        let (tx, rx) = mpmc(1000);
        let inbox = Arc::new(Mutex::new(HashMap::new()));
        
        {
            let mut map = inbox.lock().unwrap();
            map.insert(address.clone(), tx);
        }
        
        Self {
            address,
            inbox,
            receiver: rx,
        }
    }
}

impl Transport for InMemoryTransport {
    fn send(&self, addr: &Address, message: Vec<u8>) -> TransportResult<()> {
        let inbox = self.inbox.lock().unwrap();
        
        match addr {
            Address::Broadcast => {
                // Send to all addresses except self
                for (target, sender) in inbox.iter() {
                    if target != &self.address {
                        let _ = sender.try_send((self.address.clone(), message.clone()));
                    }
                }
                Ok(())
            }
            _ => {
                if let Some(sender) = inbox.get(addr) {
                    sender.try_send((self.address.clone(), message))
                } else {
                    Err(TransportError::Closed)
                }
            }
        }
    }
    
    fn recv(&self) -> TransportResult<(Address, Vec<u8>)> {
        self.receiver.recv()
    }
    
    fn try_recv(&self) -> TransportResult<(Address, Vec<u8>)> {
        self.receiver.try_recv()
    }
    
    fn local_address(&self) -> Address {
        self.address.clone()
    }
}

/// Transport manager for coordinating multiple transport backends
pub struct TransportManager {
    transports: Arc<Mutex<HashMap<String, Box<dyn Transport>>>>,
    default_transport: String,
}

impl TransportManager {
    /// Create a new transport manager
    pub fn new() -> Self {
        Self {
            transports: Arc::new(Mutex::new(HashMap::new())),
            default_transport: "inmemory".to_string(),
        }
    }
    
    /// Register a transport backend
    pub fn register(&self, name: String, transport: Box<dyn Transport>) {
        let mut transports = self.transports.lock().unwrap();
        transports.insert(name, transport);
    }
    
    /// Send a message using the appropriate transport
    pub fn send(&self, addr: &Address, message: Vec<u8>) -> TransportResult<()> {
        let transports = self.transports.lock().unwrap();
        
        // Select transport based on address type
        let transport_name = match addr {
            Address::Thread(_) | Address::Process(_) => &self.default_transport,
            Address::Remote(_, _) => "network",
            Address::Broadcast => &self.default_transport,
        };
        
        if let Some(transport) = transports.get(transport_name) {
            transport.send(addr, message)
        } else {
            Err(TransportError::Closed)
        }
    }
}

/// Universal channel that works across different transport boundaries
pub struct UniversalChannel<T: Send + 'static> {
    sender: UniversalSender<T>,
    receiver: UniversalReceiver<T>,
}

/// Sender half of universal channel
pub struct UniversalSender<T: Send + 'static> {
    transport: Arc<TransportManager>,
    target: Address,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + 'static> UniversalSender<T> {
    /// Send a message to the target address
    pub fn send(&self, value: T) -> TransportResult<()> {
        // Serialize the message (simplified - in real impl would use proper serialization)
        let message = unsafe {
            let ptr = Box::into_raw(Box::new(value));
            let bytes = std::slice::from_raw_parts(
                ptr as *const u8,
                std::mem::size_of::<T>()
            ).to_vec();
            let _ = Box::from_raw(ptr); // Prevent leak
            bytes
        };
        
        self.transport.send(&self.target, message)
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

/// Receiver half of universal channel
pub struct UniversalReceiver<T: Send + 'static> {
    transport: Arc<TransportManager>,
    source: Address,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Send + 'static> UniversalReceiver<T> {
    /// Receive a message from the source address
    pub fn recv(&self) -> TransportResult<T> {
        // In a real implementation, this would properly deserialize
        // For now, we'll return an error to indicate unimplemented
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
        write!(f, "{}:{}#{}", self.host, self.port, self.service)
    }
}

/// Network transport implementation (placeholder)
#[cfg(feature = "network")]
pub struct TcpTransport {
    // Implementation would go here
}

#[cfg(feature = "network")]
pub struct UdpTransport {
    // Implementation would go here
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_channel_compatibility() {
        let (tx, rx) = channel::<i32>(10);
        
        assert!(tx.send(42).is_ok());
        assert_eq!(rx.recv().unwrap(), 42);
    }
    
    #[test]
    fn test_in_memory_transport() {
        let transport1 = InMemoryTransport::new("t1".to_string());
        let transport2 = InMemoryTransport::new("t2".to_string());
        
        // Register transports with each other (simplified)
        transport1.inbox.lock().unwrap().insert(
            transport2.local_address(),
            transport1.receiver.clone().into() // This would need proper implementation
        );
        
        // Test would continue...
    }
}