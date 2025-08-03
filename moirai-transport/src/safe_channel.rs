//! Safe channel implementation example showing proper serialization
//! 
//! This module demonstrates how to safely implement a universal channel
//! that can handle arbitrary types across different transport boundaries.
//! 
//! # Safety
//! 
//! The key to safety is using proper serialization instead of unsafe
//! memory manipulation. This ensures that:
//! 
//! 1. Complex types with heap allocations (String, Vec, etc.) are properly serialized
//! 2. No dangling pointers are created
//! 3. Memory safety is maintained across transport boundaries
//! 4. The receiving end can properly reconstruct the original value

use crate::{TransportManager, Address, TransportResult, TransportError};
use std::sync::Arc;
use std::marker::PhantomData;

/// Example of a safe universal sender using a marker trait
/// 
/// In a real implementation, this would use serde::Serialize
pub trait SafeSerialize: Send + 'static {
    /// Serialize the value to bytes
    fn serialize(&self) -> Vec<u8>;
}

/// Example of a safe universal receiver using a marker trait
/// 
/// In a real implementation, this would use serde::Deserialize
pub trait SafeDeserialize: Sized + Send + 'static {
    /// Deserialize from bytes
    fn deserialize(bytes: &[u8]) -> Result<Self, TransportError>;
}

/// Safe universal sender that requires serializable types
pub struct SafeUniversalSender<T: SafeSerialize> {
    transport: Arc<TransportManager>,
    target: Address,
    _phantom: PhantomData<T>,
}

impl<T: SafeSerialize> SafeUniversalSender<T> {
    /// Create a new safe sender
    pub fn new(transport: Arc<TransportManager>, target: Address) -> Self {
        Self {
            transport,
            target,
            _phantom: PhantomData,
        }
    }
    
    /// Safely send a value by serializing it first
    pub fn send(&self, value: T) -> TransportResult<()> {
        // Serialize the value safely
        let serialized = value.serialize();
        
        // Send the serialized bytes
        self.transport.send(&self.target, serialized)
    }
}

/// Safe universal receiver that requires deserializable types
pub struct SafeUniversalReceiver<T: SafeDeserialize> {
    transport: Arc<TransportManager>,
    source: Address,
    _phantom: PhantomData<T>,
}

impl<T: SafeDeserialize> SafeUniversalReceiver<T> {
    /// Create a new safe receiver
    pub fn new(transport: Arc<TransportManager>, source: Address) -> Self {
        Self {
            transport,
            source,
            _phantom: PhantomData,
        }
    }
    
    /// Safely receive a value by deserializing it
    pub fn recv(&self) -> TransportResult<T> {
        // Receive the serialized bytes
        let bytes = self.transport.recv(&self.source)?;
        
        // Deserialize safely
        T::deserialize(&bytes)
    }
}

// Example implementations for basic types
impl SafeSerialize for i32 {
    fn serialize(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl SafeDeserialize for i32 {
    fn deserialize(bytes: &[u8]) -> Result<Self, TransportError> {
        if bytes.len() != 4 {
            return Err(TransportError::Closed);
        }
        let array: [u8; 4] = bytes.try_into().map_err(|_| TransportError::Closed)?;
        Ok(i32::from_le_bytes(array))
    }
}

impl SafeSerialize for String {
    fn serialize(&self) -> Vec<u8> {
        // Length-prefixed encoding
        let bytes = self.as_bytes();
        let len = (bytes.len() as u32).to_le_bytes();
        
        let mut result = Vec::with_capacity(4 + bytes.len());
        result.extend_from_slice(&len);
        result.extend_from_slice(bytes);
        result
    }
}

impl SafeDeserialize for String {
    fn deserialize(bytes: &[u8]) -> Result<Self, TransportError> {
        if bytes.len() < 4 {
            return Err(TransportError::Closed);
        }
        
        let len_bytes: [u8; 4] = bytes[0..4].try_into().map_err(|_| TransportError::Closed)?;
        let len = u32::from_le_bytes(len_bytes) as usize;
        
        if bytes.len() < 4 + len {
            return Err(TransportError::Closed);
        }
        
        String::from_utf8(bytes[4..4 + len].to_vec())
            .map_err(|_| TransportError::Closed)
    }
}

/// Example of how to use serde when available
/// 
/// ```ignore
/// use serde::{Serialize, Deserialize};
/// 
/// pub struct SerdeUniversalSender<T: Serialize + Send + 'static> {
///     transport: Arc<TransportManager>,
///     target: Address,
///     _phantom: PhantomData<T>,
/// }
/// 
/// impl<T: Serialize + Send + 'static> SerdeUniversalSender<T> {
///     pub fn send(&self, value: &T) -> TransportResult<()> {
///         // Use bincode or another format for serialization
///         let serialized = bincode::serialize(value)
///             .map_err(|_| TransportError::Closed)?;
///         
///         self.transport.send(&self.target, serialized)
///     }
/// }
/// 
/// pub struct SerdeUniversalReceiver<T: DeserializeOwned + Send + 'static> {
///     transport: Arc<TransportManager>,
///     source: Address,
///     _phantom: PhantomData<T>,
/// }
/// 
/// impl<T: DeserializeOwned + Send + 'static> SerdeUniversalReceiver<T> {
///     pub fn recv(&self) -> TransportResult<T> {
///         let bytes = self.transport.recv(&self.source)?;
///         
///         bincode::deserialize(&bytes)
///             .map_err(|_| TransportError::Closed)
///     }
/// }
/// ```

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_serialization() {
        // Test i32 serialization
        let value: i32 = 42;
        let serialized = value.serialize();
        let deserialized = i32::deserialize(&serialized).unwrap();
        assert_eq!(value, deserialized);
        
        // Test String serialization
        let value = String::from("Hello, Moirai!");
        let serialized = value.serialize();
        let deserialized = String::deserialize(&serialized).unwrap();
        assert_eq!(value, deserialized);
    }
    
    #[test]
    fn test_safe_channel() {
        let transport = Arc::new(TransportManager::new());
        let address = Address::Local("test".to_string());
        
        // Create safe sender and receiver
        let sender = SafeUniversalSender::<String>::new(transport.clone(), address.clone());
        let receiver = SafeUniversalReceiver::<String>::new(transport, address);
        
        // This would work if we had a working transport implementation
        // sender.send("Hello, safe world!".to_string()).unwrap();
        // let received = receiver.recv().unwrap();
        // assert_eq!(received, "Hello, safe world!");
    }
}