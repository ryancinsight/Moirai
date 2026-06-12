//! Zero-copy channel implemented over MemoryMappedRing.

use std::sync::Arc;

use super::error::{ZeroCopyError, ZeroCopyResult};
use super::ring::MemoryMappedRing;

/// Zero-copy channel implemented over MemoryMappedRing.
pub struct ZeroCopyChannel<T> {
    _ring: Arc<MemoryMappedRing<T>>,
}

impl<T> ZeroCopyChannel<T> {
    /// Creates a new zero-copy channel pair with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - The maximum number of elements the channel can buffer
    ///
    /// # Returns
    /// A tuple containing the sender and receiver halves of the channel
    pub fn new(capacity: usize) -> ZeroCopyResult<(ZeroCopySender<T>, ZeroCopyReceiver<T>)> {
        let ring = Arc::new(MemoryMappedRing::new(capacity)?);
        Ok((
            ZeroCopySender { ring: ring.clone() },
            ZeroCopyReceiver { ring },
        ))
    }
}

/// Zero-copy sender half of a channel.
///
/// Allows sending values through a memory-mapped ring buffer without copying data.
pub struct ZeroCopySender<T> {
    pub(super) ring: Arc<MemoryMappedRing<T>>,
}

impl<T> ZeroCopySender<T> {
    /// Sends a value through the channel.
    pub fn send(&self, value: T) -> Result<(), (T, ZeroCopyError)> {
        self.ring.send_zero_copy(value)
    }

    /// Attempts to send a value without blocking.
    pub fn try_send(&self, value: T) -> Result<(), (T, ZeroCopyError)> {
        self.ring.try_send(value)
    }

    /// Closes the sender half of the channel.
    pub fn close(&self) {
        self.ring.close();
    }

    /// Returns true if the channel is closed.
    pub fn is_closed(&self) -> bool {
        self.ring.is_closed()
    }
}

impl<T> Clone for ZeroCopySender<T> {
    fn clone(&self) -> Self {
        Self {
            ring: self.ring.clone(),
        }
    }
}

/// Zero-copy receiver half of a channel.
///
/// Allows receiving values from a memory-mapped ring buffer without copying data.
pub struct ZeroCopyReceiver<T> {
    pub(super) ring: Arc<MemoryMappedRing<T>>,
}

impl<T> ZeroCopyReceiver<T> {
    /// Receive a value with zero-copy semantics
    pub fn recv(&self) -> ZeroCopyResult<T> {
        self.ring.recv_zero_copy()
    }

    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> ZeroCopyResult<T> {
        self.ring.try_recv()
    }

    /// Check if the channel is closed
    pub fn is_closed(&self) -> bool {
        self.ring.is_closed()
    }
}

impl<T> Clone for ZeroCopyReceiver<T> {
    fn clone(&self) -> Self {
        Self {
            ring: self.ring.clone(),
        }
    }
}
