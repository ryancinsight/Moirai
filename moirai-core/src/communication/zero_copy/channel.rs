//! Zero-copy channel implemented over `MemoryMappedRing`.

use std::sync::Arc;

use super::error::{ZeroCopyError, ZeroCopyResult};
use super::ring::MemoryMappedRing;

/// Zero-copy channel implemented over `MemoryMappedRing`.
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
/// Sends values through the shared ring by pointer move (no clone, no serialization).
pub struct ZeroCopySender<T> {
    pub(super) ring: Arc<MemoryMappedRing<T>>,
}

impl<T> ZeroCopySender<T> {
    /// Sends a value through the channel (non-blocking; returns the value in
    /// `Err((value, ZeroCopyError::Full))` when the ring is full).
    pub fn send(&self, value: T) -> Result<(), (T, ZeroCopyError)> {
        self.ring.send_zero_copy(value)
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
/// Receives values from the shared ring by pointer move (no clone, no serialization).
pub struct ZeroCopyReceiver<T> {
    pub(super) ring: Arc<MemoryMappedRing<T>>,
}

impl<T> ZeroCopyReceiver<T> {
    /// Receive a value with zero-copy semantics (non-blocking; returns
    /// `Err(ZeroCopyError::Empty)` when no value is buffered).
    pub fn recv(&self) -> ZeroCopyResult<T> {
        self.ring.recv_zero_copy()
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
