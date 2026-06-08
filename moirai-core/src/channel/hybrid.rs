//! Zero-copy hybrid channel for async/sync interop.
//!
//! Uses a lock-free ring buffer with memory barriers to ensure safe
//! zero-copy communication between async and sync contexts.

use super::error::{ChannelError, Result};
use crate::communication::RingBuffer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Zero-copy hybrid channel for async/sync interop
///
/// This channel uses a lock-free ring buffer with memory barriers
/// to ensure safe zero-copy communication between async and sync contexts.
pub struct HybridChannel<T> {
    ring: Arc<RingBuffer<T>>,
    async_notifier: Arc<AtomicBool>,
    sync_notifier: Arc<AtomicBool>,
    /// Parking mechanism for efficient blocking
    parker: Arc<Mutex<Vec<std::thread::Thread>>>,
}

impl<T: Send> HybridChannel<T> {
    /// Create a new hybrid channel with specified capacity
    pub fn new(capacity: usize) -> (HybridSender<T>, HybridReceiver<T>) {
        let channel = Self {
            ring: Arc::new(RingBuffer::new(capacity)),
            async_notifier: Arc::new(AtomicBool::new(false)),
            sync_notifier: Arc::new(AtomicBool::new(false)),
            parker: Arc::new(Mutex::new(Vec::new())),
        };

        channel.split()
    }

    /// Split the channel into sender and receiver halves
    fn split(self) -> (HybridSender<T>, HybridReceiver<T>) {
        let sender = HybridSender {
            ring: self.ring.clone(),
            async_notifier: self.async_notifier.clone(),
            sync_notifier: self.sync_notifier.clone(),
            parker: self.parker.clone(),
        };

        let receiver = HybridReceiver {
            ring: self.ring,
            async_notifier: self.async_notifier,
            sync_notifier: self.sync_notifier,
            parker: self.parker,
        };

        (sender, receiver)
    }

    /// Get the capacity of the channel
    pub fn capacity(&self) -> usize {
        self.ring.capacity()
    }

    /// Check if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Check if the channel is full
    pub fn is_full(&self) -> bool {
        self.ring.is_full()
    }

    /// Get the number of items currently in the channel
    pub fn len(&self) -> usize {
        self.ring.len()
    }
}

// ---------------------------------------------------------------------------
// HybridSender
// ---------------------------------------------------------------------------

/// Sender half of hybrid channel
pub struct HybridSender<T> {
    ring: Arc<RingBuffer<T>>,
    async_notifier: Arc<AtomicBool>,
    sync_notifier: Arc<AtomicBool>,
    parker: Arc<Mutex<Vec<std::thread::Thread>>>,
}

impl<T: Send> HybridSender<T> {
    /// Send value with zero-copy when possible
    pub fn send(&self, value: T) -> Result<()> {
        self.ring
            .try_produce(value)
            .map_err(|_| ChannelError::Full)?;

        // Notify both async and sync waiters
        self.async_notifier.store(true, Ordering::Release);
        self.sync_notifier.store(true, Ordering::Release);

        // Unpark any waiting threads
        if let Ok(mut parked) = self.parker.lock() {
            for thread in parked.drain(..) {
                thread.unpark();
            }
        }

        Ok(())
    }

    /// Try to send without blocking
    pub fn try_send(&self, value: T) -> Result<()> {
        self.send(value)
    }

    /// Send with timeout
    pub fn send_timeout(&self, mut value: T, timeout: std::time::Duration) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            match self.ring.try_produce(value) {
                Ok(()) => {
                    // Notify waiters
                    self.async_notifier.store(true, Ordering::Release);
                    self.sync_notifier.store(true, Ordering::Release);

                    // Unpark any waiting threads
                    if let Ok(mut parked) = self.parker.lock() {
                        for thread in parked.drain(..) {
                            thread.unpark();
                        }
                    }

                    return Ok(());
                }
                Err(v) => {
                    value = v; // Get the value back
                    if start.elapsed() >= timeout {
                        return Err(ChannelError::Full);
                    }
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Check if the sender can send without blocking
    pub fn can_send(&self) -> bool {
        !self.ring.is_full()
    }

    /// Get the number of items that can be sent without blocking
    pub fn available_capacity(&self) -> usize {
        self.ring.capacity() - self.ring.len()
    }
}

impl<T> Clone for HybridSender<T> {
    fn clone(&self) -> Self {
        Self {
            ring: self.ring.clone(),
            async_notifier: self.async_notifier.clone(),
            sync_notifier: self.sync_notifier.clone(),
            parker: self.parker.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// HybridReceiver
// ---------------------------------------------------------------------------

/// Receiver half of hybrid channel
pub struct HybridReceiver<T> {
    ring: Arc<RingBuffer<T>>,
    #[allow(dead_code)]
    async_notifier: Arc<AtomicBool>,
    #[allow(dead_code)]
    sync_notifier: Arc<AtomicBool>,
    parker: Arc<Mutex<Vec<std::thread::Thread>>>,
}

impl<T: Send> HybridReceiver<T> {
    /// Receive value with zero-copy
    pub fn recv(&self) -> Result<T> {
        // Fast path: try to receive without blocking
        if let Some(value) = self.ring.try_consume() {
            return Ok(value);
        }

        // Slow path: park the thread and wait for notification
        let current_thread = std::thread::current();

        loop {
            // Register this thread for unparking
            if let Ok(mut parked) = self.parker.lock() {
                parked.push(current_thread.clone());
            }

            // Check again after registering (to avoid race)
            if let Some(value) = self.ring.try_consume() {
                // Remove ourselves from the parker list
                if let Ok(mut parked) = self.parker.lock() {
                    parked.retain(|t| !t.id().eq(&current_thread.id()));
                }
                return Ok(value);
            }

            // Park until unparked by sender
            std::thread::park();

            // Try again after being unparked
            if let Some(value) = self.ring.try_consume() {
                return Ok(value);
            }
        }
    }

    /// Try to receive without blocking
    pub fn try_recv(&self) -> Result<T> {
        self.ring.try_consume().ok_or(ChannelError::Empty)
    }

    /// Receive with timeout
    pub fn recv_timeout(&self, timeout: std::time::Duration) -> Result<T> {
        let start = std::time::Instant::now();

        loop {
            match self.try_recv() {
                Ok(value) => return Ok(value),
                Err(ChannelError::Empty) => {
                    if start.elapsed() >= timeout {
                        return Err(ChannelError::Empty);
                    }

                    // Register for wake-up before checking again
                    let current_thread = std::thread::current();
                    if let Ok(mut parked) = self.parker.lock() {
                        parked.push(current_thread.clone());
                    }

                    // Park for the remaining timeout budget.
                    if let Some(remaining) = timeout.checked_sub(start.elapsed()) {
                        std::thread::park_timeout(remaining);
                    }

                    // Remove from parker list
                    if let Ok(mut parked) = self.parker.lock() {
                        parked.retain(|t| !t.id().eq(&current_thread.id()));
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Check if there are messages available
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Get the number of messages available
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Drain all available messages
    pub fn drain(&self) -> Vec<T> {
        let mut messages = Vec::new();
        while let Ok(msg) = self.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// Async receive for use in async contexts
    #[cfg(feature = "std")]
    pub async fn recv_async(&self) -> Result<T> {
        loop {
            match self.ring.try_consume() {
                Some(value) => return Ok(value),
                None => {
                    // For now, busy wait with yield hints
                    // In a real implementation, we'd integrate with the async runtime
                    std::hint::spin_loop();
                    if self.async_notifier.load(Ordering::Acquire) {
                        self.async_notifier.store(false, Ordering::Release);
                    }
                }
            }
        }
    }
}
