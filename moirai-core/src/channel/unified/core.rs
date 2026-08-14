//! Core implementation of the adaptive UnifiedChannel.

#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::channel::config::ChannelConfig;
use crate::channel::error::ChannelError;
use crate::channel::stats::{ChannelStatistics, ChannelStats};
use crate::memory::UnifiedRingBuffer;

/// Unified channel that adapts to different usage patterns
pub struct UnifiedChannel<T> {
    /// Primary ring buffer for fast path operations
    pub(crate) ring_buffer: UnifiedRingBuffer<T>,
    /// Unbounded/pooled lock-free fallback overflow queue
    pub(crate) overflow_queue: Mutex<VecDeque<T>>,
    /// Advisory count of elements in the overflow queue, mutated under
    /// `overflow_queue`'s lock and read lock-free on the fast path. It is only a
    /// hint: a stale read can cause `recv`/`send` to skip an opportunistic drain
    /// into the ring buffer, but never loses a message — `recv` pops directly from
    /// `overflow_queue` whenever the ring is empty, so a missed drain is at worst
    /// a fast-path optimization miss reconciled by the next locked operation.
    pub(crate) overflow_count: AtomicUsize,
    /// Configuration parameters
    pub(crate) config: ChannelConfig,
    /// Channel state flags
    pub(crate) is_closed: AtomicBool,
    /// Statistics for adaptive behavior
    pub(crate) stats: ChannelStats,
}

impl<T> UnifiedChannel<T> {
    /// Create a new unified channel with given configuration
    pub fn new(config: ChannelConfig) -> Result<Self, ChannelError> {
        let ring_buffer =
            UnifiedRingBuffer::new(config.capacity).ok_or(ChannelError::InvalidConfig)?;

        Ok(Self {
            ring_buffer,
            overflow_queue: Mutex::new(VecDeque::new()),
            overflow_count: AtomicUsize::new(0),
            config,
            is_closed: AtomicBool::new(false),
            stats: ChannelStats::new(),
        })
    }

    /// Create with default configuration
    pub fn with_capacity(capacity: usize) -> Result<Self, ChannelError> {
        let config = ChannelConfig {
            capacity,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Send a message with automatic overflow handling (non-blocking).
    ///
    /// Returns `Err(ChannelError::Full)` when both the ring buffer and the
    /// overflow pool are full (or pooling is disabled), and `Err(Closed)` when the
    /// channel is closed. In both error cases the `message` is **consumed** (dropped):
    /// the failure is surfaced explicitly, but the value cannot be recovered. A
    /// caller that needs the value back to retry must use [`Self::try_send`], which
    /// returns it in the error. This delegates to `try_send` (single SSOT for the
    /// send path) rather than duplicating the overflow logic.
    pub fn send(&self, message: T) -> Result<(), ChannelError> {
        self.try_send(message).map_err(|(_message, err)| err)
    }

    /// Try to send without blocking, returning the message back on failure.
    pub fn try_send(&self, mut message: T) -> Result<(), (T, ChannelError)> {
        if self.is_closed.load(Ordering::Acquire) {
            return Err((message, ChannelError::Closed));
        }

        // Fast path: check if overflow queue is empty and push to ring buffer
        if self.overflow_count.load(Ordering::Acquire) == 0 {
            match self.ring_buffer.try_push(message) {
                Ok(()) => {
                    self.stats.record_send();
                    return Ok(());
                }
                Err(msg) => {
                    message = msg;
                }
            }
        }

        // Fallback: lock overflow queue
        let mut overflow = self.overflow_queue.lock().unwrap();
        self.drain_locked(&mut overflow);

        if overflow.is_empty() {
            match self.ring_buffer.try_push(message) {
                Ok(()) => {
                    self.stats.record_send();
                    return Ok(());
                }
                Err(msg) => {
                    message = msg;
                }
            }
        }

        if self.config.enable_pooling && overflow.len() < self.config.max_pool_size {
            overflow.push_back(message);
            self.overflow_count.store(overflow.len(), Ordering::Release);
            self.stats.record_send();
            self.stats.record_overflow();
            self.stats.record_contention();
            Ok(())
        } else {
            Err((message, ChannelError::Full))
        }
    }

    /// Receive a message (non-blocking).
    ///
    /// Returns `Err(Empty)` when no message is currently available and
    /// `Err(Closed)` once the channel is closed and drained; this channel has
    /// no blocking receive path.
    pub fn recv(&self) -> Result<T, ChannelError> {
        // Try fast path first: pop from ring buffer
        if let Some(message) = self.ring_buffer.try_pop() {
            self.stats.record_receive();
            // If overflow queue contains items, trigger lazy drain under lock
            if self.overflow_count.load(Ordering::Acquire) > 0 {
                if let Ok(mut overflow) = self.overflow_queue.try_lock() {
                    self.drain_locked(&mut overflow);
                }
            }
            return Ok(message);
        }

        // If ring buffer is empty but overflow queue is not, pop from overflow queue
        if self.overflow_count.load(Ordering::Acquire) > 0 {
            let mut overflow = self.overflow_queue.lock().unwrap();
            if let Some(message) = overflow.pop_front() {
                self.overflow_count.store(overflow.len(), Ordering::Release);
                self.stats.record_receive();
                self.drain_locked(&mut overflow);
                return Ok(message);
            }
        }

        // Check if channel is closed and empty
        if self.is_closed.load(Ordering::Acquire) && self.is_empty() {
            return Err(ChannelError::Closed);
        }

        Err(ChannelError::Empty)
    }

    /// Send multiple messages in batch (if batching enabled)
    pub fn send_batch(&self, messages: Vec<T>) -> Result<usize, ChannelError> {
        if !self.config.enable_batching {
            return Err(ChannelError::InvalidConfig);
        }

        if self.is_closed.load(Ordering::Acquire) {
            return Err(ChannelError::Closed);
        }

        let mut sent_count = 0;
        for message in messages {
            match self.send(message) {
                Ok(()) => sent_count += 1,
                Err(ChannelError::Full) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(sent_count)
    }

    /// Receive multiple messages in batch
    pub fn recv_batch(&self, max_count: usize) -> Vec<T> {
        let mut messages = Vec::with_capacity(max_count.min(self.config.batch_size));

        for _ in 0..max_count {
            match self.recv() {
                Ok(message) => messages.push(message),
                Err(_) => break,
            }
        }

        messages
    }

    /// Close the channel
    pub fn close(&self) {
        self.is_closed.store(true, Ordering::Release);
    }

    /// Check if channel is closed
    pub fn is_closed(&self) -> bool {
        self.is_closed.load(Ordering::Acquire)
    }

    /// Get current buffer length
    pub fn len(&self) -> usize {
        self.ring_buffer.len() + self.overflow_count.load(Ordering::Acquire)
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.ring_buffer.is_empty() && self.overflow_count.load(Ordering::Acquire) == 0
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.ring_buffer.capacity() + self.config.max_pool_size
    }

    /// Get channel statistics for monitoring
    pub fn stats(&self) -> ChannelStatistics {
        ChannelStatistics {
            messages_sent: self.stats.messages_sent.load(Ordering::Relaxed),
            messages_received: self.stats.messages_received.load(Ordering::Relaxed),
            overflow_events: self.stats.overflow_events.load(Ordering::Relaxed),
            contention_count: self.stats.contention_count.load(Ordering::Relaxed),
            current_length: self.len(),
            capacity: self.capacity(),
            throughput_ratio: self.stats.get_throughput_ratio(),
        }
    }

    /// Drain as many overflow items into the ring buffer as possible
    fn drain_locked(&self, overflow: &mut VecDeque<T>) {
        while !overflow.is_empty() {
            let item = overflow.pop_front().unwrap();
            match self.ring_buffer.try_push(item) {
                Ok(()) => {}
                Err(item) => {
                    overflow.push_front(item);
                    break;
                }
            }
        }
        self.overflow_count.store(overflow.len(), Ordering::Release);
    }
}
