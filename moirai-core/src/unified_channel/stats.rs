//! Statistics types for unified channels.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Performance statistics for adaptive channel behavior
#[derive(Debug)]
pub(crate) struct ChannelStats {
    /// Total messages sent
    pub(crate) messages_sent: AtomicUsize,
    /// Total messages received
    pub(crate) messages_received: AtomicUsize,
    /// Number of times overflow pool was used
    pub(crate) overflow_events: AtomicUsize,
    /// Contention counter for adaptive behavior
    pub(crate) contention_count: AtomicUsize,
}

impl ChannelStats {
    pub(crate) fn new() -> Self {
        Self {
            messages_sent: AtomicUsize::new(0),
            messages_received: AtomicUsize::new(0),
            overflow_events: AtomicUsize::new(0),
            contention_count: AtomicUsize::new(0),
        }
    }

    /// Record a successful send operation
    pub(crate) fn record_send(&self) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful receive operation
    pub(crate) fn record_receive(&self) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an overflow event
    pub(crate) fn record_overflow(&self) {
        self.overflow_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Record contention
    pub(crate) fn record_contention(&self) {
        self.contention_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get send/receive ratio for adaptive behavior
    // justification: message counts converted to f64 for a ratio; precision loss
    // only occurs past 2^52 messages, unreachable for these counters in practice.
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn get_throughput_ratio(&self) -> f64 {
        let sent = self.messages_sent.load(Ordering::Relaxed);
        let received = self.messages_received.load(Ordering::Relaxed);

        if received == 0 {
            return f64::INFINITY;
        }

        sent as f64 / received as f64
    }
}

/// Statistics snapshot for monitoring channel performance
#[derive(Debug, Clone)]
pub struct ChannelStatistics {
    /// Total number of messages successfully sent through the channel
    pub messages_sent: usize,
    /// Total number of messages successfully received from the channel
    pub messages_received: usize,
    /// Number of times the channel had to use overflow handling
    pub overflow_events: usize,
    /// Number of contention events detected during operations
    pub contention_count: usize,
    /// Current number of messages in the channel
    pub current_length: usize,
    /// Maximum capacity of the channel
    pub capacity: usize,
    /// Ratio of successful operations to total attempts
    pub throughput_ratio: f64,
}
