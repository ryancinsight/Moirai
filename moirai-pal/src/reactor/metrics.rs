use std::sync::atomic::AtomicU64;
use std::sync::OnceLock;
use std::time::Instant;

/// Performance metrics for the reactor
#[derive(Debug, Default)]
pub struct ReactorMetrics {
    /// Total events processed
    pub events_processed: AtomicU64,
    /// Total tasks executed
    pub tasks_executed: AtomicU64,
    /// Average event processing time (nanoseconds)
    pub avg_event_time_ns: AtomicU64,
    /// Peak number of registered file descriptors
    pub peak_fd_count: AtomicU64,
    /// Reactor uptime
    pub start_time: OnceLock<Instant>,
}
