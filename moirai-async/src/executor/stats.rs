use std::sync::atomic::AtomicU64;

/// Statistics for async executor performance monitoring.
#[derive(Debug, Default)]
pub(super) struct AsyncExecutorStats {
    pub(super) tasks_spawned: AtomicU64,
    pub(super) tasks_completed: AtomicU64,
    pub(super) total_execution_time_ns: AtomicU64,
    pub(super) waker_notifications: AtomicU64,
    pub(super) io_operations: AtomicU64,
    pub(super) tasks_pending: AtomicU64,
}

/// Public statistics structure for monitoring executor performance.
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    /// Total number of tasks spawned
    pub tasks_spawned: u64,
    /// Total number of tasks completed
    pub tasks_completed: u64,
    /// Number of tasks currently pending
    pub tasks_pending: u64,
    /// Total execution time in nanoseconds
    pub total_execution_time_ns: u64,
    /// Number of waker notifications sent
    pub waker_notifications: u64,
    /// Number of I/O operations processed
    pub io_operations: u64,
}
