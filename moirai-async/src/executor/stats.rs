use std::sync::atomic::AtomicU64;

/// Statistics for async executor performance monitoring.
#[derive(Debug, Default)]
pub(super) struct AsyncExecutorStats {
    pub(super) tasks_spawned: AtomicU64,
    pub(super) tasks_completed: AtomicU64,
    pub(super) total_execution_time_ns: AtomicU64,
    /// Currently untracked: no executor path increments this counter, so it
    /// always reads 0. The increment sites belong in the waker/poll paths of
    /// `executor/core.rs`/`executor/waker.rs`; wiring is pending there.
    pub(super) waker_notifications: AtomicU64,
    /// Currently untracked: no executor path increments this counter, so it
    /// always reads 0. The increment sites belong in the reactor dispatch path
    /// of `executor/core.rs`; wiring is pending there.
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
    /// Number of waker notifications sent.
    ///
    /// Currently untracked: the executor does not increment this counter yet,
    /// so it always reads 0 (pending wiring in the executor core/waker paths).
    pub waker_notifications: u64,
    /// Number of I/O operations processed.
    ///
    /// Currently untracked: the executor does not increment this counter yet,
    /// so it always reads 0 (pending wiring in the executor core).
    pub io_operations: u64,
}
