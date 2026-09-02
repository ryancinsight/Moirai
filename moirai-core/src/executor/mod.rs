//! Executor trait and implementations.
//!
//! This module provides the core executor abstraction for the Moirai runtime.
//! It defines traits for task spawning, management, and lifecycle control.

pub mod builder;
pub mod config;
pub mod control;
pub mod manager;
pub mod spawner;

pub use builder::ExecutorBuilder;
pub use config::{CleanupConfig, ExecutorConfig, MemoryConfig, PreemptionConfig};
pub use control::ExecutorControl;
pub use manager::{TaskManager, TaskStats, TaskStatus};
pub use spawner::TaskSpawner;

/// Combined executor trait with all capabilities.
///
/// This trait combines all executor capabilities into a single interface
/// for convenience while maintaining the segregated design internally.
///
/// # Design Philosophy
/// - Composition over inheritance
/// - Single interface for complete functionality
/// - Maintains internal separation of concerns
/// - Enables easy mocking and testing
pub trait Executor: TaskSpawner + TaskManager + ExecutorControl {
    /// Get comprehensive executor statistics.
    ///
    /// # Behavior Guarantees
    /// - Returns current snapshot of all executor metrics
    /// - Statistics are eventually consistent
    /// - Available only when metrics feature is enabled
    /// - Includes worker, queue, memory, and task statistics
    ///
    /// # Performance Characteristics
    /// - Collection overhead: < 1μs for full statistics
    /// - Memory: ~1KB for complete statistics snapshot
    /// - Thread safety: Atomic operations for consistency
    #[cfg(feature = "metrics")]
    fn stats(&self) -> ExecutorStats;
}

/// Executor statistics (basic implementation when metrics feature is disabled)
#[cfg(not(feature = "metrics"))]
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats;

/// Executor statistics with full metrics
#[cfg(feature = "metrics")]
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    /// Number of tasks executed
    pub tasks_executed: u64,
    /// Number of tasks in queue
    pub tasks_queued: usize,
    /// Average task execution time
    pub avg_execution_time_ns: u64,
}

/// Logical processors this process should parallelize across.
///
/// Derived once and cached for the process lifetime. The topology cannot
/// change while the process runs, and deriving it is not cheap: a
/// `CpuTopology::detect()` call measures 9,935 ns and 77 allocations totalling
/// 16,480 bytes on a 24-processor host, because it materializes the whole
/// NUMA and cache-level description to read one count. Callers that need a
/// worker count per operation must not pay that, so this is the one place the
/// derivation happens.
///
/// `themis` reports the machine's logical processors; `available_parallelism`
/// is the fallback when no topology is available.
#[must_use]
pub fn logical_parallelism() -> usize {
    #[cfg(feature = "std")]
    {
        static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CACHED.get_or_init(detect_logical_parallelism)
    }
    #[cfg(not(feature = "std"))]
    {
        detect_logical_parallelism()
    }
}

fn detect_logical_parallelism() -> usize {
    #[cfg(feature = "std")]
    {
        themis::CpuTopology::detect()
            .map(|topology| topology.logical_processors())
            .or_else(|| {
                std::thread::available_parallelism()
                    .ok()
                    .map(std::num::NonZeroUsize::get)
            })
            .unwrap_or(1)
            .max(1)
    }
    #[cfg(not(feature = "std"))]
    {
        4 // Reasonable default for no_std
    }
}
