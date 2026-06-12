//! Executor trait and implementations.
//!
//! This module provides the core executor abstraction for the Moirai runtime.
//! It defines traits for task spawning, management, and lifecycle control.

pub mod builder;
pub mod config;
pub mod control;
pub mod manager;
pub mod plugin;
pub mod spawner;

pub use builder::ExecutorBuilder;
pub use config::{CleanupConfig, ExecutorConfig, MemoryConfig, PreemptionConfig};
pub use control::ExecutorControl;
pub use manager::{TaskManager, TaskStats, TaskStatus};
pub use plugin::ExecutorPlugin;
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

// Helper function to get number of CPUs
pub(crate) fn num_cpus() -> usize {
    #[cfg(feature = "std")]
    {
        std::thread::available_parallelism().map_or(1, |parallelism| parallelism.get())
    }
    #[cfg(not(feature = "std"))]
    {
        4 // Reasonable default for no_std
    }
}
