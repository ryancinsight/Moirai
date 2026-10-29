//! Executor trait and implementations.
//!
//! This module provides the core executor abstraction for the Moirai runtime.
//! It defines traits for task spawning, management, and lifecycle control.
//! Without `std`, only configuration and worker-placement vocabulary remains;
//! spawning, lifecycle, task completion, clocks, and host topology require
//! operating-system capabilities.

#[cfg(feature = "std")]
pub mod builder;
pub mod config;
#[cfg(feature = "std")]
pub mod control;
#[cfg(feature = "std")]
pub mod manager;
#[cfg(feature = "std")]
pub mod parallelism;
pub mod placement;
#[cfg(feature = "std")]
pub mod spawner;

#[cfg(feature = "std")]
pub use builder::ExecutorBuilder;
pub use config::{CleanupConfig, ExecutorConfig, MemoryConfig, PreemptionConfig};
#[cfg(feature = "std")]
pub use control::ExecutorControl;
#[cfg(feature = "std")]
pub use manager::{TaskManager, TaskStats, TaskStatus};
#[cfg(feature = "std")]
pub use parallelism::logical_parallelism;
pub use placement::WorkerPlacement;
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
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
#[cfg(all(feature = "std", not(feature = "metrics")))]
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats;

/// Executor statistics with full metrics
#[cfg(all(feature = "std", feature = "metrics"))]
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    /// Number of tasks executed
    pub tasks_executed: u64,
    /// Number of tasks in queue
    pub tasks_queued: usize,
    /// Average task execution time
    pub avg_execution_time_ns: u64,
}
