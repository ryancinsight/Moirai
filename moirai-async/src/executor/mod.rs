//! Async executor module for Moirai.
//!
//! This module declares submodules for task executor components following SLAP/SRP.

/// Executor core: spawn, run, and shutdown of async tasks.
pub mod core;
/// Join handles returned by spawn operations.
pub mod handle;
pub(super) mod result_slot;
/// Executor runtime statistics.
pub mod stats;
pub(super) mod task;
pub(super) mod waker;

pub use core::AsyncExecutor;
pub use handle::AsyncHandle;
pub use stats::ExecutorStats;
