//! Async executor module for Moirai.
//!
//! This module declares submodules for task executor components following SLAP/SRP.

pub mod core;
pub mod handle;
pub(super) mod result_slot;
pub mod stats;
pub(super) mod task;
pub(super) mod waker;

pub use core::AsyncExecutor;
pub use handle::AsyncHandle;
pub use stats::ExecutorStats;
