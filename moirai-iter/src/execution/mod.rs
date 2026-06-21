//! Execution contexts for different iterator strategies.

pub mod base;
pub mod parallel;
pub mod async_ctx;
pub mod hybrid;

#[cfg(test)]
mod tests;

pub use base::{ExecutionBase, ExecutionContext};
pub use parallel::ParallelContext;
pub use async_ctx::AsyncContext;
pub use hybrid::{HybridContext, HybridConfig, PerformanceHistory, ExecutionStrategy};
