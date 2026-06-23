//! Execution contexts for different iterator strategies.

pub mod async_ctx;
pub mod base;
pub mod hybrid;
pub mod parallel;

#[cfg(test)]
mod tests;

pub use async_ctx::AsyncContext;
pub use base::{ExecutionBase, ExecutionContext};
pub use hybrid::{ExecutionStrategy, HybridConfig, HybridContext, PerformanceHistory};
pub use parallel::ParallelContext;
