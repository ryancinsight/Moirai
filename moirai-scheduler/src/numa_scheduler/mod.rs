//! NUMA-aware work stealing scheduler.
//!
//! This module provides a scheduler that understands NUMA topology and optimizes
//! work distribution to minimize memory latency and maximize cache efficiency.

pub mod backoff;
pub mod queue;
pub mod scheduler;
pub mod topology;

#[cfg(test)]
mod tests;

pub use backoff::AdaptiveBackoff;
pub use queue::StealStatistics;
pub use scheduler::{NumaAwareScheduler, NumaSchedulerError, NumaSchedulerStats};
pub use topology::{CacheLevel, CpuTopology, NumaNode};
