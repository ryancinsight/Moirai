//! NUMA-aware scheduling primitives.
//!
//! Hardware NUMA/cache topology discovery and adaptive backoff — the inputs a
//! scheduler uses for locality-aware victim selection. The runtime scheduler
//! (`moirai_executor`'s `ThreadScheduler`) consumes these; this module holds no
//! scheduler of its own.

pub mod backoff;
pub mod topology;

pub use backoff::AdaptiveBackoff;
pub use topology::{CacheLevel, CpuTopology, NumaNode};
