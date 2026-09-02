//! NUMA-aware scheduling primitives.
//!
//! Adaptive backoff for stealing loops. Hardware topology is themis's to
//! answer — `themis::CpuTopology` — and is not mirrored here (ADR-037). The
//! runtime scheduler (`moirai_executor`'s `ThreadScheduler`) consumes these;
//! this module holds no scheduler of its own.

pub mod backoff;

pub use backoff::AdaptiveBackoff;
