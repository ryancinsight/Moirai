//! Unified thread scheduler module.

mod construction;
pub mod core;
pub mod data_parallel;
mod lifecycle;
pub(super) use lifecycle::close_admission;
pub(super) mod placement;
pub mod scope;

/// Diagnostic probes for scheduler internals used by provider conformance tests.
#[cfg(feature = "scheduler-diagnostics")]
pub mod diagnostics;
