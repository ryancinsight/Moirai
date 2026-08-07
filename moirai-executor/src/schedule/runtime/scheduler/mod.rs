//! Unified thread scheduler module.

pub mod core;
pub mod data_parallel;
pub mod scope;

/// Diagnostic probes for scheduler internals used by provider conformance tests.
#[cfg(feature = "scheduler-diagnostics")]
pub mod diagnostics;
