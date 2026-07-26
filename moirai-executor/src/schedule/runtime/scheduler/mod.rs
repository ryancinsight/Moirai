//! Unified thread scheduler module.

pub mod core;
pub mod data_parallel;
pub mod scope;

#[cfg(feature = "scheduler-diagnostics")]
pub mod diagnostics;
