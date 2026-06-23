//! Unified thread scheduler runtime.

pub(super) mod idle;
pub mod scheduler;
pub mod types;
pub mod worker;

#[cfg(test)]
mod tests;

pub use types::{ScheduleMetrics, SchedulerScope, ThreadScheduler};

#[cfg(feature = "scheduler-diagnostics")]
pub use types::{
    ContendedWakeDecision, DiagnosticWakeDecision, EmptyWakeDecision, SaturatedWakeDecision,
};
