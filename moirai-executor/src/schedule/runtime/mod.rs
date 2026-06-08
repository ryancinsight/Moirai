//! Unified thread scheduler runtime.

pub mod types;
pub mod worker;
pub mod scheduler;

#[cfg(test)]
mod tests;

pub use types::{
    ScheduleMetrics,
    SchedulerScope,
    ThreadScheduler,
};

#[cfg(feature = "scheduler-diagnostics")]
pub use types::{
    ContendedWakeDecision,
    DiagnosticWakeDecision,
    EmptyWakeDecision,
    SaturatedWakeDecision,
};
