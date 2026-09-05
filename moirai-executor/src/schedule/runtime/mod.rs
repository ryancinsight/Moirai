//! Unified thread scheduler runtime.

pub(super) mod blocking;
pub(super) mod idle;
pub mod idle_hooks;
pub mod scheduler;
pub mod types;
pub mod worker;

#[cfg(test)]
mod tests;

pub use idle_hooks::{register_idle_hook, run_idle_hooks, IdleHook, IdleHookRegistrationError};
pub use types::{ScheduleMetrics, SchedulerScope, ThreadScheduler};

#[cfg(feature = "scheduler-diagnostics")]
pub use types::{
    ContendedWakeDecision, DiagnosticWakeDecision, EmptyWakeDecision, SaturatedWakeDecision,
};
