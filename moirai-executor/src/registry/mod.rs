//! Task registry for tracking and managing task lifecycle.

pub(crate) mod diagnostics;
mod directory;
#[allow(clippy::module_inception)]
pub(crate) mod registry;
mod retention;
#[cfg(test)]
mod retention_tests;
pub(crate) mod state;
#[cfg(test)]
mod tests;
pub(crate) mod token;
#[cfg(test)]
mod token_tests;

pub(crate) use registry::CancelOutcome;
pub use registry::TaskRegistry;
pub use retention::RetentionPolicy;
pub(crate) use token::{
    OwnedStateLease, RunningTaskToken, SchedulerStateLease, StateLease, TaskLifecycleToken,
};
