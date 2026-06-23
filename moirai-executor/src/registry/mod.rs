//! Task registry for tracking and managing task lifecycle.

pub(crate) mod diagnostics;
#[allow(clippy::module_inception)]
pub(crate) mod registry;
pub(crate) mod state;
#[cfg(test)]
mod tests;
pub(crate) mod token;

pub use registry::TaskRegistry;
pub(crate) use token::{RunningTaskToken, TaskLifecycleToken};
