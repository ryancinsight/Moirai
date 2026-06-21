//! Task registry for tracking and managing task lifecycle.

pub(crate) mod state;
pub(crate) mod token;
#[allow(clippy::module_inception)]
pub(crate) mod registry;
pub(crate) mod diagnostics;
#[cfg(test)]
mod tests;

pub use registry::TaskRegistry;
pub(crate) use token::{TaskLifecycleToken, RunningTaskToken};
