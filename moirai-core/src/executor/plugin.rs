//! Plugin interface for executor extensions.

use crate::error::ExecutorResult;
use crate::{Priority, TaskId};

/// Plugin interface for extending executor functionality.
///
/// Plugins provide a way to add custom behavior to the executor lifecycle
/// without modifying the core execution logic.
#[allow(clippy::module_name_repetitions)]
pub trait ExecutorPlugin: Send + Sync + 'static {
    /// Initialize the plugin with access to executor configuration.
    ///
    /// # Errors
    /// Returns `ExecutorError` if the plugin cannot be properly initialized due to:
    /// - Invalid configuration parameters
    /// - Resource allocation failures
    /// - Dependency conflicts with other plugins
    fn initialize(&mut self) -> ExecutorResult<()> {
        Ok(())
    }

    /// Called before a task is spawned.
    fn before_task_spawn(&self, task_id: TaskId, priority: Priority) {
        let _ = (task_id, priority); // Default: no-op
    }

    /// Called after a task is spawned.
    fn after_task_spawn(&self, task_id: TaskId) {
        let _ = task_id; // Default: no-op
    }

    /// Called before a task starts executing.
    fn before_task_execute(&self, task_id: TaskId) {
        let _ = task_id; // Default: no-op
    }

    /// Called after a task completes.
    fn after_task_complete(&self, task_id: TaskId, success: bool) {
        let _ = (task_id, success); // Default: no-op
    }

    /// Called during executor shutdown.
    fn on_shutdown(&self) {
        // Default: no-op
    }

    /// Get plugin name for debugging.
    fn name(&self) -> &'static str {
        "unknown"
    }
}
