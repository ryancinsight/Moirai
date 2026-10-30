//! Task lifecycle operations on the runtime's existing executor.

use super::Moirai;
use moirai_core::{
    TaskId,
    error::ExecutorResult,
    executor::{TaskManager, TaskStats, TaskStatus},
};
use std::time::Duration;

impl TaskManager for Moirai {
    fn cancel_task(&self, id: TaskId) -> ExecutorResult<()> {
        self.executor.cancel_task(id)
    }

    fn task_status(&self, id: TaskId) -> Option<TaskStatus> {
        self.executor.task_status(id)
    }

    fn wait_for_task(
        &self,
        id: TaskId,
        timeout: Option<Duration>,
    ) -> impl Future<Output = ExecutorResult<()>> + Send {
        self.executor.wait_for_task(id, timeout)
    }

    fn task_stats(&self, id: TaskId) -> Option<TaskStats> {
        self.executor.task_stats(id)
    }
}
