use std::sync::Arc;

use moirai_core::{
    constants::DEFAULT_POLL_INTERVAL_MS,
    error::{ExecutorError, ExecutorResult, TaskError},
    executor::{TaskManager, TaskStats, TaskStatus},
    task::TaskId,
    Priority,
};

use super::HybridExecutor;

impl TaskManager for HybridExecutor {
    fn cancel_task(&self, id: TaskId) -> ExecutorResult<()> {
        let registry = self.task_registry.lock().map_err(|_| {
            ExecutorError::ResourceExhausted("task registry lock poisoned".to_string())
        })?;

        if registry.get_metadata(id.0).is_some() {
            Ok(())
        } else {
            Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation))
        }
    }

    fn task_status(&self, id: TaskId) -> Option<TaskStatus> {
        let registry = self.task_registry.lock().ok()?;
        registry.get_metadata(id.0).map(|metadata| {
            if metadata.completed_at.is_some() {
                TaskStatus::Completed
            } else if metadata.started_at.is_some() {
                TaskStatus::Running
            } else {
                TaskStatus::Queued
            }
        })
    }

    fn wait_for_task(
        &self,
        id: TaskId,
        timeout: Option<core::time::Duration>,
    ) -> impl core::future::Future<Output = ExecutorResult<()>> + Send {
        let registry = Arc::clone(&self.task_registry);
        async move {
            let start = std::time::Instant::now();

            loop {
                let registry = registry.lock().map_err(|_| {
                    ExecutorError::ResourceExhausted("task registry lock poisoned".to_string())
                })?;
                if registry.is_completed(id.0) {
                    return Ok(());
                }

                if registry.get_metadata(id.0).is_none() {
                    return Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation));
                }
                drop(registry);

                if let Some(timeout) = timeout {
                    if start.elapsed() >= timeout {
                        return Err(ExecutorError::ResourceExhausted(
                            "Task wait timeout".to_string(),
                        ));
                    }
                }

                std::thread::sleep(std::time::Duration::from_millis(DEFAULT_POLL_INTERVAL_MS));
            }
        }
    }

    fn task_stats(&self, id: TaskId) -> Option<TaskStats> {
        let registry = self.task_registry.lock().ok()?;
        registry.get_metadata(id.0).map(|metadata| TaskStats {
            id,
            priority: Priority::Normal,
            status: if metadata.completed_at.is_some() {
                TaskStatus::Completed
            } else if metadata.started_at.is_some() {
                TaskStatus::Running
            } else {
                TaskStatus::Queued
            },
            spawn_time: metadata.created_at,
            start_time: metadata.started_at,
            completion_time: metadata.completed_at,
            preemption_count: 0,
            cpu_time_ns: metadata
                .execution_duration()
                .map_or(0, |duration| duration.as_nanos() as u64),
            memory_used_bytes: 0,
        })
    }
}
