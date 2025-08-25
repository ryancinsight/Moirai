//! Task registry for tracking and managing task lifecycle.
//!
//! This module provides centralized task tracking, enabling monitoring,
//! debugging, and coordination of task execution across the system.

use super::task::TaskMetadata;
use std::collections::HashMap;
use std::time::Instant;

/// Central registry for tracking all tasks in the system
#[derive(Debug)]
pub struct TaskRegistry {
    tasks: HashMap<u64, TaskMetadata>,
    next_id: u64,
}

impl TaskRegistry {
    /// Create a new task registry
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            next_id: 1,
        }
    }

    /// Register a new task and return its ID
    pub fn register_task(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let metadata = TaskMetadata::new(id);
        self.tasks.insert(id, metadata);
        id
    }

    /// Mark a task as started
    pub fn mark_started(&mut self, task_id: u64, worker_id: usize) {
        if let Some(metadata) = self.tasks.get_mut(&task_id) {
            metadata.mark_started(worker_id);
        }
    }

    /// Mark a task as completed
    pub fn mark_completed(&mut self, task_id: u64) {
        if let Some(metadata) = self.tasks.get_mut(&task_id) {
            metadata.mark_completed();
        }
    }

    /// Check if a task is completed
    pub fn is_completed(&self, task_id: u64) -> bool {
        self.tasks
            .get(&task_id)
            .map(|m| m.completed_at.is_some())
            .unwrap_or(false)
    }

    /// Get task metadata
    pub fn get_metadata(&self, task_id: u64) -> Option<&TaskMetadata> {
        self.tasks.get(&task_id)
    }

    /// Remove old completed tasks to prevent memory growth
    pub fn cleanup_completed(&mut self, older_than: std::time::Duration) {
        let cutoff = Instant::now() - older_than;
        self.tasks.retain(|_, metadata| {
            match metadata.completed_at {
                Some(completed) => completed > cutoff,
                None => true, // Keep running tasks
            }
        });
    }

    /// Get count of active tasks
    pub fn active_count(&self) -> usize {
        self.tasks
            .values()
            .filter(|m| m.completed_at.is_none())
            .count()
    }

    /// Get count of completed tasks
    pub fn completed_count(&self) -> usize {
        self.tasks
            .values()
            .filter(|m| m.completed_at.is_some())
            .count()
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}
