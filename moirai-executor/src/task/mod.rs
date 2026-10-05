//! Task management and execution abstractions.
//!
//! This module provides types and functionality for task lifecycle management,
//! specifically per-task metadata tracking consumed by the task registry.

use std::time::{Duration, Instant};

use moirai_core::Priority;

/// Task metadata for tracking and debugging
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    /// Executor-unique task identifier.
    pub id: u64,
    /// Instant the task was accepted.
    pub created_at: Instant,
    /// Instant the body started, once running.
    pub started_at: Option<Instant>,
    /// Instant the body finished, once complete.
    pub completed_at: Option<Instant>,
    /// Worker that executed the body, once assigned.
    pub worker_id: Option<usize>,
    /// Priority the task was spawned with.
    pub priority: Priority,
    /// True when a cancel request was honored before the task body ran.
    pub cancelled: bool,
}

impl TaskMetadata {
    /// Create new task metadata
    pub fn new(id: u64) -> Self {
        Self {
            id,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            worker_id: None,
            priority: Priority::Normal,
            cancelled: false,
        }
    }

    /// Mark task as started
    pub fn mark_started(&mut self, worker_id: usize) {
        self.started_at = Some(Instant::now());
        self.worker_id = Some(worker_id);
    }

    /// Mark task as completed
    pub fn mark_completed(&mut self) {
        self.completed_at = Some(Instant::now());
    }

    /// Get task execution duration
    pub fn execution_duration(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }
}
