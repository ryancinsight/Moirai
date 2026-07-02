//! Task management and execution abstractions.
//!
//! This module provides types and functionality for task lifecycle management,
//! specifically per-task metadata tracking consumed by the task registry.

use std::time::{Duration, Instant};

/// Task metadata for tracking and debugging
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    pub id: u64,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub worker_id: Option<usize>,
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
