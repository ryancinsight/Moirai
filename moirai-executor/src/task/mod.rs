//! Task management and execution abstractions.
//!
//! This module provides types and functionality for task lifecycle management,
//! including metadata tracking, performance metrics, and async task wrappers.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// Task performance metrics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct TaskPerformanceMetrics {
    pub total_tasks: u64,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub average_completion_time: Duration,
    pub total_execution_time: Duration,
    pub last_updated: Instant,
}

impl TaskPerformanceMetrics {
    /// Create new performance metrics tracker
    pub fn new() -> Self {
        Self {
            total_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            average_completion_time: Duration::ZERO,
            total_execution_time: Duration::ZERO,
            last_updated: Instant::now(),
        }
    }

    /// Record task completion
    pub fn record_completion(&mut self, execution_time: Duration) {
        self.completed_tasks += 1;
        self.total_execution_time += execution_time;
        self.average_completion_time =
            self.total_execution_time / self.completed_tasks.max(1) as u32;
        self.last_updated = Instant::now();
    }

    /// Record task failure
    pub fn record_failure(&mut self) {
        self.failed_tasks += 1;
        self.last_updated = Instant::now();
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_tasks == 0 {
            100.0
        } else {
            (self.completed_tasks as f64 / self.total_tasks as f64) * 100.0
        }
    }
}

impl Default for TaskPerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

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

/// Future for waiting on task completion
pub struct TaskWaitFuture {
    pub(crate) task_id: u64,
    pub(crate) registry: Arc<Mutex<super::registry::TaskRegistry>>,
}

impl Future for TaskWaitFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let registry = self.registry.lock().unwrap();
        if registry.is_completed(self.task_id) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}
