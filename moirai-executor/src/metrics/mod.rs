//! Performance metrics and monitoring for the executor system.
//!
//! This module provides comprehensive metrics collection and reporting
//! for monitoring executor performance and identifying bottlenecks.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Percentage conversion factor to maintain precision across metrics
const PERCENTAGE_PRECISION_FACTOR: f64 = 100.0;

/// Maximum success rate when no tasks have failed
const MAX_SUCCESS_RATE: f64 = 100.0;

/// Default utilization when no workers are available
const DEFAULT_UTILIZATION: f64 = 0.0;

/// Executor performance metrics.
///
/// Every field is written by a real production path; untracked quantities
/// (process memory, CPU utilization) deliberately have no fields here.
#[derive(Debug)]
pub struct ExecutorMetrics {
    // Task counters
    pub tasks_spawned: AtomicU64,
    pub tasks_completed: AtomicU64,
    pub tasks_failed: AtomicU64,
    /// Tasks whose cancel request was honored before the body ran.
    pub tasks_cancelled: AtomicU64,

    // Timing metrics
    pub total_execution_time: AtomicU64, // in nanoseconds

    // Thread pool metrics
    pub active_workers: AtomicUsize,
    pub idle_workers: AtomicUsize,
    pub total_workers: AtomicUsize,

    // Queue metrics
    pub pending_tasks: AtomicUsize,
    pub max_queue_depth: AtomicUsize,

    // System metrics
    pub started_at: Instant,
    pub last_updated_after_ns: AtomicU64,
}

impl ExecutorMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            tasks_spawned: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
            tasks_failed: AtomicU64::new(0),
            tasks_cancelled: AtomicU64::new(0),
            total_execution_time: AtomicU64::new(0),
            active_workers: AtomicUsize::new(0),
            idle_workers: AtomicUsize::new(0),
            total_workers: AtomicUsize::new(0),
            pending_tasks: AtomicUsize::new(0),
            max_queue_depth: AtomicUsize::new(0),
            started_at: now,
            last_updated_after_ns: AtomicU64::new(0),
        }
    }

    /// Record a new task being spawned
    pub fn record_task_spawned(&self) {
        self.tasks_spawned.fetch_add(1, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Record a task completion with execution time
    pub fn record_task_completed(&self, execution_time: Duration) {
        let exec_nanos = execution_time.as_nanos() as u64;
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time
            .fetch_add(exec_nanos, Ordering::Relaxed);

        self.update_timestamp();
    }

    /// Record a task failure
    pub fn record_task_failed(&self) {
        self.tasks_failed.fetch_add(1, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Record a task whose cancel request was honored before its body ran.
    pub fn record_task_cancelled(&self) {
        self.tasks_cancelled.fetch_add(1, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Update worker count metrics
    pub fn update_worker_counts(&self, active: usize, idle: usize, total: usize) {
        self.active_workers.store(active, Ordering::Relaxed);
        self.idle_workers.store(idle, Ordering::Relaxed);
        self.total_workers.store(total, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Update queue metrics
    pub fn update_queue_metrics(&self, pending: usize) {
        self.pending_tasks.store(pending, Ordering::Relaxed);

        // Update max queue depth if this is a new maximum
        let current_max = self.max_queue_depth.load(Ordering::Relaxed);
        if pending > current_max {
            self.max_queue_depth.store(pending, Ordering::Relaxed);
        }

        self.update_timestamp();
    }

    /// Get current throughput (tasks per second)
    pub fn throughput(&self) -> f64 {
        let elapsed = self.started_at.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.tasks_completed.load(Ordering::Relaxed) as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        let completed = self.tasks_completed.load(Ordering::Relaxed);
        let failed = self.tasks_failed.load(Ordering::Relaxed);
        let total = completed + failed;

        if total > 0 {
            (completed as f64 / total as f64) * PERCENTAGE_PRECISION_FACTOR
        } else {
            MAX_SUCCESS_RATE
        }
    }

    /// Get average task duration
    pub fn average_task_duration(&self) -> Duration {
        let completed = self.tasks_completed.load(Ordering::Relaxed);
        let average = self
            .total_execution_time
            .load(Ordering::Relaxed)
            .checked_div(completed)
            .unwrap_or(0);

        Duration::from_nanos(average)
    }

    /// Get worker utilization percentage
    pub fn worker_utilization(&self) -> f64 {
        let total = self.total_workers.load(Ordering::Relaxed);
        if total > 0 {
            let active = self.active_workers.load(Ordering::Relaxed);
            (active as f64 / total as f64) * PERCENTAGE_PRECISION_FACTOR
        } else {
            DEFAULT_UTILIZATION
        }
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Get the last metrics update timestamp.
    pub fn last_updated(&self) -> Instant {
        self.started_at
            .checked_add(Duration::from_nanos(
                self.last_updated_after_ns.load(Ordering::Relaxed),
            ))
            .unwrap_or(self.started_at)
    }

    /// Update the last updated timestamp
    fn update_timestamp(&self) {
        self.last_updated_after_ns
            .store(elapsed_nanos_since(self.started_at), Ordering::Relaxed);
    }

    /// Reset all metrics (useful for testing)
    pub fn reset(&self) {
        self.tasks_spawned.store(0, Ordering::Relaxed);
        self.tasks_completed.store(0, Ordering::Relaxed);
        self.tasks_failed.store(0, Ordering::Relaxed);
        self.tasks_cancelled.store(0, Ordering::Relaxed);
        self.total_execution_time.store(0, Ordering::Relaxed);
        self.active_workers.store(0, Ordering::Relaxed);
        self.idle_workers.store(0, Ordering::Relaxed);
        self.total_workers.store(0, Ordering::Relaxed);
        self.pending_tasks.store(0, Ordering::Relaxed);
        self.max_queue_depth.store(0, Ordering::Relaxed);
        self.update_timestamp();
    }
}

fn elapsed_nanos_since(origin: Instant) -> u64 {
    origin.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

impl Default for ExecutorMetrics {
    fn default() -> Self {
        Self::new()
    }
}
