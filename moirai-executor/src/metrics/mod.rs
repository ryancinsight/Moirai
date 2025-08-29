//! Performance metrics and monitoring for the executor system.
//!
//! This module provides comprehensive metrics collection and reporting
//! for monitoring executor performance and identifying bottlenecks.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// Import centralized constants (SSOT compliance)
use moirai_core::constants::{
    BYTES_TO_MB_FACTOR, DEFAULT_UTILIZATION, MAX_SUCCESS_RATE, PERCENTAGE_PRECISION_FACTOR,
};

/// Comprehensive executor performance metrics
#[derive(Debug)]
pub struct ExecutorMetrics {
    // Task counters
    pub tasks_spawned: AtomicU64,
    pub tasks_completed: AtomicU64,
    pub tasks_failed: AtomicU64,

    // Timing metrics
    pub total_execution_time: AtomicU64,  // in nanoseconds
    pub average_task_duration: AtomicU64, // in nanoseconds

    // Thread pool metrics
    pub active_workers: AtomicUsize,
    pub idle_workers: AtomicUsize,
    pub total_workers: AtomicUsize,

    // Queue metrics
    pub pending_tasks: AtomicUsize,
    pub max_queue_depth: AtomicUsize,

    // Resource utilization
    pub memory_usage: AtomicUsize,  // in bytes
    pub cpu_utilization: AtomicU64, // percentage * CPU_UTILIZATION_PRECISION

    // System metrics
    pub started_at: Instant,
    pub last_updated: std::sync::Mutex<Instant>,
}

impl ExecutorMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            tasks_spawned: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
            tasks_failed: AtomicU64::new(0),
            total_execution_time: AtomicU64::new(0),
            average_task_duration: AtomicU64::new(0),
            active_workers: AtomicUsize::new(0),
            idle_workers: AtomicUsize::new(0),
            total_workers: AtomicUsize::new(0),
            pending_tasks: AtomicUsize::new(0),
            max_queue_depth: AtomicUsize::new(0),
            memory_usage: AtomicUsize::new(0),
            cpu_utilization: AtomicU64::new(0),
            started_at: now,
            last_updated: std::sync::Mutex::new(now),
        }
    }

    /// Record a new task being spawned
    pub fn record_task_spawned(&self) {
        self.tasks_spawned.fetch_add(1, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Record a task completion with execution time
    pub fn record_task_completed(&self, execution_time: Duration) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
        let exec_nanos = execution_time.as_nanos() as u64;
        self.total_execution_time
            .fetch_add(exec_nanos, Ordering::Relaxed);

        // Update average (simple approach - more sophisticated methods could be used)
        let completed = self.tasks_completed.load(Ordering::Relaxed);
        let total_time = self.total_execution_time.load(Ordering::Relaxed);
        if completed > 0 {
            self.average_task_duration
                .store(total_time / completed, Ordering::Relaxed);
        }

        self.update_timestamp();
    }

    /// Record a task failure
    pub fn record_task_failed(&self) {
        self.tasks_failed.fetch_add(1, Ordering::Relaxed);
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

    /// Update resource utilization metrics
    pub fn update_resource_metrics(&self, memory_bytes: usize, cpu_percent: f64) {
        self.memory_usage.store(memory_bytes, Ordering::Relaxed);
        self.cpu_utilization.store(
            (cpu_percent * PERCENTAGE_PRECISION_FACTOR) as u64,
            Ordering::Relaxed,
        );
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
        Duration::from_nanos(self.average_task_duration.load(Ordering::Relaxed))
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

    /// Get memory usage in MB
    pub fn memory_usage_mb(&self) -> f64 {
        self.memory_usage.load(Ordering::Relaxed) as f64 / BYTES_TO_MB_FACTOR
    }

    /// Get CPU utilization percentage
    pub fn cpu_utilization_percent(&self) -> f64 {
        self.cpu_utilization.load(Ordering::Relaxed) as f64 / PERCENTAGE_PRECISION_FACTOR
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Update the last updated timestamp
    fn update_timestamp(&self) {
        if let Ok(mut last_updated) = self.last_updated.try_lock() {
            *last_updated = Instant::now();
        }
    }

    /// Reset all metrics (useful for testing)
    pub fn reset(&self) {
        self.tasks_spawned.store(0, Ordering::Relaxed);
        self.tasks_completed.store(0, Ordering::Relaxed);
        self.tasks_failed.store(0, Ordering::Relaxed);
        self.total_execution_time.store(0, Ordering::Relaxed);
        self.average_task_duration.store(0, Ordering::Relaxed);
        self.active_workers.store(0, Ordering::Relaxed);
        self.idle_workers.store(0, Ordering::Relaxed);
        self.total_workers.store(0, Ordering::Relaxed);
        self.pending_tasks.store(0, Ordering::Relaxed);
        self.max_queue_depth.store(0, Ordering::Relaxed);
        self.memory_usage.store(0, Ordering::Relaxed);
        self.cpu_utilization.store(0, Ordering::Relaxed);
        self.update_timestamp();
    }
}

impl Default for ExecutorMetrics {
    fn default() -> Self {
        Self::new()
    }
}
