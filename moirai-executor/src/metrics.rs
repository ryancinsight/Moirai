//! Performance metrics and monitoring for the hybrid executor.
//!
//! This module provides comprehensive performance tracking capabilities
//! for both individual tasks and worker threads, following the Information
//! Expert pattern where each metric owns its data and calculations.

use moirai_utils::CacheAligned;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
use crate::types::WorkerId;

/// Performance metrics for individual task execution tracking.
/// 
/// Each task's performance characteristics are captured to enable:
/// - Task scheduling optimization based on historical performance
/// - Resource usage analysis and optimization
/// - Performance regression detection
/// - Load balancing decision making
#[derive(Debug, Clone)]
pub struct TaskPerformanceMetrics {
    /// CPU time consumed by this task in nanoseconds
    pub cpu_time_ns: u64,
    /// Memory usage at task start in bytes
    pub memory_start_bytes: u64,
    /// Peak memory usage during execution in bytes
    pub memory_peak_bytes: u64,
    /// Number of times the task was preempted
    pub preemption_count: u32,
    /// Task execution start time
    pub start_time: Instant,
    /// Last time metrics were updated
    pub last_update: Instant,
}

impl TaskPerformanceMetrics {
    /// Create new task performance metrics with initial values.
    pub fn new(memory_start_bytes: u64) -> Self {
        let now = Instant::now();
        Self {
            cpu_time_ns: 0,
            memory_start_bytes,
            memory_peak_bytes: memory_start_bytes,
            preemption_count: 0,
            start_time: now,
            last_update: now,
        }
    }

    /// Calculate total execution time from start.
    pub fn execution_time(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get memory growth since task start.
    pub fn memory_growth(&self) -> u64 {
        self.memory_peak_bytes.saturating_sub(self.memory_start_bytes)
    }
    
    /// Check if task has been preempted.
    pub fn was_preempted(&self) -> bool {
        self.preemption_count > 0
    }
    
    /// Update metrics with current values.
    pub fn update(&mut self, current_memory: u64) {
        self.memory_peak_bytes = self.memory_peak_bytes.max(current_memory);
        self.last_update = Instant::now();
    }
    
    /// Increment preemption count.
    pub fn increment_preemption(&mut self) {
        self.preemption_count += 1;
        self.last_update = Instant::now();
    }
}

/// Metrics collected per worker thread.
/// 
/// Each atomic counter is cache-aligned to prevent false sharing
/// between worker threads, ensuring optimal performance in multi-core
/// scenarios where multiple workers are updating metrics concurrently.
#[derive(Debug)]
pub struct WorkerMetrics {
    /// Total number of tasks executed by this worker
    pub(crate) tasks_executed: CacheAligned<AtomicU64>,
    /// Number of work-stealing attempts made by this worker
    pub(crate) steal_attempts: CacheAligned<AtomicU64>,
    /// Number of successful work-stealing operations
    pub(crate) successful_steals: CacheAligned<AtomicU64>,
    /// Total execution time spent by this worker (nanoseconds)
    pub(crate) execution_time_ns: CacheAligned<AtomicU64>,
}

impl Default for WorkerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerMetrics {
    /// Create new worker metrics with zero initial values.
    pub fn new() -> Self {
        Self {
            tasks_executed: CacheAligned::new(AtomicU64::new(0)),
            steal_attempts: CacheAligned::new(AtomicU64::new(0)),
            successful_steals: CacheAligned::new(AtomicU64::new(0)),
            execution_time_ns: CacheAligned::new(AtomicU64::new(0)),
        }
    }

    /// Record completion of a task execution.
    pub fn record_task_completion(&self, execution_time_ns: u64) {
        self.tasks_executed.fetch_add(1, Ordering::Relaxed);
        self.execution_time_ns.fetch_add(execution_time_ns, Ordering::Relaxed);
    }

    /// Record a work-stealing attempt.
    pub fn record_steal_attempt(&self, successful: bool) {
        self.steal_attempts.fetch_add(1, Ordering::Relaxed);
        if successful {
            self.successful_steals.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get number of tasks executed by this worker.
    pub fn tasks_executed(&self) -> u64 {
        self.tasks_executed.load(Ordering::Relaxed)
    }

    /// Get number of steal attempts made by this worker.
    pub fn steal_attempts(&self) -> u64 {
        self.steal_attempts.load(Ordering::Relaxed)
    }

    /// Get number of successful steals by this worker.
    pub fn successful_steals(&self) -> u64 {
        self.successful_steals.load(Ordering::Relaxed)
    }

    /// Get total execution time in nanoseconds.
    pub fn execution_time_ns(&self) -> u64 {
        self.execution_time_ns.load(Ordering::Relaxed)
    }

    /// Calculate steal success rate as a percentage.
    pub fn steal_success_rate(&self) -> f64 {
        let attempts = self.steal_attempts();
        if attempts == 0 {
            0.0
        } else {
            (self.successful_steals() as f64 / attempts as f64) * 100.0
        }
    }

    /// Get average task execution time in nanoseconds.
    pub fn average_task_time_ns(&self) -> f64 {
        let tasks = self.tasks_executed();
        if tasks == 0 {
            0.0
        } else {
            self.execution_time_ns() as f64 / tasks as f64
        }
    }
}

/// Snapshot of worker metrics for reporting purposes.
/// 
/// Provides a consistent view of worker performance at a specific
/// point in time, useful for monitoring and debugging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerSnapshot {
    /// Worker identifier
    pub id: WorkerId,
    /// Total tasks executed by this worker
    pub tasks_executed: u64,
    /// Number of work-stealing attempts
    pub steal_attempts: u64,
    /// Number of successful work-stealing operations
    pub successful_steals: u64,
    /// Total execution time in nanoseconds
    pub execution_time_ns: u64,
}

impl WorkerSnapshot {
    /// Create a snapshot from worker metrics.
    pub fn from_metrics(id: WorkerId, metrics: &WorkerMetrics) -> Self {
        Self {
            id,
            tasks_executed: metrics.tasks_executed(),
            steal_attempts: metrics.steal_attempts(),
            successful_steals: metrics.successful_steals(),
            execution_time_ns: metrics.execution_time_ns(),
        }
    }

    /// Calculate steal success rate as a percentage.
    pub fn steal_success_rate(&self) -> f64 {
        if self.steal_attempts == 0 {
            0.0
        } else {
            (self.successful_steals as f64 / self.steal_attempts as f64) * 100.0
        }
    }

    /// Get average task execution time in nanoseconds.
    pub fn average_task_time_ns(&self) -> f64 {
        if self.tasks_executed == 0 {
            0.0
        } else {
            self.execution_time_ns as f64 / self.tasks_executed as f64
        }
    }
}