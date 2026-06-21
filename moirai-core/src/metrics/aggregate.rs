//! Aggregated and compound metric types (TaskData, SchedulerData, GlobalMetrics, Snapshot).

use super::collector::{Counter, Gauge, Histogram};
use crate::scheduler::SchedulerId;
use std::collections::HashMap;
use std::time::Duration;

/// Metrics for individual tasks.
#[derive(Debug)]
pub struct TaskData {
    /// Number of tasks spawned in total
    pub spawned: Counter,
    /// Number of tasks that completed successfully
    pub completed: Counter,
    /// Histogram of task execution times in microseconds
    pub execution_time: Histogram,
    /// Histogram of task wait times in microseconds
    pub wait_time: Histogram,
}

impl TaskData {
    /// Create new task metrics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            spawned: Counter::new(),
            completed: Counter::new(),
            execution_time: Histogram::new(),
            wait_time: Histogram::new(),
        }
    }

    /// Record task execution metrics.
    pub fn record_execution(&self, execution_time: Duration) {
        self.completed.increment();
        // Handle potential truncation with try_from
        let micros = execution_time.as_micros();
        if let Ok(micros_u64) = u64::try_from(micros) {
            self.execution_time.record(micros_u64);
        } else {
            // For extremely long durations, record maximum value
            self.execution_time.record(u64::MAX);
        }
    }

    /// Record task wait time.
    pub fn record_wait(&self, wait_time: Duration) {
        // Handle potential truncation with try_from
        let micros = wait_time.as_micros();
        if let Ok(micros_u64) = u64::try_from(micros) {
            self.wait_time.record(micros_u64);
        } else {
            // For extremely long wait times, record maximum value
            self.wait_time.record(u64::MAX);
        }
    }

    /// Calculate task completion rate.
    pub fn completion_rate(&self) -> f64 {
        let spawned = self.spawned.get();
        if spawned == 0 {
            0.0
        } else {
            // Intentional precision loss for rate calculation
            #[allow(clippy::cast_precision_loss)]
            {
                self.completed.get() as f64 / spawned as f64
            }
        }
    }
}

impl Default for TaskData {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for individual schedulers.
#[derive(Debug)]
pub struct SchedulerData {
    /// Current number of tasks in the scheduler's queue
    pub queue_length: Gauge,
    /// Total number of tasks processed by this scheduler
    pub tasks_processed: Counter,
    /// Number of work-stealing attempts made
    pub steal_attempts: Counter,
    /// Number of successful work-stealing operations
    pub successful_steals: Counter,
    /// Current CPU utilization percentage (0-100)
    pub cpu_utilization: Gauge,
}

impl SchedulerData {
    /// Create new scheduler metrics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            queue_length: Gauge::new(),
            tasks_processed: Counter::new(),
            steal_attempts: Counter::new(),
            successful_steals: Counter::new(),
            cpu_utilization: Gauge::new(),
        }
    }

    /// Record CPU utilization as a percentage.
    pub fn record_cpu_utilization(&self, utilization: f32) -> f32 {
        // Handle potential truncation and sign loss with bounds checking
        let utilization_percent = (utilization * 100.0).clamp(0.0, 100.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            self.cpu_utilization.set(utilization_percent as u64);
        }
        utilization_percent
    }

    /// Calculate steal success rate.
    pub fn steal_success_rate(&self) -> f64 {
        let attempts = self.steal_attempts.get();
        if attempts == 0 {
            0.0
        } else {
            // Intentional precision loss for rate calculation
            #[allow(clippy::cast_precision_loss)]
            {
                self.successful_steals.get() as f64 / attempts as f64
            }
        }
    }
}

impl Default for SchedulerData {
    fn default() -> Self {
        Self::new()
    }
}

/// Global metrics aggregated across all executor components.
///
/// This struct provides system-wide performance and operational metrics,
/// combining data from all schedulers, workers, and runtime components.
#[allow(clippy::module_name_repetitions)]
pub struct GlobalMetrics {
    /// Task-related metrics aggregated across all schedulers
    pub tasks: TaskData,
    /// Per-scheduler metrics indexed by scheduler ID
    pub schedulers: HashMap<SchedulerId, SchedulerData>,
}

impl GlobalMetrics {
    /// Create new global metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: TaskData::new(),
            schedulers: HashMap::new(),
        }
    }

    /// Get or create scheduler metrics.
    pub fn scheduler(&mut self, id: SchedulerId) -> &mut SchedulerData {
        self.schedulers.entry(id).or_default()
    }

    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> Snapshot {
        let total_queue_length = self
            .schedulers
            .values()
            .map(|s| s.queue_length.get())
            .sum::<u64>();

        let scheduler_count = self.schedulers.len();
        let average_queue_length = if scheduler_count == 0 {
            0.0
        } else {
            // Intentional precision loss for averaging
            #[allow(clippy::cast_precision_loss)]
            {
                total_queue_length as f64 / scheduler_count as f64
            }
        };

        let (total_steal_attempts, total_successful_steals) =
            self.schedulers
                .values()
                .fold((0, 0), |(attempts, steals), scheduler| {
                    (
                        attempts + scheduler.steal_attempts.get(),
                        steals + scheduler.successful_steals.get(),
                    )
                });

        Snapshot {
            total_tasks_spawned: self.tasks.spawned.get(),
            total_tasks_completed: self.tasks.completed.get(),
            average_queue_length,
            total_steal_attempts,
            total_successful_steals,
        }
    }
}

impl Default for GlobalMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Total number of tasks spawned across all schedulers
    pub total_tasks_spawned: u64,
    /// Total number of tasks completed across all schedulers  
    pub total_tasks_completed: u64,
    /// Average queue length across all active schedulers
    pub average_queue_length: f64,
    /// Total number of work-stealing attempts made
    pub total_steal_attempts: u64,
    /// Total number of successful work-stealing operations
    pub total_successful_steals: u64,
}

impl Snapshot {
    /// Calculate overall steal success rate.
    #[must_use]
    pub fn steal_success_rate(&self) -> f64 {
        if self.total_steal_attempts == 0 {
            0.0
        } else {
            // Intentional precision loss for rate calculation
            #[allow(clippy::cast_precision_loss)]
            {
                (self.total_successful_steals as f64 / self.total_steal_attempts as f64) * 100.0
            }
        }
    }
}
