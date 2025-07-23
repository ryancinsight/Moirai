//! Metrics collection and monitoring for Moirai.

use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::Duration;
use crate::scheduler::SchedulerId;

/// High-precision timestamp for performance measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Instant(u64);

impl Instant {
    /// Create a new instant representing the current time.
    #[must_use]
    pub fn now() -> Self {
        Self(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX), // Handle potential truncation gracefully
        )
    }

    /// Calculate the duration since an earlier instant.
    #[must_use]
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }

    /// Get the elapsed time since this instant.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        Self::now().duration_since(*self)
    }
}

/// A duration type optimized for performance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimeDuration(u64);

impl TimeDuration {
    /// Create a duration from nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Create a duration from microseconds.
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }

    /// Create a duration from milliseconds.
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }

    /// Create a duration from seconds.
    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }

    /// Get the duration in nanoseconds.
    #[must_use]
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Get the duration in microseconds.
    #[must_use]
    pub const fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }

    /// Get the duration in milliseconds.
    #[must_use]
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }

    /// Get the duration in seconds.
    #[must_use]
    pub const fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Get the duration as seconds with fractional precision.
    #[must_use]
    pub fn as_secs_f64(&self) -> f64 {
        // Use explicit conversion to handle precision loss intentionally
        #[allow(clippy::cast_precision_loss)]
        {
            self.0 as f64 / 1_000_000_000.0
        }
    }
}

impl std::fmt::Display for TimeDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 < 1_000 {
            write!(f, "{}ns", self.0)
        } else if self.0 < 1_000_000 {
            #[allow(clippy::cast_precision_loss)]
            {
                write!(f, "{:.1}μs", self.0 as f64 / 1_000.0)
            }
        } else if self.0 < 1_000_000_000 {
            #[allow(clippy::cast_precision_loss)]
            {
                write!(f, "{:.1}ms", self.0 as f64 / 1_000_000.0)
            }
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                write!(f, "{:.1}s", self.0 as f64 / 1_000_000_000.0)
            }
        }
    }
}

/// Thread-safe counter for performance metrics.
#[derive(Debug)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Create a new counter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Increment the counter by 1.
    pub fn increment(&self) {
        self.add(1);
    }

    /// Add a value to the counter.
    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe gauge for current values.
#[derive(Debug)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Create a new gauge.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Set the gauge value.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Increment the gauge by 1.
    pub fn increment(&self) {
        self.add(1);
    }

    /// Add to the gauge value.
    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    /// Subtract from the gauge value.
    pub fn subtract(&self, value: u64) {
        self.value.fetch_sub(value, Ordering::Relaxed);
    }
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe histogram for value distributions.
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; 16],
    sum: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram.
    #[must_use]
    pub const fn new() -> Self {
        // Use const fn to avoid interior mutable const warning
        const fn new_atomic() -> AtomicU64 {
            AtomicU64::new(0)
        }
        
        Self {
            buckets: [
                new_atomic(), new_atomic(), new_atomic(), new_atomic(),
                new_atomic(), new_atomic(), new_atomic(), new_atomic(),
                new_atomic(), new_atomic(), new_atomic(), new_atomic(),
                new_atomic(), new_atomic(), new_atomic(), new_atomic(),
            ],
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a value in the histogram.
    pub fn record(&self, value: u64) {
        // Simple bucket assignment - in a real implementation this would be more sophisticated
        let bucket_index = if value == 0 {
            0
        } else {
            // Safe calculation to avoid overflow
            let leading_zeros = value.leading_zeros();
            if leading_zeros >= 15 {
                0
            } else {
                (15 - leading_zeros as usize).min(15)
            }
        };

        self.buckets[bucket_index].fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the total count of recorded values.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the sum of all recorded values.
    pub fn sum(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
    }

    /// Calculate the average of recorded values.
    pub fn average(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            0.0
        } else {
            // Intentional precision loss for averaging - use explicit allow
            #[allow(clippy::cast_precision_loss)]
            {
                self.sum() as f64 / count as f64
            }
        }
    }

    /// Get the count for a specific bucket.
    pub fn bucket_count(&self, bucket: usize) -> u64 {
        if bucket < self.buckets.len() {
            self.buckets[bucket].load(Ordering::Relaxed)
        } else {
            0
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

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

        let (total_steal_attempts, total_successful_steals) = self
            .schedulers
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);
        
        counter.increment();
        assert_eq!(counter.get(), 1);
        
        counter.add(5);
        assert_eq!(counter.get(), 6);
        
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new();
        assert_eq!(gauge.get(), 0);
        
        gauge.set(10);
        assert_eq!(gauge.get(), 10);
        
        gauge.increment();
        assert_eq!(gauge.get(), 11);
        
        gauge.subtract(1);
        assert_eq!(gauge.get(), 10);
        
        gauge.add(5);
        assert_eq!(gauge.get(), 15);
        
        gauge.subtract(3);
        assert_eq!(gauge.get(), 12);
    }

    #[test]
    fn test_histogram() {
        let histogram = Histogram::new();
        assert_eq!(histogram.count(), 0);
        assert_eq!(histogram.sum(), 0);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(histogram.average(), 0.0);
        }
        
        histogram.record(10);
        histogram.record(20);
        histogram.record(30);
        
        assert_eq!(histogram.count(), 3);
        assert_eq!(histogram.sum(), 60);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(histogram.average(), 20.0);
        }
    }

    #[test]
    fn test_task_data() {
        let metrics = TaskData::new();
        
        metrics.spawned.increment();
        metrics.spawned.increment();
        assert_eq!(metrics.spawned.get(), 2);
        
        metrics.record_execution(core::time::Duration::from_millis(1));
        assert_eq!(metrics.completed.get(), 1);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(metrics.completion_rate(), 0.5);
        }
        
        metrics.record_wait(core::time::Duration::from_millis(1));
    }

    #[test]
    fn test_scheduler_data() {
        let metrics = SchedulerData::new();
        
        metrics.queue_length.set(5);
        assert_eq!(metrics.queue_length.get(), 5);
        
        metrics.steal_attempts.increment();
        metrics.steal_attempts.increment();
        metrics.steal_attempts.increment();
        
        assert_eq!(metrics.steal_attempts.get(), 3);
        assert_eq!(metrics.successful_steals.get(), 0);
        assert_eq!(metrics.steal_success_rate(), 0.0);

        let utilization = metrics.record_cpu_utilization(0.75);
        assert_eq!(utilization, 75.0);
        assert_eq!(metrics.cpu_utilization.get(), 75);
    }

    #[test]
    fn test_global_metrics() {
        let mut global = GlobalMetrics::new();
        
        global.scheduler(SchedulerId::new(1)).queue_length.set(5);
        global.scheduler(SchedulerId::new(2)).queue_length.set(10);
        global.scheduler(SchedulerId::new(3)).queue_length.set(15);

        assert_eq!(global.snapshot().average_queue_length, 10.0);

        global.scheduler(SchedulerId::new(1)).steal_attempts.increment();
        global.scheduler(SchedulerId::new(2)).steal_attempts.increment();
        global.scheduler(SchedulerId::new(3)).steal_attempts.increment();
        global.scheduler(SchedulerId::new(1)).steal_attempts.increment();

        assert_eq!(global.snapshot().total_steal_attempts, 4);

        global.scheduler(SchedulerId::new(1)).successful_steals.increment();
        global.scheduler(SchedulerId::new(2)).successful_steals.increment();
        global.scheduler(SchedulerId::new(3)).successful_steals.increment();
        global.scheduler(SchedulerId::new(1)).successful_steals.increment();

        assert_eq!(global.snapshot().total_successful_steals, 4);
    }
}