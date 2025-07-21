//! Performance metrics and monitoring for Moirai.

use crate::scheduler::SchedulerId;
use alloc::collections::BTreeMap;
use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "std")]
use std::time::Instant;

#[cfg(not(feature = "std"))]
pub struct Instant {
    // Minimal fallback implementation for no_std
    // In a real implementation, this would use a monotonic timer
    timestamp: u64,
}

#[cfg(not(feature = "std"))]
impl Instant {
    pub fn now() -> Self {
        Self { timestamp: 0 }
    }

    pub fn elapsed(&self) -> core::time::Duration {
        core::time::Duration::from_secs(0)
    }
}

/// Performance counter that tracks events.
#[derive(Debug)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Create a new counter starting at zero.
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Increment the counter by one.
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

/// Gauge that tracks a current value that can go up or down.
#[derive(Debug)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Create a new gauge starting at zero.
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Set the gauge to a specific value.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment the gauge by one.
    pub fn increment(&self) {
        self.add(1);
    }

    /// Decrement the gauge by one.
    pub fn decrement(&self) {
        self.subtract(1);
    }

    /// Add a value to the gauge.
    pub fn add(&self, value: u64) {
        self.value.fetch_add(value, Ordering::Relaxed);
    }

    /// Subtract a value from the gauge.
    pub fn subtract(&self, value: u64) {
        self.value.fetch_sub(value, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the gauge to zero.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

/// Histogram that tracks the distribution of values.
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; 16], // Simple fixed-size buckets
    sum: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram.
    pub const fn new() -> Self {
        const ATOMIC_ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            buckets: [ATOMIC_ZERO; 16],
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
            ((64 - value.leading_zeros()) as usize).min(15)
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
            self.sum() as f64 / count as f64
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

    /// Reset all counters.
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.sum.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Task execution metrics.
#[derive(Debug)]
pub struct TaskMetrics {
    /// Number of tasks spawned
    pub spawned: Counter,
    /// Number of tasks completed
    pub completed: Counter,
    /// Number of tasks cancelled
    pub cancelled: Counter,
    /// Task execution time histogram (microseconds)
    pub execution_time: Histogram,
    /// Task queue wait time histogram (microseconds)
    pub wait_time: Histogram,
}

impl TaskMetrics {
    /// Create new task metrics.
    pub const fn new() -> Self {
        Self {
            spawned: Counter::new(),
            completed: Counter::new(),
            cancelled: Counter::new(),
            execution_time: Histogram::new(),
            wait_time: Histogram::new(),
        }
    }

    /// Record task spawn.
    pub fn record_spawn(&self) {
        self.spawned.increment();
    }

    /// Record task completion with execution time.
    pub fn record_completion(&self, execution_time: core::time::Duration) {
        self.completed.increment();
        self.execution_time.record(execution_time.as_micros() as u64);
    }

    /// Record task cancellation.
    pub fn record_cancellation(&self) {
        self.cancelled.increment();
    }

    /// Record task wait time.
    pub fn record_wait_time(&self, wait_time: core::time::Duration) {
        self.wait_time.record(wait_time.as_micros() as u64);
    }

    /// Get completion rate (0.0 to 1.0).
    pub fn completion_rate(&self) -> f64 {
        let spawned = self.spawned.get();
        if spawned == 0 {
            0.0
        } else {
            self.completed.get() as f64 / spawned as f64
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.spawned.reset();
        self.completed.reset();
        self.cancelled.reset();
        self.execution_time.reset();
        self.wait_time.reset();
    }
}

impl Default for TaskMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduler metrics.
#[derive(Debug)]
pub struct SchedulerMetrics {
    /// Current queue length
    pub queue_length: Gauge,
    /// Number of steal attempts
    pub steal_attempts: Counter,
    /// Number of successful steals
    pub successful_steals: Counter,
    /// Number of times stolen from
    pub stolen_from: Counter,
    /// CPU utilization gauge (0-100)
    pub cpu_utilization: Gauge,
}

impl SchedulerMetrics {
    /// Create new scheduler metrics.
    pub const fn new() -> Self {
        Self {
            queue_length: Gauge::new(),
            steal_attempts: Counter::new(),
            successful_steals: Counter::new(),
            stolen_from: Counter::new(),
            cpu_utilization: Gauge::new(),
        }
    }

    /// Record a steal attempt.
    pub fn record_steal_attempt(&self, successful: bool) {
        self.steal_attempts.increment();
        if successful {
            self.successful_steals.increment();
        }
    }

    /// Record being stolen from.
    pub fn record_stolen_from(&self) {
        self.stolen_from.increment();
    }

    /// Update queue length.
    pub fn update_queue_length(&self, length: usize) {
        self.queue_length.set(length as u64);
    }

    /// Update CPU utilization.
    pub fn update_cpu_utilization(&self, utilization: f32) {
        self.cpu_utilization.set((utilization * 100.0) as u64);
    }

    /// Get steal success rate.
    pub fn steal_success_rate(&self) -> f64 {
        let attempts = self.steal_attempts.get();
        if attempts == 0 {
            0.0
        } else {
            self.successful_steals.get() as f64 / attempts as f64
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.queue_length.reset();
        self.steal_attempts.reset();
        self.successful_steals.reset();
        self.stolen_from.reset();
        self.cpu_utilization.reset();
    }
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global metrics collector.
#[derive(Debug)]
pub struct Metrics {
    /// Task metrics
    pub tasks: TaskMetrics,
    /// Scheduler metrics by ID
    pub schedulers: BTreeMap<SchedulerId, SchedulerMetrics>,
}

impl Metrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            tasks: TaskMetrics::new(),
            schedulers: BTreeMap::new(),
        }
    }

    /// Get or create scheduler metrics.
    pub fn scheduler(&mut self, id: SchedulerId) -> &mut SchedulerMetrics {
        self.schedulers.entry(id).or_insert_with(SchedulerMetrics::new)
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.tasks.reset();
        for metrics in self.schedulers.values() {
            metrics.reset();
        }
    }

    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            tasks_spawned: self.tasks.spawned.get(),
            tasks_completed: self.tasks.completed.get(),
            tasks_cancelled: self.tasks.cancelled.get(),
            avg_execution_time_us: self.tasks.execution_time.average(),
            avg_wait_time_us: self.tasks.wait_time.average(),
            total_steal_attempts: self.schedulers.values()
                .map(|s| s.steal_attempts.get())
                .sum(),
            total_successful_steals: self.schedulers.values()
                .map(|s| s.successful_steals.get())
                .sum(),
            avg_queue_length: if self.schedulers.is_empty() {
                0.0
            } else {
                self.schedulers.values()
                    .map(|s| s.queue_length.get())
                    .sum::<u64>() as f64 / self.schedulers.len() as f64
            },
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total tasks spawned
    pub tasks_spawned: u64,
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total tasks cancelled
    pub tasks_cancelled: u64,
    /// Average execution time in microseconds
    pub avg_execution_time_us: f64,
    /// Average wait time in microseconds
    pub avg_wait_time_us: f64,
    /// Total steal attempts across all schedulers
    pub total_steal_attempts: u64,
    /// Total successful steals across all schedulers
    pub total_successful_steals: u64,
    /// Average queue length across all schedulers
    pub avg_queue_length: f64,
}

impl core::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Moirai Metrics Snapshot:")?;
        writeln!(f, "  Tasks:")?;
        writeln!(f, "    Spawned: {}", self.tasks_spawned)?;
        writeln!(f, "    Completed: {}", self.tasks_completed)?;
        writeln!(f, "    Cancelled: {}", self.tasks_cancelled)?;
        writeln!(f, "    Avg Execution Time: {:.2}μs", self.avg_execution_time_us)?;
        writeln!(f, "    Avg Wait Time: {:.2}μs", self.avg_wait_time_us)?;
        writeln!(f, "  Work Stealing:")?;
        writeln!(f, "    Total Attempts: {}", self.total_steal_attempts)?;
        writeln!(f, "    Successful: {}", self.total_successful_steals)?;
        writeln!(f, "    Success Rate: {:.2}%", 
            if self.total_steal_attempts > 0 {
                (self.total_successful_steals as f64 / self.total_steal_attempts as f64) * 100.0
            } else {
                0.0
            })?;
        writeln!(f, "  Scheduling:")?;
        writeln!(f, "    Avg Queue Length: {:.2}", self.avg_queue_length)?;
        Ok(())
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
        
        gauge.decrement();
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
        assert_eq!(histogram.average(), 0.0);
        
        histogram.record(10);
        histogram.record(20);
        histogram.record(30);
        
        assert_eq!(histogram.count(), 3);
        assert_eq!(histogram.sum(), 60);
        assert_eq!(histogram.average(), 20.0);
    }

    #[test]
    fn test_task_metrics() {
        let metrics = TaskMetrics::new();
        
        metrics.record_spawn();
        metrics.record_spawn();
        assert_eq!(metrics.spawned.get(), 2);
        
        metrics.record_completion(core::time::Duration::from_millis(1));
        assert_eq!(metrics.completed.get(), 1);
        assert_eq!(metrics.completion_rate(), 0.5);
        
        metrics.record_cancellation();
        assert_eq!(metrics.cancelled.get(), 1);
    }

    #[test]
    fn test_scheduler_metrics() {
        let metrics = SchedulerMetrics::new();
        
        metrics.update_queue_length(5);
        assert_eq!(metrics.queue_length.get(), 5);
        
        metrics.record_steal_attempt(false);
        metrics.record_steal_attempt(true);
        metrics.record_steal_attempt(true);
        
        assert_eq!(metrics.steal_attempts.get(), 3);
        assert_eq!(metrics.successful_steals.get(), 2);
        assert_eq!(metrics.steal_success_rate(), 2.0 / 3.0);
    }
}