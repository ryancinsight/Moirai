//! Metrics and observability traits for Moirai.
//!
//! This module provides the foundational traits and types for collecting
//! performance metrics and observability data across the Moirai runtime.

use crate::{TaskId, Priority};
use core::time::{Duration, Instant};

/// Core trait for collecting metrics.
pub trait MetricsCollector: Send + Sync + 'static {
    /// Record when a task is spawned.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting spawn performance
    /// - Called on the spawning thread
    fn record_task_spawn(&self, task_id: TaskId, priority: Priority, spawn_time: Instant) {
        let _ = (task_id, priority, spawn_time); // Default: no-op
    }

    /// Record when a task starts executing.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting task startup
    /// - Called on the executing thread
    fn record_task_start(&self, task_id: TaskId, start_time: Instant) {
        let _ = (task_id, start_time); // Default: no-op
    }

    /// Record when a task completes.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting task completion
    /// - Called on the executing thread
    fn record_task_complete(&self, task_id: TaskId, completion_time: Instant, success: bool) {
        let _ = (task_id, completion_time, success); // Default: no-op
    }

    /// Record a work steal attempt.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting steal performance
    /// - Called on the stealing thread
    fn record_steal_attempt(&self, stealer_id: usize, victim_id: usize, success: bool, items_stolen: usize) {
        let _ = (stealer_id, victim_id, success, items_stolen); // Default: no-op
    }

    /// Record message send operation.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting message performance
    /// - Called on the sending thread
    fn record_message_send(&self, size_bytes: usize, transport_type: &str, latency: Duration) {
        let _ = (size_bytes, transport_type, latency); // Default: no-op
    }

    /// Record memory allocation.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting allocation performance
    /// - May be called very frequently
    fn record_allocation(&self, size_bytes: usize, allocation_type: &str) {
        let _ = (size_bytes, allocation_type); // Default: no-op
    }

    /// Record memory deallocation.
    /// 
    /// # Performance Characteristics
    /// - Should be O(1) to avoid impacting deallocation performance
    /// - May be called very frequently
    fn record_deallocation(&self, size_bytes: usize, allocation_type: &str) {
        let _ = (size_bytes, allocation_type); // Default: no-op
    }

    /// Flush any pending metrics.
    /// 
    /// # Behavior Guarantees
    /// - Ensures all recorded metrics are persisted/transmitted
    /// - May block briefly for I/O operations
    fn flush(&self) {
        // Default: no-op
    }
}

/// A no-op metrics collector that does nothing.
/// 
/// This is useful as a default implementation when metrics are disabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpMetricsCollector;

impl MetricsCollector for NoOpMetricsCollector {
    // All methods use default implementations (no-op)
}

/// Metrics configuration.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Whether to enable task-level metrics
    pub enable_task_metrics: bool,
    /// Whether to enable work-stealing metrics
    pub enable_steal_metrics: bool,
    /// Whether to enable transport metrics
    pub enable_transport_metrics: bool,
    /// Whether to enable memory metrics
    pub enable_memory_metrics: bool,
    /// How often to flush metrics (in milliseconds)
    pub flush_interval_ms: u64,
    /// Maximum number of metrics to buffer before flushing
    pub buffer_size: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_task_metrics: true,
            enable_steal_metrics: true,
            enable_transport_metrics: true,
            enable_memory_metrics: false, // Can be expensive
            flush_interval_ms: 1000,
            buffer_size: 10000,
        }
    }
}

/// Aggregated metrics for a time period.
#[derive(Debug, Clone, Default)]
pub struct AggregatedMetrics {
    /// Time period these metrics cover
    pub period_start: Option<Instant>,
    /// End of the time period
    pub period_end: Option<Instant>,
    
    // Task metrics
    /// Total tasks spawned
    pub tasks_spawned: u64,
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Total tasks failed
    pub tasks_failed: u64,
    /// Average task execution time (microseconds)
    pub avg_task_execution_time_us: f64,
    /// 95th percentile task execution time (microseconds)
    pub p95_task_execution_time_us: f64,
    /// Maximum task execution time (microseconds)
    pub max_task_execution_time_us: u64,
    
    // Work stealing metrics
    /// Total steal attempts
    pub steal_attempts: u64,
    /// Successful steal attempts
    pub successful_steals: u64,
    /// Total items stolen
    pub items_stolen: u64,
    
    // Transport metrics
    /// Total messages sent
    pub messages_sent: u64,
    /// Total bytes transmitted
    pub bytes_transmitted: u64,
    /// Average message latency (microseconds)
    pub avg_message_latency_us: f64,
    
    // Memory metrics
    /// Total allocations
    pub allocations: u64,
    /// Total deallocations
    pub deallocations: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: u64,
    /// Current memory usage (bytes)
    pub current_memory_usage: u64,
}

impl AggregatedMetrics {
    /// Calculate the steal success rate.
    pub fn steal_success_rate(&self) -> f64 {
        if self.steal_attempts == 0 {
            0.0
        } else {
            self.successful_steals as f64 / self.steal_attempts as f64
        }
    }

    /// Calculate the task completion rate.
    pub fn task_completion_rate(&self) -> f64 {
        if self.tasks_spawned == 0 {
            0.0
        } else {
            self.tasks_completed as f64 / self.tasks_spawned as f64
        }
    }

    /// Calculate the task failure rate.
    pub fn task_failure_rate(&self) -> f64 {
        if self.tasks_spawned == 0 {
            0.0
        } else {
            self.tasks_failed as f64 / self.tasks_spawned as f64
        }
    }

    /// Calculate the average items stolen per successful steal.
    pub fn avg_items_per_steal(&self) -> f64 {
        if self.successful_steals == 0 {
            0.0
        } else {
            self.items_stolen as f64 / self.successful_steals as f64
        }
    }

    /// Calculate throughput in tasks per second.
    pub fn task_throughput_per_second(&self) -> f64 {
        if let (Some(start), Some(end)) = (self.period_start, self.period_end) {
            let duration_secs = end.duration_since(start).as_secs_f64();
            if duration_secs > 0.0 {
                self.tasks_completed as f64 / duration_secs
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Calculate message throughput in messages per second.
    pub fn message_throughput_per_second(&self) -> f64 {
        if let (Some(start), Some(end)) = (self.period_start, self.period_end) {
            let duration_secs = end.duration_since(start).as_secs_f64();
            if duration_secs > 0.0 {
                self.messages_sent as f64 / duration_secs
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Calculate bandwidth in bytes per second.
    pub fn bandwidth_bytes_per_second(&self) -> f64 {
        if let (Some(start), Some(end)) = (self.period_start, self.period_end) {
            let duration_secs = end.duration_since(start).as_secs_f64();
            if duration_secs > 0.0 {
                self.bytes_transmitted as f64 / duration_secs
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

/// Trait for objects that can provide aggregated metrics.
pub trait MetricsProvider: Send + Sync {
    /// Get metrics for the specified time period.
    /// 
    /// # Parameters
    /// - `start`: Start of the time period (None for all-time)
    /// - `end`: End of the time period (None for up to now)
    /// 
    /// # Performance Characteristics
    /// - May involve aggregation computation
    /// - Suitable for periodic reporting
    fn get_metrics(&self, start: Option<Instant>, end: Option<Instant>) -> AggregatedMetrics;

    /// Get real-time metrics (last few seconds).
    /// 
    /// # Performance Characteristics
    /// - O(1) operation for cached metrics
    /// - Suitable for dashboards and monitoring
    fn get_realtime_metrics(&self) -> AggregatedMetrics;

    /// Reset all metrics counters.
    /// 
    /// # Behavior Guarantees
    /// - Atomically resets all counters
    /// - Does not affect ongoing operations
    fn reset_metrics(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();
        assert!(config.enable_task_metrics);
        assert!(config.enable_steal_metrics);
        assert!(config.enable_transport_metrics);
        assert!(!config.enable_memory_metrics);
        assert_eq!(config.flush_interval_ms, 1000);
        assert_eq!(config.buffer_size, 10000);
    }

    #[test]
    fn test_aggregated_metrics_calculations() {
        let mut metrics = AggregatedMetrics::default();
        metrics.tasks_spawned = 1000;
        metrics.tasks_completed = 950;
        metrics.tasks_failed = 50;
        metrics.steal_attempts = 200;
        metrics.successful_steals = 150;
        metrics.items_stolen = 300;

        assert_eq!(metrics.steal_success_rate(), 0.75);
        assert_eq!(metrics.task_completion_rate(), 0.95);
        assert_eq!(metrics.task_failure_rate(), 0.05);
        assert_eq!(metrics.avg_items_per_steal(), 2.0);
    }

    #[test]
    fn test_throughput_calculations() {
        let start = Instant::now();
        let end = start + Duration::from_secs(10);
        
        let mut metrics = AggregatedMetrics::default();
        metrics.period_start = Some(start);
        metrics.period_end = Some(end);
        metrics.tasks_completed = 1000;
        metrics.messages_sent = 500;
        metrics.bytes_transmitted = 1024000;

        assert_eq!(metrics.task_throughput_per_second(), 100.0);
        assert_eq!(metrics.message_throughput_per_second(), 50.0);
        assert_eq!(metrics.bandwidth_bytes_per_second(), 102400.0);
    }

    #[test]
    fn test_no_op_metrics_collector() {
        let collector = NoOpMetricsCollector;
        
        // These should all be no-ops and not panic
        collector.record_task_spawn(TaskId::new(1), Priority::Normal, Instant::now());
        collector.record_task_start(TaskId::new(1), Instant::now());
        collector.record_task_complete(TaskId::new(1), Instant::now(), true);
        collector.record_steal_attempt(0, 1, true, 5);
        collector.record_message_send(1024, "tcp", Duration::from_millis(10));
        collector.record_allocation(4096, "task");
        collector.record_deallocation(4096, "task");
        collector.flush();
    }
}