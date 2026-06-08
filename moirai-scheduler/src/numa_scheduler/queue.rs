//! Per-NUMA-node queue internals: `NodeQueue`, `LoadMetrics`, `StealStatistics`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use moirai_core::{Priority, ScheduledTask};

/// Per-NUMA-node task queue.
pub(super) struct NodeQueue {
    /// Node ID
    pub(super) _node_id: usize,
    /// Local task deque (using existing Chase-Lev implementation)
    pub(super) _local_queue: crate::ChaseLevDeque<ScheduledTask>,
    /// Priority queues for different task priorities
    pub(super) priority_queues: [crate::ChaseLevDeque<ScheduledTask>; 4],
    /// Queue load metrics
    pub(super) load_metrics: LoadMetrics,
    /// Lock for exclusive operations
    pub(super) _exclusive_lock: std::sync::Mutex<()>,
}

/// Load metrics for a node queue.
#[derive(Debug)]
pub(super) struct LoadMetrics {
    /// Total tasks processed
    pub(super) tasks_processed: AtomicUsize,
    /// Current queue length
    pub(super) current_load: AtomicUsize,
    /// Average processing time
    pub(super) _avg_processing_time_ns: AtomicUsize,
    /// Last update timestamp
    pub(super) _last_update: std::sync::Mutex<Instant>,
}

/// Statistics for steal operations.
#[derive(Debug)]
pub struct StealStatistics {
    /// Successful steals from same NUMA node
    pub(super) same_numa_steals: AtomicUsize,
    /// Successful steals from different NUMA node
    pub(super) cross_numa_steals: AtomicUsize,
    /// Failed steal attempts
    pub(super) failed_steals: AtomicUsize,
    /// Total steal attempts
    pub(super) total_attempts: AtomicUsize,
    /// Average steal latency (nanoseconds)
    pub(super) avg_steal_latency_ns: AtomicUsize,
}

impl NodeQueue {
    pub(super) fn new(node_id: usize) -> Self {
        Self {
            _node_id: node_id,
            _local_queue: crate::ChaseLevDeque::new(1024),
            priority_queues: [
                crate::ChaseLevDeque::new(1024), // Critical
                crate::ChaseLevDeque::new(1024), // High
                crate::ChaseLevDeque::new(1024), // Normal
                crate::ChaseLevDeque::new(1024), // Low
            ],
            load_metrics: LoadMetrics {
                tasks_processed: AtomicUsize::new(0),
                current_load: AtomicUsize::new(0),
                _avg_processing_time_ns: AtomicUsize::new(0),
                _last_update: std::sync::Mutex::new(Instant::now()),
            },
            _exclusive_lock: std::sync::Mutex::new(()),
        }
    }

    pub(super) fn push_task(&self, task: ScheduledTask, priority: Priority) {
        let queue_index = match priority {
            Priority::Critical => 0,
            Priority::High => 1,
            Priority::Normal => 2,
            Priority::Low => 3,
        };

        self.priority_queues[queue_index].push(task);
        self.load_metrics
            .current_load
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn pop_task(&self) -> Option<ScheduledTask> {
        // Try priority queues in order (highest first)
        for queue in &self.priority_queues {
            if let Some(task) = queue.pop() {
                self.load_metrics
                    .current_load
                    .fetch_sub(1, Ordering::Relaxed);
                self.load_metrics
                    .tasks_processed
                    .fetch_add(1, Ordering::Relaxed);
                return Some(task);
            }
        }
        None
    }

    pub(super) fn steal_task(&self) -> Option<ScheduledTask> {
        // Try to steal from priority queues (lower priority first for fairness)
        for queue in self.priority_queues.iter().rev() {
            if let crate::StealResult::Success(task) = queue.steal() {
                self.load_metrics
                    .current_load
                    .fetch_sub(1, Ordering::Relaxed);
                return Some(task);
            }
        }
        None
    }

    pub(super) fn current_load(&self) -> usize {
        self.load_metrics.current_load.load(Ordering::Relaxed)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.current_load() == 0
    }
}
