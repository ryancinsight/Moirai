//! Scheduler configuration, strategy enums, steal context, and statistics.

use super::traits::SchedulerId;

#[cfg(feature = "std")]
use std::time::SystemTime;

/// Configuration parameters for scheduler behavior.
///
/// This struct contains settings that control how schedulers operate,
/// including work-stealing policies, queue sizes, and performance tuning parameters.
pub struct Config {
    /// Strategy used for work-stealing between schedulers
    pub work_stealing_strategy: WorkStealingStrategy,
    /// Type of queue implementation to use
    pub queue_type: QueueType,
    /// Maximum number of tasks in each scheduler's local queue
    pub max_local_queue_size: usize,
    /// Maximum number of tasks in the global shared queue
    pub max_global_queue_size: usize,
    /// Number of steal attempts before giving up
    pub max_steal_attempts: usize,
    /// Minimum number of tasks before allowing steals
    pub steal_threshold: usize,
    /// Whether to enable detailed performance metrics
    pub enable_metrics: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            work_stealing_strategy: WorkStealingStrategy::default(),
            queue_type: QueueType::ChaseLev,
            max_local_queue_size: 1024,
            max_global_queue_size: 16384,
            max_steal_attempts: 3,
            steal_threshold: 1,
            enable_metrics: true,
        }
    }
}

/// Type alias for backwards compatibility
pub type SchedulerConfig = Config;

/// Strategies for work-stealing between schedulers.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkStealingStrategy {
    /// Random victim selection
    Random {
        /// Maximum number of steal attempts before giving up
        max_attempts: usize,
    },
    /// Round-robin victim selection
    RoundRobin {
        /// Maximum number of steal attempts before giving up
        max_attempts: usize,
    },
    /// Locality-aware victim selection
    LocalityAware {
        /// Maximum number of steal attempts before giving up
        max_attempts: usize,
        /// Weight factor for locality preference (0.0 to 1.0)
        locality_factor: f64,
    },
    /// Load-based victim selection
    LoadBased {
        /// Maximum number of steal attempts before giving up
        max_attempts: usize,
        /// Minimum load difference required to attempt stealing
        min_load_diff: usize,
    },
    /// Adaptive strategy that adjusts based on success rate
    Adaptive {
        /// Base strategy to adapt from
        base_strategy: Box<WorkStealingStrategy>,
        /// Rate at which to adapt the strategy (0.0 to 1.0)
        adaptation_rate: f64,
    },
}

impl Default for WorkStealingStrategy {
    fn default() -> Self {
        Self::Random { max_attempts: 3 }
    }
}

/// Queue implementation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    /// Chase-Lev work-stealing deque
    ChaseLev,
    /// Simple FIFO queue with locks
    SimpleFifo,
    /// Priority queue
    Priority,
    /// Segmented queue for better cache locality
    Segmented,
}

/// Context information for work-stealing operations.
///
/// This struct tracks the state and history of steal attempts to optimize
/// future stealing decisions and avoid repeated failed attempts.
pub struct StealContext {
    /// Number of consecutive failed steal attempts
    pub attempts: usize,
    /// Timestamp of the last successful steal
    pub last_success: Option<SystemTime>,
    /// Ring of recently attempted victim schedulers (bounded, O(1) eviction).
    /// Using VecDeque for O(1) pop_front rotation instead of O(n) Vec::remove(0).
    pub recent_victims: std::collections::VecDeque<SchedulerId>,
    /// Current backoff delay for failed steals
    pub backoff_delay: core::time::Duration,
}

impl Default for StealContext {
    fn default() -> Self {
        Self {
            attempts: 0,
            last_success: None,
            recent_victims: std::collections::VecDeque::new(),
            backoff_delay: core::time::Duration::from_millis(10),
        }
    }
}

/// Performance and operational statistics for scheduler instances.
///
/// This struct provides detailed metrics about scheduler performance,
/// helping with monitoring, debugging, and optimization.
#[derive(Debug, Clone)]
pub struct Stats {
    /// Unique identifier of this scheduler
    pub scheduler_id: SchedulerId,
    /// Total number of tasks scheduled since creation
    pub total_scheduled: u64,
    /// Total number of tasks completed
    pub total_completed: u64,
    /// Number of tasks currently in the queue
    pub current_load: usize,
    /// Peak number of tasks ever queued simultaneously
    pub peak_load: usize,
    /// Number of successful steal operations (tasks stolen by others)
    pub steals_given: u64,
    /// Number of successful steal operations (tasks stolen from others)
    pub steals_taken: u64,
    /// Number of failed steal attempts
    pub steal_failures: u64,
    /// Average time tasks spend in queue (microseconds)
    pub avg_queue_time_us: u64,
    /// Total CPU time spent on scheduling operations
    pub scheduling_time_us: u64,
}
