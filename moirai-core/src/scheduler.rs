//! Scheduler trait definitions and work-stealing abstractions.

use crate::{Task, TaskId, Priority, error::SchedulerResult};
use core::fmt;

/// The core trait for task schedulers in the Moirai runtime.
pub trait Scheduler: Send + Sync + 'static {
    /// Schedule a task for execution.
    fn schedule<T>(&self, task: T) -> SchedulerResult<()>
    where
        T: Task;

    /// Try to get the next task to execute.
    fn next_task(&self) -> SchedulerResult<Option<Box<dyn Task<Output = ()>>>>;

    /// Try to steal work from another scheduler.
    fn try_steal(&self, victim: &dyn Scheduler) -> SchedulerResult<Option<Box<dyn Task<Output = ()>>>>;

    /// Get the current load (number of pending tasks).
    fn load(&self) -> usize;

    /// Check if the scheduler is idle.
    fn is_idle(&self) -> bool {
        self.load() == 0
    }

    /// Get the scheduler's unique identifier.
    fn id(&self) -> SchedulerId;

    /// Check if this scheduler can be stolen from.
    fn can_be_stolen_from(&self) -> bool {
        self.load() > 1
    }
}

/// A unique identifier for schedulers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchedulerId(usize);

impl SchedulerId {
    /// Create a new scheduler ID.
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for SchedulerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scheduler({})", self.0)
    }
}

/// Configuration for scheduler behavior.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum queue size before rejecting tasks
    pub max_queue_size: usize,
    /// Whether to use priority-based scheduling
    pub priority_scheduling: bool,
    /// Work stealing configuration
    pub work_stealing: WorkStealingStrategy,
    /// Queue implementation to use
    pub queue_type: QueueType,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1024,
            priority_scheduling: true,
            work_stealing: WorkStealingStrategy::default(),
            queue_type: QueueType::ChaseLev,
        }
    }
}

/// Work stealing strategies.
#[derive(Debug, Clone)]
pub enum WorkStealingStrategy {
    /// Random victim selection
    Random {
        /// Maximum number of steal attempts
        max_attempts: usize,
    },
    /// Round-robin victim selection
    RoundRobin {
        /// Maximum number of steal attempts
        max_attempts: usize,
    },
    /// Locality-aware stealing (prefer nearby cores)
    LocalityAware {
        /// Maximum number of steal attempts
        max_attempts: usize,
        /// Locality preference factor (0.0 = no preference, 1.0 = strong preference)
        locality_factor: f32,
    },
}

impl Default for WorkStealingStrategy {
    fn default() -> Self {
        Self::Random { max_attempts: 3 }
    }
}

/// Types of task queues available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    /// Chase-Lev work-stealing deque
    ChaseLev,
    /// FIFO queue
    Fifo,
    /// LIFO stack
    Lifo,
    /// Priority queue
    Priority,
}

/// Statistics about scheduler performance.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total number of tasks scheduled
    pub tasks_scheduled: u64,
    /// Total number of tasks executed
    pub tasks_executed: u64,
    /// Number of successful steals
    pub successful_steals: u64,
    /// Number of failed steal attempts
    pub failed_steals: u64,
    /// Number of times this scheduler was stolen from
    pub stolen_from: u64,
    /// Current queue length
    pub current_queue_length: usize,
    /// Maximum queue length seen
    pub max_queue_length: usize,
    /// Average task wait time (microseconds)
    pub avg_wait_time: f64,
}

impl SchedulerStats {
    /// Calculate the steal success rate.
    pub fn steal_success_rate(&self) -> f64 {
        let total_attempts = self.successful_steals + self.failed_steals;
        if total_attempts == 0 {
            0.0
        } else {
            self.successful_steals as f64 / total_attempts as f64
        }
    }

    /// Calculate the task throughput (tasks per second).
    pub fn throughput(&self, elapsed_seconds: f64) -> f64 {
        if elapsed_seconds <= 0.0 {
            0.0
        } else {
            self.tasks_executed as f64 / elapsed_seconds
        }
    }

    /// Calculate the queue utilization.
    pub fn queue_utilization(&self, max_capacity: usize) -> f64 {
        if max_capacity == 0 {
            0.0
        } else {
            self.current_queue_length as f64 / max_capacity as f64
        }
    }
}

/// A trait for objects that can provide scheduler statistics.
pub trait SchedulerMetrics {
    /// Get current scheduler statistics.
    fn stats(&self) -> SchedulerStats;

    /// Reset statistics counters.
    fn reset_stats(&self);
}

/// Work stealing victim selection strategy.
pub trait VictimSelector: Send + Sync {
    /// Select a victim scheduler to steal from.
    fn select_victim(&self, schedulers: &[SchedulerId], current: SchedulerId) -> Option<SchedulerId>;

    /// Update victim selection state after a steal attempt.
    fn update_after_steal(&self, victim: SchedulerId, success: bool);
}

/// Random victim selector.
#[derive(Debug)]
pub struct RandomVictimSelector {
    /// Random number generator state
    state: core::sync::atomic::AtomicU64,
}

impl RandomVictimSelector {
    /// Create a new random victim selector.
    pub fn new() -> Self {
        Self {
            state: core::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Generate a pseudo-random number using xorshift.
    fn next_random(&self) -> u64 {
        use core::sync::atomic::Ordering;
        
        let current = self.state.load(Ordering::Relaxed);
        let mut x = current;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state.store(x, Ordering::Relaxed);
        x
    }
}

impl Default for RandomVictimSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl VictimSelector for RandomVictimSelector {
    fn select_victim(&self, schedulers: &[SchedulerId], current: SchedulerId) -> Option<SchedulerId> {
        if schedulers.len() <= 1 {
            return None;
        }

        let candidates: Vec<_> = schedulers.iter()
            .filter(|&&id| id != current)
            .copied()
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let index = (self.next_random() as usize) % candidates.len();
        Some(candidates[index])
    }

    fn update_after_steal(&self, _victim: SchedulerId, _success: bool) {
        // Random selection doesn't need to update state
    }
}

/// Round-robin victim selector.
#[derive(Debug)]
pub struct RoundRobinVictimSelector {
    /// Current position in the round-robin
    position: core::sync::atomic::AtomicUsize,
}

impl RoundRobinVictimSelector {
    /// Create a new round-robin victim selector.
    pub fn new() -> Self {
        Self {
            position: core::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Default for RoundRobinVictimSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl VictimSelector for RoundRobinVictimSelector {
    fn select_victim(&self, schedulers: &[SchedulerId], current: SchedulerId) -> Option<SchedulerId> {
        if schedulers.len() <= 1 {
            return None;
        }

        use core::sync::atomic::Ordering;

        let candidates: Vec<_> = schedulers.iter()
            .filter(|&&id| id != current)
            .copied()
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let pos = self.position.fetch_add(1, Ordering::Relaxed);
        let index = pos % candidates.len();
        Some(candidates[index])
    }

    fn update_after_steal(&self, _victim: SchedulerId, _success: bool) {
        // Round-robin doesn't need to update state based on success
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_id() {
        let id = SchedulerId::new(42);
        assert_eq!(id.get(), 42);
        assert_eq!(format!("{}", id), "Scheduler(42)");
    }

    #[test]
    fn test_scheduler_stats() {
        let mut stats = SchedulerStats::default();
        stats.successful_steals = 80;
        stats.failed_steals = 20;
        stats.tasks_executed = 1000;
        stats.current_queue_length = 50;

        assert_eq!(stats.steal_success_rate(), 0.8);
        assert_eq!(stats.throughput(10.0), 100.0);
        assert_eq!(stats.queue_utilization(100), 0.5);
    }

    #[test]
    fn test_random_victim_selector() {
        let selector = RandomVictimSelector::new();
        let schedulers = vec![
            SchedulerId::new(0),
            SchedulerId::new(1),
            SchedulerId::new(2),
        ];
        let current = SchedulerId::new(0);

        let victim = selector.select_victim(&schedulers, current);
        assert!(victim.is_some());
        assert_ne!(victim.unwrap(), current);
    }

    #[test]
    fn test_round_robin_victim_selector() {
        let selector = RoundRobinVictimSelector::new();
        let schedulers = vec![
            SchedulerId::new(0),
            SchedulerId::new(1),
            SchedulerId::new(2),
        ];
        let current = SchedulerId::new(0);

        let victim1 = selector.select_victim(&schedulers, current);
        let victim2 = selector.select_victim(&schedulers, current);
        
        assert!(victim1.is_some());
        assert!(victim2.is_some());
        assert_ne!(victim1.unwrap(), current);
        assert_ne!(victim2.unwrap(), current);
    }
}