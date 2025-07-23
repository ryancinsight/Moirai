//! Scheduler trait definitions and work-stealing abstractions.

use crate::{Task, BoxedTask, error::SchedulerResult, Box, Vec};
use core::fmt;

/// The core trait for task schedulers in the Moirai runtime.
pub trait Scheduler: Send + Sync + 'static {
    /// Schedule a task for execution.
    fn schedule_task(&self, task: Box<dyn BoxedTask>) -> SchedulerResult<()>;

    /// Try to get the next task to execute.
    fn next_task(&self) -> SchedulerResult<Option<Box<dyn BoxedTask>>>;

    /// Try to steal work from another scheduler.
    fn try_steal(&self, victim: &dyn Scheduler) -> SchedulerResult<Option<Box<dyn BoxedTask>>>;

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

/// A generic scheduler trait for type-safe task scheduling.
pub trait GenericScheduler: Send + Sync + 'static {
    /// Schedule a task for execution.
    fn schedule<T>(&self, task: T) -> SchedulerResult<()>
    where
        T: Task;
}

/// A unique identifier for schedulers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// Load-based stealing (steal from most loaded)
    LoadBased {
        /// Maximum number of steal attempts
        max_attempts: usize,
        /// Minimum load difference to trigger steal
        min_load_diff: usize,
    },
    /// Adaptive strategy that changes based on runtime conditions
    Adaptive {
        /// Base strategy to start with
        base_strategy: Box<WorkStealingStrategy>,
        /// Adaptation period in milliseconds
        adaptation_period_ms: u64,
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

/// Work stealing context for tracking steal attempts.
#[derive(Debug, Clone)]
pub struct StealContext {
    /// Number of steal attempts made
    pub attempts: usize,
    /// Number of successful steals
    pub successes: usize,
    /// Total items stolen
    pub items_stolen: usize,
    /// Average steal latency in nanoseconds
    pub avg_steal_latency_ns: u64,
}

impl Default for StealContext {
    fn default() -> Self {
        Self {
            attempts: 0,
            successes: 0,
            items_stolen: 0,
            avg_steal_latency_ns: 0,
        }
    }
}

/// Scheduler statistics for monitoring and debugging.
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Scheduler identifier
    pub id: SchedulerId,
    /// Current queue length
    pub queue_length: usize,
    /// Total tasks scheduled
    pub tasks_scheduled: u64,
    /// Total tasks completed
    pub tasks_completed: u64,
    /// Work stealing statistics
    pub steal_context: StealContext,
    /// Average task execution time in microseconds
    pub avg_task_execution_us: f64,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
}

/// Work stealing coordinator that manages steal attempts across schedulers.
pub struct WorkStealingCoordinator {
    schedulers: Vec<Box<dyn Scheduler>>,
    strategy: WorkStealingStrategy,
    stats: Vec<SchedulerStats>,
}

impl WorkStealingCoordinator {
    /// Create a new work stealing coordinator.
    pub fn new(strategy: WorkStealingStrategy) -> Self {
        Self {
            schedulers: Vec::new(),
            strategy,
            stats: Vec::new(),
        }
    }

    /// Register a scheduler with the coordinator.
    pub fn register_scheduler(&mut self, scheduler: Box<dyn Scheduler>) {
        let id = scheduler.id();
        self.schedulers.push(scheduler);
        self.stats.push(SchedulerStats {
            id,
            queue_length: 0,
            tasks_scheduled: 0,
            tasks_completed: 0,
            steal_context: StealContext::default(),
            avg_task_execution_us: 0.0,
            cpu_utilization: 0.0,
        });
    }

    /// Attempt to steal work for the given scheduler.
    pub fn try_steal_for(&self, thief_id: SchedulerId) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        match &self.strategy {
            WorkStealingStrategy::Random { max_attempts } => {
                self.random_steal(thief_id, *max_attempts)
            }
            WorkStealingStrategy::RoundRobin { max_attempts } => {
                self.round_robin_steal(thief_id, *max_attempts)
            }
            WorkStealingStrategy::LocalityAware { max_attempts, locality_factor } => {
                self.locality_aware_steal(thief_id, *max_attempts, *locality_factor)
            }
            WorkStealingStrategy::LoadBased { max_attempts, min_load_diff } => {
                self.load_based_steal(thief_id, *max_attempts, *min_load_diff)
            }
            WorkStealingStrategy::Adaptive { base_strategy, .. } => {
                // For now, use the base strategy
                // In a full implementation, this would adapt based on runtime metrics
                match base_strategy.as_ref() {
                    WorkStealingStrategy::Random { max_attempts } => {
                        self.random_steal(thief_id, *max_attempts)
                    }
                    _ => Ok(None), // Simplified for now
                }
            }
        }
    }

    fn random_steal(&self, thief_id: SchedulerId, max_attempts: usize) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        use core::num::Wrapping;
        
        // Simple pseudo-random selection based on scheduler ID
        let mut seed = Wrapping(thief_id.get() as u32);
        
        for _ in 0..max_attempts {
            if self.schedulers.is_empty() {
                return Ok(None);
            }
            
            // Simple LCG for pseudo-random selection
            seed = seed * Wrapping(1_103_515_245) + Wrapping(12345);
            let victim_idx = (seed.0 as usize) % self.schedulers.len();
            
            let victim = &self.schedulers[victim_idx];
            if victim.id() != thief_id && victim.can_be_stolen_from() {
                if let Some(task) = victim.try_steal(&**victim)? {
                    return Ok(Some(task));
                }
            }
        }
        
        Ok(None)
    }

    fn round_robin_steal(&self, thief_id: SchedulerId, max_attempts: usize) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        let start_idx = thief_id.get() % self.schedulers.len().max(1);
        
        for i in 0..max_attempts.min(self.schedulers.len()) {
            let victim_idx = (start_idx + i) % self.schedulers.len();
            let victim = &self.schedulers[victim_idx];
            
            if victim.id() != thief_id && victim.can_be_stolen_from() {
                if let Some(task) = victim.try_steal(&**victim)? {
                    return Ok(Some(task));
                }
            }
        }
        
        Ok(None)
    }

    fn locality_aware_steal(&self, thief_id: SchedulerId, max_attempts: usize, locality_factor: f32) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        // Simplified locality-aware stealing
        // In a real implementation, this would consider CPU topology
        let candidates: Vec<_> = self.schedulers.iter()
            .enumerate()
            .filter(|(_, s)| s.id() != thief_id && s.can_be_stolen_from())
            .map(|(idx, s)| {
                // Calculate locality score (simplified)
                let distance = ((idx as i32) - (thief_id.get() as i32)).abs() as f32;
                let locality_score = 1.0 / (1.0 + distance * locality_factor);
                (idx, s, locality_score)
            })
            .collect();

        // Sort by locality score (higher is better)
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(core::cmp::Ordering::Equal));

        for (_, victim, _) in sorted_candidates.iter().take(max_attempts) {
            if let Some(task) = victim.try_steal(&***victim)? {
                return Ok(Some(task));
            }
        }

        Ok(None)
    }

    fn load_based_steal(&self, thief_id: SchedulerId, max_attempts: usize, min_load_diff: usize) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        let thief_load = self.schedulers.iter()
            .find(|s| s.id() == thief_id)
            .map(|s| s.load())
            .unwrap_or(0);

        let candidates: Vec<_> = self.schedulers.iter()
            .filter(|s| {
                s.id() != thief_id && 
                s.can_be_stolen_from() && 
                s.load() > thief_load + min_load_diff
            })
            .collect();

        // Sort by load (highest first)
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by(|a, b| b.load().cmp(&a.load()));

        for victim in sorted_candidates.iter().take(max_attempts) {
            if let Some(task) = victim.try_steal(&***victim)? {
                return Ok(Some(task));
            }
        }

        Ok(None)
    }

    /// Get statistics for all schedulers.
    pub fn get_stats(&self) -> &[SchedulerStats] {
        &self.stats
    }

    /// Update statistics for a scheduler.
    pub fn update_stats(&mut self, id: SchedulerId, stats: SchedulerStats) {
        if let Some(existing_stats) = self.stats.iter_mut().find(|s| s.id == id) {
            *existing_stats = stats;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_id() {
        let id = SchedulerId::new(42);
        assert_eq!(id.get(), 42);
        assert_eq!(format!("{id}"), "Scheduler(42)");
    }

    #[test]
    fn test_work_stealing_strategy_default() {
        let strategy = WorkStealingStrategy::default();
        matches!(strategy, WorkStealingStrategy::Random { max_attempts: 3 });
    }

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.max_queue_size, 1024);
        assert!(config.priority_scheduling);
        assert_eq!(config.queue_type, QueueType::ChaseLev);
    }

    #[test]
    fn test_steal_context_default() {
        let ctx = StealContext::default();
        assert_eq!(ctx.attempts, 0);
        assert_eq!(ctx.successes, 0);
        assert_eq!(ctx.items_stolen, 0);
        assert_eq!(ctx.avg_steal_latency_ns, 0);
    }
}