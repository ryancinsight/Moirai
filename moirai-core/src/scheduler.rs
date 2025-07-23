//! Scheduler trait definitions and work-stealing abstractions.

use crate::{Task, BoxedTask, error::SchedulerResult, Box, Vec};
use core::fmt;
use std::time::SystemTime;
use std::num::Wrapping;
use crate::task::Closure;

/// Core scheduling interface for task distribution and execution.
///
/// This trait defines the fundamental operations that all scheduler implementations
/// must support for managing task queues and work distribution.
pub trait Scheduler: Send + Sync + 'static {
    /// Adds a task to the scheduler's queue for execution.
    ///
    /// # Arguments
    /// * `task` - The boxed task to be scheduled
    ///
    /// # Returns
    /// `Ok(())` if the task was successfully queued, `Err` otherwise.
    ///
    /// # Errors
    /// Returns `SchedulerError` in the following cases:
    /// - `QueueFull` if the scheduler's queue is at capacity
    /// - `ShutdownInProgress` if the scheduler is being shut down
    /// - `InvalidTask` if the task is malformed or cannot be executed
    fn schedule_task(&self, task: Box<dyn BoxedTask>) -> SchedulerResult<()>;

    /// Retrieves the next available task for execution.
    ///
    /// # Returns
    /// `Ok(Some(task))` if a task is available, `Ok(None)` if the queue is empty.
    ///
    /// # Errors
    /// Returns `SchedulerError` if there's an internal error accessing the queue
    /// or if the scheduler is in an invalid state.
    fn next_task(&self) -> SchedulerResult<Option<Box<dyn BoxedTask>>>;

    /// Attempts to steal a task from another scheduler (work-stealing).
    ///
    /// # Arguments
    /// * `victim` - The scheduler to attempt stealing from
    ///
    /// # Returns
    /// `Ok(Some(task))` if a task was successfully stolen, `Ok(None)` if no tasks available.
    ///
    /// # Errors
    /// Returns `SchedulerError` if the steal operation fails due to:
    /// - Lock contention or synchronization issues
    /// - Invalid victim scheduler state
    /// - Internal queue corruption
    fn try_steal(&self, victim: &dyn Scheduler) -> SchedulerResult<Option<Box<dyn BoxedTask>>>;

    /// Returns the current number of queued tasks.
    fn load(&self) -> usize;

    /// Returns a unique identifier for this scheduler instance.
    fn id(&self) -> SchedulerId;

    /// Returns whether this scheduler can have tasks stolen from it.
    ///
    /// # Returns
    /// `true` if the scheduler has stealable tasks, `false` otherwise.
    fn can_be_stolen_from(&self) -> bool {
        self.load() > 0
    }
}

/// Generic scheduling interface with type-safe task handling.
///
/// This trait provides a higher-level interface for schedulers that can work
/// with specific task types while maintaining type safety.
pub trait Generic: Send + Sync + 'static {
    /// Schedules a typed task for execution.
    ///
    /// # Arguments
    /// * `task` - The task to schedule
    ///
    /// # Returns
    /// `Ok(())` if the task was successfully scheduled.
    ///
    /// # Errors
    /// Returns `SchedulerError` if the task cannot be scheduled due to:
    /// - Resource constraints (queue full, memory limits)
    /// - Scheduler shutdown or invalid state
    /// - Task validation failures
    fn schedule<T>(&self, task: T) -> SchedulerResult<()>
    where
        T: Task;
}

/// A unique identifier for schedulers within the work-stealing system.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchedulerId(usize);

impl SchedulerId {
    /// Creates a new scheduler ID.
    ///
    /// # Arguments
    /// * `id` - The numeric identifier for this scheduler
    ///
    /// # Returns
    /// A new scheduler ID instance
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    ///
    /// # Returns
    /// The numeric identifier for this scheduler
    #[must_use]
    pub const fn get(&self) -> usize {
        self.0
    }
}

impl fmt::Display for SchedulerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scheduler({})", self.0)
    }
}

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
            max_global_queue_size: 1024,
            max_steal_attempts: 3,
            steal_threshold: 10,
            enable_metrics: true,
        }
    }
}

/// Type alias for backward compatibility with existing code.
pub type SchedulerConfig = Config;

/// Work stealing strategies define how schedulers attempt to balance load.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkStealingStrategy {
    /// Random victim selection with configurable attempts
    Random { 
        /// Maximum number of steal attempts before giving up
        max_attempts: usize 
    },
    /// Round-robin victim selection
    RoundRobin { 
        /// Maximum number of steal attempts before giving up
        max_attempts: usize 
    },
    /// Prefer victims with better locality (e.g., same CPU socket)
    LocalityAware { 
        /// Maximum number of steal attempts before giving up
        max_attempts: usize, 
        /// Factor influencing locality preference (0.0 = no preference, 1.0 = strong preference)
        locality_factor: f32 
    },
    /// Steal from schedulers with significantly higher load
    LoadBased { 
        /// Maximum number of steal attempts before giving up
        max_attempts: usize, 
        /// Minimum load difference required to trigger stealing
        min_load_diff: usize 
    },
    /// Adaptive strategy that changes based on runtime metrics
    Adaptive { 
        /// Base strategy to use as foundation for adaptation
        base_strategy: Box<WorkStealingStrategy>,
        /// Time interval between strategy adaptations
        adaptation_interval: core::time::Duration,
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
    /// List of recently attempted victim schedulers
    pub recent_victims: Vec<SchedulerId>,
    /// Current backoff delay for failed steals
    pub backoff_delay: core::time::Duration,
}

impl Default for StealContext {
    fn default() -> Self {
        Self {
            attempts: 0,
            last_success: None,
            recent_victims: Vec::new(),
            backoff_delay: core::time::Duration::from_millis(10),
        }
    }
}

/// Performance and operational statistics for scheduler instances.
///
/// This struct provides detailed metrics about scheduler performance,
/// helping with monitoring, debugging, and optimization.
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

/// Work stealing coordinator that manages steal attempts across schedulers.
pub struct WorkStealingCoordinator {
    schedulers: Vec<Box<dyn Scheduler>>,
    strategy: WorkStealingStrategy,
    stats: Vec<Stats>,
}

impl WorkStealingCoordinator {
    /// Creates a new work-stealing coordinator with the specified strategy.
    #[must_use]
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
        self.stats.push(Stats {
            scheduler_id: id,
            total_scheduled: 0,
            total_completed: 0,
            current_load: 0,
            peak_load: 0,
            steals_given: 0,
            steals_taken: 0,
            steal_failures: 0,
            avg_queue_time_us: 0,
            scheduling_time_us: 0,
        });
    }

    /// Attempt to steal tasks from other schedulers.
    ///
    /// # Arguments
    /// * `thief_id` - The ID of the scheduler attempting to steal work
    /// * `context` - Context information for the steal attempt
    ///
    /// # Returns
    /// `Ok(Some(task))` if a task was successfully stolen, `Ok(None)` if no tasks available.
    ///
    /// # Errors
    /// Returns `SchedulerError` if the steal attempt failed due to:
    /// - System constraints or resource exhaustion
    /// - Invalid scheduler configuration
    /// - Internal synchronization failures
    ///
    /// # Implementation Status
    /// **WARNING: This is a placeholder implementation that does not perform actual work stealing.**
    /// 
    /// The current implementation simulates successful steals by creating empty tasks.
    /// A production implementation would need to:
    /// 1. Access victim scheduler's actual task queues
    /// 2. Implement lock-free or low-contention stealing algorithms
    /// 3. Handle queue synchronization and memory ordering
    /// 4. Provide backoff strategies for failed steal attempts
    /// 5. Maintain work-stealing statistics and metrics
    pub fn steal_task(&self, thief_id: SchedulerId, context: &mut StealContext) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        // Find potential victims for work stealing
        let victims = self.select_victims(thief_id);
        
        if victims.is_empty() {
            context.attempts += 1;
            return Ok(None);
        }

        // Try to steal from each victim
        for victim_id in victims {
            if let Some(scheduler) = self.schedulers.iter().find(|s| s.id() == victim_id) {
                // TODO: CRITICAL - Implement actual work stealing
                // This placeholder implementation does not perform real work stealing
                // and would result in a non-functional scheduler in production.
                //
                // A real implementation would:
                // 1. Access the victim's task queue (local deque, global queue, etc.)
                // 2. Attempt to steal from the back/front of the queue atomically
                // 3. Handle contention with the victim scheduler
                // 4. Respect task affinity and stealing policies
                // 5. Update stealing statistics and context
                
                if scheduler.load() > 0 {
                    // PLACEHOLDER: This creates a fake task instead of stealing a real one
                    context.attempts = 0; // Reset attempts on success
                    context.last_success = Some(SystemTime::now());
                    
                    // Create a placeholder task (NOT a real stolen task)
                    let placeholder_task = Closure::new(|| {
                        // This is a placeholder - real stolen tasks would have actual work
                    }, crate::TaskContext::new(crate::TaskId::new(0)));
                    
                    return Ok(Some(Box::new(placeholder_task) as Box<dyn BoxedTask>));
                }
            }
        }

        context.attempts += 1;
        Ok(None)
    }

    fn select_victims(&self, thief_id: SchedulerId) -> Vec<SchedulerId> {
        let mut victims = Vec::new();
        
        match &self.strategy {
            WorkStealingStrategy::Random { max_attempts } => {
                // Use a simple PRNG for victim selection
                #[allow(clippy::cast_possible_truncation)]
                let mut seed = Wrapping(thief_id.get() as u32);
                
                for scheduler in &self.schedulers {
                    if scheduler.id() != thief_id && scheduler.load() > 0 {
                        // Simple linear congruential generator
                        seed = seed * Wrapping(1_103_515_245) + Wrapping(12_345);
                        if (seed.0 % 3) == 0 {  // ~33% selection probability
                            victims.push(scheduler.id());
                        }
                        if victims.len() >= *max_attempts {
                            break;
                        }
                    }
                }
            }
            WorkStealingStrategy::RoundRobin { max_attempts } => {
                // Simple round-robin selection
                for (i, scheduler) in self.schedulers.iter().enumerate() {
                    if scheduler.id() != thief_id && scheduler.load() > 0 {
                        victims.push(scheduler.id());
                        if victims.len() >= *max_attempts {
                            break;
                        }
                    }
                    if i >= *max_attempts {
                        break;
                    }
                }
            }
            WorkStealingStrategy::LocalityAware { max_attempts, locality_factor: _ } => {
                // For now, just use round-robin (locality awareness would require more context)
                for (i, scheduler) in self.schedulers.iter().enumerate() {
                    if scheduler.id() != thief_id && scheduler.load() > 0 {
                        victims.push(scheduler.id());
                        if victims.len() >= *max_attempts {
                            break;
                        }
                    }
                    if i >= *max_attempts {
                        break;
                    }
                }
            }
            WorkStealingStrategy::LoadBased { max_attempts, min_load_diff: _ } => {
                // Select victims based on their current load
                let thief_load = self.schedulers.iter()
                    .find(|s| s.id() == thief_id)
                    .map_or(0, |s| s.load());
                
                // Get candidates with higher load than the thief
                let candidates: Vec<_> = self.schedulers.iter()
                    .filter(|s| s.id() != thief_id && s.load() > thief_load)
                    .collect();
                
                if !candidates.is_empty() {
                    // Sort by load (highest first) using sort_by_key
                    let mut sorted_candidates = candidates;
                    sorted_candidates.sort_by_key(|b| std::cmp::Reverse(b.load()));
                    
                    // Take up to max_attempts victims
                    for scheduler in sorted_candidates.into_iter().take(*max_attempts) {
                        victims.push(scheduler.id());
                    }
                }
            }
            WorkStealingStrategy::Adaptive { base_strategy, .. } => {
                // Use the base strategy for now (adaptive logic would require more state)
                if let WorkStealingStrategy::Random { max_attempts } = base_strategy.as_ref() {
                    #[allow(clippy::cast_possible_truncation)]
                    let mut seed = Wrapping(thief_id.get() as u32);
                    for _ in 0..*max_attempts {
                        if self.schedulers.is_empty() {
                            break;
                        }
                        seed = seed * Wrapping(1_103_515_245) + Wrapping(12_345);
                        let victim_idx = (seed.0 as usize) % self.schedulers.len();
                        if let Some(scheduler) = self.schedulers.get(victim_idx) {
                            if scheduler.id() != thief_id && scheduler.load() > 0 {
                                victims.push(scheduler.id());
                            }
                        }
                    }
                }
            }
        }
        
        victims
    }

    #[allow(dead_code)]
    fn find_best_victim(&self, thief_id: SchedulerId) -> Option<SchedulerId> {
        let thief_load = self.schedulers.iter()
            .find(|s| s.id() == thief_id)
            .map_or(0, |s| s.load());

        // Find schedulers with significantly higher load
        let mut candidates: Vec<_> = self.schedulers
            .iter()
            .filter(|s| s.id() != thief_id && s.load() > thief_load + 2)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort by load (highest first) and return the busiest
        candidates.sort_by_key(|b| std::cmp::Reverse(b.load()));
        candidates.first().map(|s| s.id())
    }

    /// Returns statistics for all registered schedulers.
    #[must_use]
    pub fn get_stats(&self) -> &[Stats] {
        // Note: This is a simplified implementation
        // In a real implementation, this would return actual statistics
        &[]
    }

    /// Update statistics for a scheduler.
    pub fn update_stats(&mut self, id: SchedulerId, stats: Stats) {
        if let Some(existing_stats) = self.stats.iter_mut().find(|s| s.scheduler_id == id) {
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
        let config = Config::default();
        assert_eq!(config.max_local_queue_size, 1024);
        assert!(config.enable_metrics);
        assert_eq!(config.work_stealing_strategy, WorkStealingStrategy::default());
    }

    #[test]
    fn test_steal_context_default() {
        let ctx = StealContext::default();
        assert_eq!(ctx.attempts, 0);
        assert!(ctx.last_success.is_none());
        assert!(ctx.recent_victims.is_empty());
        assert_eq!(ctx.backoff_delay, core::time::Duration::from_millis(10)); // Default backoff
    }
}