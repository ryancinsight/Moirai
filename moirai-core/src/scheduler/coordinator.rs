//! Work-stealing coordinator managing steal attempts across all schedulers.

use super::config::{Stats, StealContext, WorkStealingStrategy};
use super::deque::WorkStealingDeque;
use super::task::ScheduledTask;
use super::traits::{Scheduler, SchedulerId};
use crate::error::{SchedulerError, SchedulerResult};
use crate::platform::*;
use core::cmp::Reverse;
use core::num::Wrapping;

#[cfg(feature = "std")]
use std::time::SystemTime;

/// Work stealing coordinator that manages steal attempts across schedulers.
///
/// This implementation now uses work-stealing deques for better performance.
pub struct WorkStealingCoordinator<S: Scheduler> {
    schedulers: Vec<S>,
    strategy: WorkStealingStrategy,
    stats: Arc<Mutex<Vec<Stats>>>,
    /// Global injector queue for load balancing
    injector: Arc<WorkStealingDeque<ScheduledTask>>,
}

impl<S: Scheduler> WorkStealingCoordinator<S> {
    /// Creates a new work-stealing coordinator with the specified strategy.
    #[must_use]
    pub fn new(strategy: WorkStealingStrategy) -> Self {
        Self {
            schedulers: Vec::new(),
            strategy,
            stats: Arc::new(Mutex::new(Vec::new())),
            injector: Arc::new(WorkStealingDeque::new(4096)),
        }
    }

    /// Register a scheduler with the coordinator.
    pub fn register_scheduler(&mut self, scheduler: S) {
        let id = scheduler.id();
        self.schedulers.push(scheduler);

        // Use expect() to treat poisoned mutex as fatal error to maintain consistency
        self.stats
            .lock()
            .expect("Stats mutex poisoned during scheduler registration")
            .push(Stats {
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

    /// Submit a task to the global injector queue
    pub fn inject_task(&self, task: ScheduledTask) {
        self.injector.push(task);
    }

    /// Try to steal from the global injector
    pub fn steal_from_injector(&self) -> Option<ScheduledTask> {
        self.injector.steal()
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
    pub fn steal_task(
        &self,
        thief_id: SchedulerId,
        context: &mut StealContext,
    ) -> SchedulerResult<Option<ScheduledTask>> {
        // First try to steal from global injector (fast path)
        if let Some(task) = self.steal_from_injector() {
            context.attempts = 0;
            context.last_success = Some(SystemTime::now());
            return Ok(Some(task));
        }

        // Find potential victims for work stealing
        let victims = self.select_victims(thief_id);

        if victims.is_empty() {
            context.attempts += 1;
            return Ok(None);
        }

        // Try to steal from each victim scheduler
        for victim_id in victims {
            if let Some(victim_scheduler) = self.schedulers.iter().find(|s| s.id() == victim_id) {
                // Check if the victim has tasks available for stealing
                if !victim_scheduler.can_be_stolen_from() {
                    continue;
                }

                // Attempt to steal a task from the victim
                match self.attempt_steal_from_victim(thief_id, victim_scheduler, context) {
                    Ok(Some(stolen_task)) => {
                        // Successfully stole a task
                        context.attempts = 0; // Reset attempts on success
                        context.last_success = Some(SystemTime::now());

                        // Update statistics for both thief and victim
                        self.update_steal_statistics(thief_id, victim_id, true);

                        return Ok(Some(stolen_task));
                    }
                    Ok(None) => {
                        // No task available from this victim, try next
                        self.update_steal_statistics(thief_id, victim_id, false);
                    }
                    Err(e) => {
                        // Steal attempt failed, update context and continue
                        context.attempts += 1;
                        self.update_steal_statistics(thief_id, victim_id, false);

                        // If it's a critical error, return it
                        if matches!(e, SchedulerError::SystemFailure(_)) {
                            return Err(e);
                        }
                        // Otherwise continue trying other victims
                    }
                }
            }
        }

        context.attempts += 1;
        Ok(None)
    }

    /// Attempt to steal a task from a specific victim scheduler.
    ///
    /// This implements the core work-stealing algorithm that tries to
    /// extract a task from the victim's queue using the scheduler's try_steal method.
    fn attempt_steal_from_victim(
        &self,
        thief_id: SchedulerId,
        victim_scheduler: &S,
        context: &mut StealContext,
    ) -> SchedulerResult<Option<ScheduledTask>> {
        // Find the thief scheduler to perform the steal operation
        if let Some(thief_scheduler) = self.schedulers.iter().find(|s| s.id() == thief_id) {
            // Use the scheduler's built-in try_steal method
            match thief_scheduler.try_steal(victim_scheduler) {
                Ok(Some(stolen_task)) => {
                    // Add the victim to recent victims list to avoid immediate re-stealing.
                    // M-16 fix: use push_back/pop_front (O(1)) on VecDeque rather than
                    // Vec::remove(0) which is O(n) due to element shifting.
                    context.recent_victims.push_back(victim_scheduler.id());

                    // Limit the recent victims list size
                    if context.recent_victims.len() > 10 {
                        context.recent_victims.pop_front();
                    }

                    Ok(Some(stolen_task))
                }
                Ok(None) => {
                    // No task available for stealing
                    Ok(None)
                }
                Err(e) => {
                    // Steal operation failed
                    Err(e)
                }
            }
        } else {
            // Thief scheduler not found
            Err(SchedulerError::InvalidScheduler)
        }
    }

    /// Update stealing statistics for performance monitoring.
    ///
    /// This tracks successful and failed steal attempts to help optimize
    /// work-stealing strategies and identify performance bottlenecks.
    fn update_steal_statistics(
        &self,
        thief_id: SchedulerId,
        victim_id: SchedulerId,
        success: bool,
    ) {
        // Use expect() to treat poisoned mutex as fatal error for consistent statistics
        let mut stats = self
            .stats
            .lock()
            .expect("Stats mutex poisoned during steal statistics update");

        // Update thief statistics
        if let Some(thief_stats) = stats.iter_mut().find(|s| s.scheduler_id == thief_id) {
            if success {
                thief_stats.steals_taken += 1;
            } else {
                thief_stats.steal_failures += 1;
            }
        }

        // Update victim statistics
        if let Some(victim_stats) = stats.iter_mut().find(|s| s.scheduler_id == victim_id) {
            if success {
                victim_stats.steals_given += 1;
            }
        }
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
                        if (seed.0 % 3) == 0 {
                            // ~33% selection probability
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
            WorkStealingStrategy::LocalityAware {
                max_attempts,
                locality_factor: _,
            } => {
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
            WorkStealingStrategy::LoadBased {
                max_attempts,
                min_load_diff: _,
            } => {
                // Select victims based on their current load
                let thief_load = self
                    .schedulers
                    .iter()
                    .find(|s| s.id() == thief_id)
                    .map_or(0, |s| s.load());

                // Get candidates with higher load than the thief
                let candidates: Vec<_> = self
                    .schedulers
                    .iter()
                    .filter(|s| s.id() != thief_id && s.load() > thief_load)
                    .collect();

                if !candidates.is_empty() {
                    // Sort by load (highest first) using sort_by_key
                    let mut sorted_candidates = candidates;
                    sorted_candidates.sort_by_key(|b| Reverse(b.load()));

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
        let thief_load = self
            .schedulers
            .iter()
            .find(|s| s.id() == thief_id)
            .map_or(0, |s| s.load());

        // Find schedulers with significantly higher load
        let mut candidates: Vec<_> = self
            .schedulers
            .iter()
            .filter(|s| s.id() != thief_id && s.load() > thief_load + 2)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort by load (highest first) and return the busiest
        candidates.sort_by_key(|b| Reverse(b.load()));
        candidates.first().map(|s| s.id())
    }

    /// Returns statistics for all registered schedulers.
    #[must_use]
    pub fn get_stats(&self) -> Vec<Stats> {
        // Use expect() to treat poisoned mutex as fatal error for consistent statistics
        self.stats
            .lock()
            .expect("Stats mutex poisoned during stats retrieval")
            .clone()
    }

    /// Update statistics for a scheduler.
    pub fn update_stats(&mut self, id: SchedulerId, stats: Stats) {
        // Use expect() to treat poisoned mutex as fatal error for consistent statistics
        let mut stats_vec = self
            .stats
            .lock()
            .expect("Stats mutex poisoned during stats update");
        if let Some(existing_stats) = stats_vec.iter_mut().find(|s| s.scheduler_id == id) {
            *existing_stats = stats;
        }
    }
}
