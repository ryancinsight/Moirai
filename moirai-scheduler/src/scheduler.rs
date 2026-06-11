//! Work-stealing scheduler, statistics, and coordinator.
//!
//! Provides:
//! - [`WorkStealingScheduler`] — concrete [`Scheduler`] implementation backed
//!   by a [`ChaseLevDeque`] local queue.
//! - [`SchedulerStats`] — per-scheduler atomic counters (cache-aligned).
//! - [`SchedulerStatsSnapshot`] — point-in-time snapshot of [`SchedulerStats`].
//! - [`WorkStealingCoordinator`] — multi-scheduler steal coordinator with
//!   pluggable [`WorkStealingStrategy`].

use crate::deque::{ChaseLevDeque, StealResult};
use moirai_core::{
    error::SchedulerResult,
    scheduler::{QueueType, Scheduler, SchedulerConfig, SchedulerId, WorkStealingStrategy},
    CacheAligned, ScheduledTask, Task,
};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};

/// Default queue capacity for Chase-Lev deques.
const DEFAULT_CHASELEV_CAPACITY: usize = 1024;

/// Default queue capacity for other queue types.
const DEFAULT_QUEUE_CAPACITY: usize = 256;

/// Linear congruential generator multiplier (standard LCG constant).
const LCG_MULTIPLIER: usize = 1103515245;

/// Linear congruential generator increment (standard LCG constant).
const LCG_INCREMENT: usize = 12345;

// ── SchedulerStats ────────────────────────────────────────────────────────────

/// Statistics for scheduler performance monitoring.
/// Each counter is cache-aligned to prevent false sharing between threads.
#[derive(Debug)]
pub struct SchedulerStats {
    /// Total tasks scheduled
    tasks_scheduled: CacheAligned<AtomicUsize>,
    /// Total tasks executed
    tasks_executed: CacheAligned<AtomicUsize>,
    /// Total steal attempts
    steal_attempts: CacheAligned<AtomicUsize>,
    /// Successful steals
    successful_steals: CacheAligned<AtomicUsize>,
    /// Time spent executing tasks (nanoseconds)
    execution_time_ns: CacheAligned<AtomicUsize>,
    /// Last activity timestamp
    last_activity: CacheAligned<AtomicUsize>,
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self {
            tasks_scheduled: CacheAligned::new(AtomicUsize::new(0)),
            tasks_executed: CacheAligned::new(AtomicUsize::new(0)),
            steal_attempts: CacheAligned::new(AtomicUsize::new(0)),
            successful_steals: CacheAligned::new(AtomicUsize::new(0)),
            execution_time_ns: CacheAligned::new(AtomicUsize::new(0)),
            last_activity: CacheAligned::new(AtomicUsize::new(0)),
        }
    }
}

// ── SchedulerStatsSnapshot ────────────────────────────────────────────────────

/// Snapshot of scheduler statistics at a point in time.
#[derive(Debug, Clone)]
pub struct SchedulerStatsSnapshot {
    pub scheduler_id: SchedulerId,
    pub tasks_scheduled: usize,
    pub tasks_executed: usize,
    pub steal_attempts: usize,
    pub successful_steals: usize,
    pub execution_time_ns: usize,
    pub current_load: usize,
    pub steal_success_rate: f64,
}

// ── WorkStealingScheduler ─────────────────────────────────────────────────────

/// Work-stealing scheduler implementation.
pub struct WorkStealingScheduler {
    /// Unique identifier for this scheduler
    id: SchedulerId,
    /// Configuration for this scheduler
    _config: SchedulerConfig,
    /// Local work queue (Chase-Lev deque)
    local_queue: ChaseLevDeque<ScheduledTask>,
    /// Global work queue for load balancing
    global_queue: Mutex<VecDeque<ScheduledTask>>,
    /// Statistics for this scheduler
    stats: SchedulerStats,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler.
    pub fn new(id: SchedulerId, config: SchedulerConfig) -> Self {
        let initial_capacity = match config.queue_type {
            QueueType::ChaseLev => DEFAULT_CHASELEV_CAPACITY,
            _ => DEFAULT_QUEUE_CAPACITY,
        };

        Self {
            id,
            _config: config,
            local_queue: ChaseLevDeque::new(initial_capacity),
            global_queue: Mutex::new(VecDeque::new()),
            stats: SchedulerStats::default(),
        }
    }

    /// Schedule a concrete task without a task trait object.
    pub fn schedule_task<T>(&self, task: T) -> SchedulerResult<()>
    where
        T: Task,
    {
        self.schedule(ScheduledTask::new(task))
    }

    /// Try to execute the next available task.
    pub fn try_execute_next_task(&self) -> SchedulerResult<bool> {
        // First, try local queue
        if let Some(task) = self.local_queue.pop() {
            self.execute_task(task);
            return Ok(true);
        }

        // Then try global queue
        if let Ok(mut global) = self.global_queue.try_lock() {
            if let Some(task) = global.pop_front() {
                drop(global); // Release lock before execution
                self.execute_task(task);
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Try to steal work from another scheduler.
    pub fn try_steal_from(&self, other: &WorkStealingScheduler) -> StealResult<ScheduledTask> {
        self.stats.steal_attempts.fetch_add(1, Ordering::Relaxed);

        match other.local_queue.steal() {
            StealResult::Success(task) => {
                self.stats.successful_steals.fetch_add(1, Ordering::Relaxed);
                StealResult::Success(task)
            }
            other_result => other_result,
        }
    }

    /// Execute a single task.
    fn execute_task(&self, task: ScheduledTask) {
        let start_time = Instant::now();

        // Execute the task
        task.execute();

        // Update statistics
        let execution_time = start_time.elapsed().as_nanos() as usize;
        self.stats.tasks_executed.fetch_add(1, Ordering::Relaxed);
        self.stats
            .execution_time_ns
            .fetch_add(execution_time, Ordering::Relaxed);
        self.stats.last_activity.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as usize,
            Ordering::Relaxed,
        );
    }

    /// Get current load (number of queued tasks).
    pub fn load(&self) -> usize {
        let local_load = self.local_queue.len();
        let global_load = self
            .global_queue
            .lock()
            .map(|queue| queue.len())
            .unwrap_or(0);
        local_load + global_load
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> SchedulerStatsSnapshot {
        SchedulerStatsSnapshot {
            scheduler_id: self.id,
            tasks_scheduled: self.stats.tasks_scheduled.load(Ordering::Relaxed),
            tasks_executed: self.stats.tasks_executed.load(Ordering::Relaxed),
            steal_attempts: self.stats.steal_attempts.load(Ordering::Relaxed),
            successful_steals: self.stats.successful_steals.load(Ordering::Relaxed),
            execution_time_ns: self.stats.execution_time_ns.load(Ordering::Relaxed),
            current_load: self.load(),
            steal_success_rate: {
                let attempts = self.stats.steal_attempts.load(Ordering::Relaxed);
                let successes = self.stats.successful_steals.load(Ordering::Relaxed);
                if attempts > 0 {
                    (successes as f64) / (attempts as f64)
                } else {
                    0.0
                }
            },
        }
    }
}

impl Scheduler for WorkStealingScheduler {
    fn schedule(&self, task: ScheduledTask) -> SchedulerResult<()> {
        self.stats.tasks_scheduled.fetch_add(1, Ordering::Relaxed);

        // Prefer local queue for better cache locality
        self.local_queue.push(task);
        Ok(())
    }

    fn next_task(&self) -> SchedulerResult<Option<ScheduledTask>> {
        // First, try local queue
        if let Some(task) = self.local_queue.pop() {
            return Ok(Some(task));
        }

        // Then try global queue
        if let Ok(mut global) = self.global_queue.try_lock() {
            if let Some(task) = global.pop_front() {
                return Ok(Some(task));
            }
        }

        Ok(None)
    }

    fn try_steal<S>(&self, victim: &S) -> SchedulerResult<Option<ScheduledTask>>
    where
        S: Scheduler,
    {
        // For simplicity, we'll use the load-based approach as a fallback
        // In a real implementation, we'd have a more sophisticated mechanism
        if victim.can_be_stolen_from() {
            // Try to get a task from the victim's next_task method
            // This is not as efficient as direct stealing but works with the trait
            victim.next_task()
        } else {
            Ok(None)
        }
    }

    fn load(&self) -> usize {
        self.load()
    }

    fn id(&self) -> SchedulerId {
        self.id
    }
}

// ── WorkStealingCoordinator ───────────────────────────────────────────────────

/// Coordinator for work-stealing between multiple schedulers.
pub struct WorkStealingCoordinator {
    /// Strategy for selecting steal targets
    strategy: WorkStealingStrategy,
    /// Random number generator state for random stealing
    rng_state: AtomicUsize,
}

impl WorkStealingCoordinator {
    /// Create a new work-stealing coordinator.
    pub fn new(strategy: WorkStealingStrategy) -> Self {
        Self {
            strategy,
            rng_state: AtomicUsize::new(1), // Simple LCG seed
        }
    }

    /// Attempt to steal work for an idle scheduler.
    pub fn steal_work(
        &self,
        idle_scheduler: &WorkStealingScheduler,
        all_schedulers: &[std::sync::Arc<WorkStealingScheduler>],
    ) -> Option<ScheduledTask> {
        match &self.strategy {
            WorkStealingStrategy::Random { max_attempts } => {
                self.random_steal(idle_scheduler, all_schedulers, *max_attempts)
            }
            WorkStealingStrategy::RoundRobin { max_attempts } => {
                self.round_robin_steal(idle_scheduler, all_schedulers, *max_attempts)
            }
            WorkStealingStrategy::LoadBased { max_attempts, .. } => {
                self.load_based_steal(idle_scheduler, all_schedulers, *max_attempts)
            }
            WorkStealingStrategy::LocalityAware { max_attempts, .. } => {
                self.locality_aware_steal(idle_scheduler, all_schedulers, *max_attempts)
            }
            WorkStealingStrategy::Adaptive { base_strategy, .. } => {
                // Use base strategy for now
                match base_strategy.as_ref() {
                    WorkStealingStrategy::Random { max_attempts } => {
                        self.random_steal(idle_scheduler, all_schedulers, *max_attempts)
                    }
                    _ => None,
                }
            }
        }
    }

    /// Random work stealing strategy.
    fn random_steal(
        &self,
        idle_scheduler: &WorkStealingScheduler,
        all_schedulers: &[std::sync::Arc<WorkStealingScheduler>],
        max_attempts: usize,
    ) -> Option<ScheduledTask> {
        for _ in 0..max_attempts {
            let target_idx = self.next_random() % all_schedulers.len();
            let target = &all_schedulers[target_idx];

            // Don't steal from ourselves
            if target.id() == idle_scheduler.id() {
                continue;
            }

            match idle_scheduler.try_steal_from(target) {
                StealResult::Success(task) => return Some(task),
                StealResult::Retry => continue,
                StealResult::Empty => continue,
            }
        }
        None
    }

    /// Round-robin work stealing strategy.
    fn round_robin_steal(
        &self,
        idle_scheduler: &WorkStealingScheduler,
        all_schedulers: &[std::sync::Arc<WorkStealingScheduler>],
        max_attempts: usize,
    ) -> Option<ScheduledTask> {
        let start_idx = (idle_scheduler.id().get() + 1) % all_schedulers.len();

        for i in 0..max_attempts.min(all_schedulers.len()) {
            let target_idx = (start_idx + i) % all_schedulers.len();
            let target = &all_schedulers[target_idx];

            // Don't steal from ourselves
            if target.id() == idle_scheduler.id() {
                continue;
            }

            match idle_scheduler.try_steal_from(target) {
                StealResult::Success(task) => return Some(task),
                StealResult::Retry => {
                    // For round-robin, we give each scheduler one chance
                    continue;
                }
                StealResult::Empty => continue,
            }
        }
        None
    }

    /// Load-based work stealing strategy.
    fn load_based_steal(
        &self,
        idle_scheduler: &WorkStealingScheduler,
        all_schedulers: &[std::sync::Arc<WorkStealingScheduler>],
        max_attempts: usize,
    ) -> Option<ScheduledTask> {
        // Find the scheduler with the highest load
        let mut best_target: Option<&WorkStealingScheduler> = None;
        let mut max_load = 0;

        for scheduler in all_schedulers {
            if scheduler.id() == idle_scheduler.id() {
                continue;
            }

            let load = scheduler.load();
            if load > max_load {
                max_load = load;
                best_target = Some(scheduler);
            }
        }

        if let Some(target) = best_target {
            for _ in 0..max_attempts {
                match idle_scheduler.try_steal_from(target) {
                    StealResult::Success(task) => return Some(task),
                    StealResult::Retry => continue,
                    StealResult::Empty => break,
                }
            }
        }

        None
    }

    /// Locality-aware work stealing strategy.
    fn locality_aware_steal(
        &self,
        idle_scheduler: &WorkStealingScheduler,
        all_schedulers: &[std::sync::Arc<WorkStealingScheduler>],
        max_attempts: usize,
    ) -> Option<ScheduledTask> {
        // Simplified locality-aware stealing based on scheduler ID distance
        let idle_id = idle_scheduler.id().get();

        let mut candidates: Vec<_> = all_schedulers
            .iter()
            .filter(|s| s.id() != idle_scheduler.id() && s.load() > 0)
            .map(|s| {
                let distance = ((s.id().get() as i32) - (idle_id as i32)).abs() as usize;
                (s, distance)
            })
            .collect();

        // Sort by distance (closer first)
        candidates.sort_by_key(|(_, distance)| *distance);

        for (target, _) in candidates.iter().take(max_attempts) {
            match idle_scheduler.try_steal_from(target) {
                StealResult::Success(task) => return Some(task),
                StealResult::Retry => continue,
                StealResult::Empty => continue,
            }
        }

        None
    }

    /// Simple linear congruential generator for random numbers.
    fn next_random(&self) -> usize {
        let current = self.rng_state.load(Ordering::Relaxed);
        let next = current
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        self.rng_state.store(next, Ordering::Relaxed);
        next
    }
}
