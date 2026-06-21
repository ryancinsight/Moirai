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

const DEFAULT_CHASELEV_CAPACITY: usize = 1024;
const DEFAULT_QUEUE_CAPACITY: usize = 256;
const LCG_MULTIPLIER: usize = 1103515245;
const LCG_INCREMENT: usize = 12345;

// ── SchedulerStats ────────────────────────────────────────────────────────────

/// Statistics for scheduler performance monitoring.
/// Each counter is cache-aligned to prevent false sharing between threads.
#[derive(Debug)]
pub struct SchedulerStats {
    tasks_scheduled: CacheAligned<AtomicUsize>,
    tasks_executed: CacheAligned<AtomicUsize>,
    steal_attempts: CacheAligned<AtomicUsize>,
    successful_steals: CacheAligned<AtomicUsize>,
    execution_time_ns: CacheAligned<AtomicUsize>,
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
    id: SchedulerId,
    _config: SchedulerConfig,
    local_queue: ChaseLevDeque<ScheduledTask>,
    global_queue: Mutex<VecDeque<ScheduledTask>>,
    global_len: AtomicUsize,
    stats: SchedulerStats,
}

impl WorkStealingScheduler {
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
            global_len: AtomicUsize::new(0),
            stats: SchedulerStats::default(),
        }
    }

    pub fn schedule_task<T>(&self, task: T) -> SchedulerResult<()>
    where
        T: Task,
    {
        self.schedule(ScheduledTask::new(task))
    }

    pub fn try_execute_next_task(&self) -> SchedulerResult<bool> {
        if let Some(task) = self.local_queue.pop() {
            self.execute_task(task);
            return Ok(true);
        }

        if let Ok(mut global) = self.global_queue.try_lock() {
            if let Some(task) = global.pop_front() {
                self.global_len.fetch_sub(1, Ordering::Relaxed);
                drop(global);
                self.execute_task(task);
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub fn try_steal_from(&self, other: &WorkStealingScheduler) -> StealResult<ScheduledTask> {
        self.stats.steal_attempts.fetch_add(1, Ordering::Relaxed);

        if other.local_queue.is_empty() {
            return StealResult::Empty;
        }

        match other.local_queue.steal() {
            StealResult::Success(task) => {
                self.stats.successful_steals.fetch_add(1, Ordering::Relaxed);
                StealResult::Success(task)
            }
            other_result => other_result,
        }
    }

    pub fn try_steal_batch_from(
        &self,
        other: &WorkStealingScheduler,
    ) -> StealResult<ScheduledTask> {
        self.stats.steal_attempts.fetch_add(1, Ordering::Relaxed);

        if other.local_queue.is_empty() {
            return StealResult::Empty;
        }

        let dest_queue = &self.local_queue;
        match other.local_queue.steal_batch_with(|task| {
            dest_queue.push(task);
        }) {
            StealResult::Success(task) => {
                self.stats.successful_steals.fetch_add(1, Ordering::Relaxed);
                StealResult::Success(task)
            }
            other_result => other_result,
        }
    }

    fn execute_task(&self, task: ScheduledTask) {
        let start_time = Instant::now();

        task.execute();

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

    pub fn load(&self) -> usize {
        let local_load = self.local_queue.len();
        let global_load = self.global_len.load(Ordering::Relaxed);
        local_load + global_load
    }

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
        self.local_queue.push(task);
        Ok(())
    }

    fn next_task(&self) -> SchedulerResult<Option<ScheduledTask>> {
        if let Some(task) = self.local_queue.pop() {
            return Ok(Some(task));
        }

        if let Ok(mut global) = self.global_queue.try_lock() {
            if let Some(task) = global.pop_front() {
                self.global_len.fetch_sub(1, Ordering::Relaxed);
                return Ok(Some(task));
            }
        }

        Ok(None)
    }

    fn steal_task(&self) -> SchedulerResult<Option<ScheduledTask>> {
        match self.local_queue.steal() {
            StealResult::Success(task) => Ok(Some(task)),
            StealResult::Retry | StealResult::Empty => Ok(None),
        }
    }

    fn try_steal<S>(&self, victim: &S) -> SchedulerResult<Option<ScheduledTask>>
    where
        S: Scheduler,
    {
        victim.steal_task()
    }

    fn load(&self) -> usize {
        self.load()
    }

    fn id(&self) -> SchedulerId {
        self.id
    }
}

// ── WorkStealingCoordinator ───────────────────────────────────────────────────

pub struct WorkStealingCoordinator {
    strategy: WorkStealingStrategy,
    rng_state: AtomicUsize,
}

impl WorkStealingCoordinator {
    pub fn new(strategy: WorkStealingStrategy) -> Self {
        Self {
            strategy,
            rng_state: AtomicUsize::new(1),
        }
    }

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
                match base_strategy.as_ref() {
                    WorkStealingStrategy::Random { max_attempts } => {
                        self.random_steal(idle_scheduler, all_schedulers, *max_attempts)
                    }
                    _ => None,
                }
            }
        }
    }

    fn random_steal(
        &self,
        idle_scheduler: &WorkStealingScheduler,
        all_schedulers: &[std::sync::Arc<WorkStealingScheduler>],
        max_attempts: usize,
    ) -> Option<ScheduledTask> {
        for _ in 0..max_attempts {
            let target_idx = self.next_random() % all_schedulers.len();
            let target = &all_schedulers[target_idx];

            if target.id() == idle_scheduler.id() {
                continue;
            }

            match idle_scheduler.try_steal_batch_from(target) {
                StealResult::Success(task) => return Some(task),
                StealResult::Retry => continue,
                StealResult::Empty => continue,
            }
        }
        None
    }

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

            if target.id() == idle_scheduler.id() {
                continue;
            }

            match idle_scheduler.try_steal_batch_from(target) {
                StealResult::Success(task) => return Some(task),
                StealResult::Retry => {
                    continue;
                }
                StealResult::Empty => continue,
            }
        }
        None
    }

    fn load_based_steal(
        &self,
        idle_scheduler: &WorkStealingScheduler,
        all_schedulers: &[std::sync::Arc<WorkStealingScheduler>],
        max_attempts: usize,
    ) -> Option<ScheduledTask> {
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
                match idle_scheduler.try_steal_batch_from(target) {
                    StealResult::Success(task) => return Some(task),
                    StealResult::Retry => continue,
                    StealResult::Empty => break,
                }
            }
        }

        None
    }

    fn locality_aware_steal(
        &self,
        idle_scheduler: &WorkStealingScheduler,
        all_schedulers: &[std::sync::Arc<WorkStealingScheduler>],
        max_attempts: usize,
    ) -> Option<ScheduledTask> {
        let idle_id = idle_scheduler.id().get();
        let len = all_schedulers.len();
        if len <= 1 {
            return None;
        }

        // Zero-allocation search: track visited indexes using a stack-allocated array of size 128
        let mut visited = [usize::MAX; 128];
        let mut attempts = 0;

        while attempts < max_attempts {
            let mut best_idx = None;
            let mut min_distance = usize::MAX;

            for (idx, scheduler) in all_schedulers.iter().enumerate() {
                if scheduler.id() == idle_scheduler.id() {
                    continue;
                }

                // Check if already visited
                if visited.iter().take(attempts.min(128)).any(|&x| x == idx) {
                    continue;
                }

                let load = scheduler.load();
                if load > 0 {
                    let distance = ((scheduler.id().get() as i32) - (idle_id as i32)).abs() as usize;
                    if distance < min_distance {
                        min_distance = distance;
                        best_idx = Some(idx);
                    }
                }
            }

            if let Some(idx) = best_idx {
                if attempts < 128 {
                    visited[attempts] = idx;
                }

                let target = &all_schedulers[idx];
                match idle_scheduler.try_steal_batch_from(target) {
                    StealResult::Success(task) => return Some(task),
                    StealResult::Retry | StealResult::Empty => {}
                }
                attempts += 1;
            } else {
                break;
            }
        }

        None
    }

    fn next_random(&self) -> usize {
        thread_local! {
            static THREAD_RNG_STATE: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
        }
        let owner = self as *const Self as usize;
        let (cached_owner, mut state) = THREAD_RNG_STATE.get();
        if cached_owner != owner {
            state = 0;
        }
        if state == 0 {
            state = self.rng_state.fetch_add(1, Ordering::Relaxed);
            if state == 0 {
                state = 1;
            }
        }
        let next = state
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(LCG_INCREMENT);
        THREAD_RNG_STATE.set((owner, next));
        next
    }
}
