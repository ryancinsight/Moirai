//! Work-stealing scheduler implementation for Moirai concurrency library.

use moirai_core::{
    BoxedTask, scheduler::{Scheduler, SchedulerId, SchedulerConfig, QueueType, WorkStealingStrategy},
    error::SchedulerResult, Box,
};
use std::{
    sync::{
        atomic::{AtomicIsize, AtomicPtr, AtomicUsize, Ordering},
    },
    ptr,
    collections::VecDeque,
    sync::Mutex,
    time::Instant,
};

/// A lock-free work-stealing deque implementation based on the Chase-Lev algorithm.
pub struct ChaseLevDeque<T> {
    /// Bottom index (only modified by owner)
    bottom: AtomicIsize,
    /// Top index (modified by thieves)
    top: AtomicIsize,
    /// Array of task pointers
    array: AtomicPtr<Array<T>>,
    /// Old arrays pending deallocation
    old_arrays: Mutex<Vec<*mut Array<T>>>,
}

/// Array wrapper for the deque with atomic operations
struct Array<T> {
    /// Capacity of this array (always power of 2)
    capacity: usize,
    /// Mask for fast modulo operations
    mask: usize,
    /// The actual storage
    data: Box<[AtomicPtr<T>]>,
}

impl<T> Array<T> {
    fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(AtomicPtr::new(ptr::null_mut()));
        }
        
        Self {
            capacity,
            mask: capacity - 1,
            data: data.into_boxed_slice(),
        }
    }

    fn get(&self, index: isize) -> *mut T {
        let idx = (index as usize) & self.mask;
        self.data[idx].load(Ordering::Acquire)
    }

    fn put(&self, index: isize, item: *mut T) {
        let idx = (index as usize) & self.mask;
        self.data[idx].store(item, Ordering::Release);
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> ChaseLevDeque<T> {
    /// Create a new Chase-Lev deque with the specified initial capacity.
    pub fn new(initial_capacity: usize) -> Self {
        let capacity = initial_capacity.next_power_of_two().max(16);
        let array = Box::new(Array::new(capacity));
        
        Self {
            bottom: AtomicIsize::new(0),
            top: AtomicIsize::new(0),
            array: AtomicPtr::new(Box::into_raw(array)),
            old_arrays: Mutex::new(Vec::new()),
        }
    }

    /// Push an item to the bottom of the deque (owner operation).
    pub fn push(&self, item: T) {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Acquire);
        
        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };
        
        // Check if we need to resize
        if b - t >= array.capacity() as isize - 1 {
            self.resize();
        }
        
        // Re-load array pointer after potential resize
        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };
        
        // Store the item
        let item_ptr = Box::into_raw(Box::new(item));
        array.put(b, item_ptr);
        
        // Release the item to thieves
        self.bottom.store(b + 1, Ordering::Release);
    }

    /// Pop an item from the bottom of the deque (owner operation).
    pub fn pop(&self) -> Option<T> {
        let b = self.bottom.load(Ordering::Relaxed) - 1;
        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };
        
        self.bottom.store(b, Ordering::Relaxed);
        
        std::sync::atomic::fence(Ordering::SeqCst);
        
        let t = self.top.load(Ordering::Relaxed);
        
        if t <= b {
            // Non-empty queue
            let item_ptr = array.get(b);
            if t == b {
                // Single last element, race with thieves
                if self.top.compare_exchange_weak(
                    t, t + 1, 
                    Ordering::SeqCst, 
                    Ordering::Relaxed
                ).is_err() {
                    // Failed race, restore bottom
                    self.bottom.store(b + 1, Ordering::Relaxed);
                    return None;
                }
                self.bottom.store(b + 1, Ordering::Relaxed);
            }
            
            if !item_ptr.is_null() {
                let item = unsafe { Box::from_raw(item_ptr) };
                return Some(*item);
            }
        } else {
            // Empty queue, restore bottom
            self.bottom.store(b + 1, Ordering::Relaxed);
        }
        
        None
    }

    /// Steal an item from the top of the deque (thief operation).
    pub fn steal(&self) -> StealResult<T> {
        let t = self.top.load(Ordering::Acquire);
        
        std::sync::atomic::fence(Ordering::SeqCst);
        
        let b = self.bottom.load(Ordering::Acquire);
        
        if t < b {
            // Non-empty queue
            let array_ptr = self.array.load(Ordering::Relaxed);
            let array = unsafe { &*array_ptr };
            let item_ptr = array.get(t);
            
            if !item_ptr.is_null() {
                if self.top.compare_exchange_weak(
                    t, t + 1,
                    Ordering::SeqCst,
                    Ordering::Relaxed
                ).is_ok() {
                    let item = unsafe { Box::from_raw(item_ptr) };
                    return StealResult::Success(*item);
                }
            }
            return StealResult::Retry;
        }
        
        StealResult::Empty
    }

    /// Get the current size of the deque.
    pub fn len(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        (b - t).max(0) as usize
    }

    /// Check if the deque is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resize the underlying array when it becomes full.
    fn resize(&self) {
        let old_array_ptr = self.array.load(Ordering::Relaxed);
        let old_array = unsafe { &*old_array_ptr };
        let new_capacity = old_array.capacity() * 2;
        let new_array = Box::new(Array::new(new_capacity));
        
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        
        // Copy elements to new array
        for i in t..b {
            let item_ptr = old_array.get(i);
            new_array.put(i, item_ptr);
        }
        
        // Atomically replace the array
        let new_array_ptr = Box::into_raw(new_array);
        self.array.store(new_array_ptr, Ordering::Release);
        
        // Push the old array into the list of arrays pending deallocation
        let mut old_arrays = self.old_arrays.lock().unwrap();
        old_arrays.push(old_array_ptr);
        
        // Note: Proper memory reclamation is deferred to a safe point
    }
}

impl<T> ChaseLevDeque<T> {
    /// Safely deallocate old arrays when it is safe to do so.
    pub fn reclaim_memory(&self) {
        let mut old_arrays = self.old_arrays.lock().unwrap();
        for array_ptr in old_arrays.drain(..) {
            unsafe {
                // Deallocate the old array
                drop(Box::from_raw(array_ptr));
            }
        }
    }
}
/// Result of a steal operation.
#[derive(Debug, Clone, PartialEq)]
pub enum StealResult<T> {
    /// Successfully stole an item
    Success(T),
    /// Queue was empty
    Empty,
    /// Race condition occurred, should retry
    Retry,
}

// Safety: ChaseLevDeque is thread-safe by design
unsafe impl<T: Send> Send for ChaseLevDeque<T> {}
unsafe impl<T: Send> Sync for ChaseLevDeque<T> {}

/// Work-stealing scheduler implementation.
pub struct WorkStealingScheduler {
    /// Unique identifier for this scheduler
    id: SchedulerId,
    /// Configuration for this scheduler
    _config: SchedulerConfig,
    /// Local work queue (Chase-Lev deque)
    local_queue: ChaseLevDeque<Box<dyn BoxedTask>>,
    /// Global work queue for load balancing
    global_queue: Mutex<VecDeque<Box<dyn BoxedTask>>>,
    /// Statistics for this scheduler
    stats: SchedulerStats,
}

/// Statistics for scheduler performance monitoring.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    /// Total tasks scheduled
    tasks_scheduled: AtomicUsize,
    /// Total tasks executed
    tasks_executed: AtomicUsize,
    /// Total steal attempts
    steal_attempts: AtomicUsize,
    /// Successful steals
    successful_steals: AtomicUsize,
    /// Time spent executing tasks (nanoseconds)
    execution_time_ns: AtomicUsize,
    /// Last activity timestamp
    last_activity: AtomicUsize,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler.
    pub fn new(id: SchedulerId, config: SchedulerConfig) -> Self {
        let initial_capacity = match config.queue_type {
            QueueType::ChaseLev => 1024, // Default capacity
            _ => 256, // Smaller capacity for other types
        };

        Self {
            id,
            _config: config,
            local_queue: ChaseLevDeque::new(initial_capacity),
            global_queue: Mutex::new(VecDeque::new()),
            stats: SchedulerStats::default(),
        }
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
    pub fn try_steal_from(&self, other: &WorkStealingScheduler) -> StealResult<Box<dyn BoxedTask>> {
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
    fn execute_task(&self, task: Box<dyn BoxedTask>) {
        let start_time = Instant::now();
        
        // Execute the task
        task.execute_boxed();
        
        // Update statistics
        let execution_time = start_time.elapsed().as_nanos() as usize;
        self.stats.tasks_executed.fetch_add(1, Ordering::Relaxed);
        self.stats.execution_time_ns.fetch_add(execution_time, Ordering::Relaxed);
        self.stats.last_activity.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as usize,
            Ordering::Relaxed
        );
    }

    /// Get current load (number of queued tasks).
    pub fn load(&self) -> usize {
        let local_load = self.local_queue.len();
        let global_load = self.global_queue.lock()
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
    fn schedule_task(&self, task: Box<dyn BoxedTask>) -> SchedulerResult<()> {
        self.stats.tasks_scheduled.fetch_add(1, Ordering::Relaxed);
        
        // Prefer local queue for better cache locality
        self.local_queue.push(task);
        Ok(())
    }

    fn next_task(&self) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
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

    fn try_steal(&self, victim: &dyn Scheduler) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
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
    ) -> Option<Box<dyn BoxedTask>> {
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
    ) -> Option<Box<dyn BoxedTask>> {
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
    ) -> Option<Box<dyn BoxedTask>> {
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
    ) -> Option<Box<dyn BoxedTask>> {
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
    ) -> Option<Box<dyn BoxedTask>> {
        // Simplified locality-aware stealing based on scheduler ID distance
        let idle_id = idle_scheduler.id().get();
        
        let mut candidates: Vec<_> = all_schedulers.iter()
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
        let next = current.wrapping_mul(1103515245).wrapping_add(12345);
        self.rng_state.store(next, Ordering::Relaxed);
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moirai_core::{Task, TaskContext, TaskId};

    // Test task implementation
    struct TestTask {
        id: u32,
        context: TaskContext,
    }

    impl TestTask {
        fn new(id: u32) -> Self {
            Self {
                id,
                context: TaskContext::new(TaskId::new(id as u64)),
            }
        }
    }

    impl Task for TestTask {
        type Output = u32;

        fn execute(self) -> Self::Output {
            self.id * 2
        }

        fn context(&self) -> &TaskContext {
            &self.context
        }
    }

    impl BoxedTask for TestTask {
        fn execute_boxed(self: Box<Self>) {
            let _ = (*self).execute();
        }

        fn context(&self) -> &TaskContext {
            Task::context(self)
        }
    }

    #[test]
    fn test_chase_lev_deque_basic_operations() {
        let deque: ChaseLevDeque<i32> = ChaseLevDeque::new(16);
        
        // Test push and pop
        deque.push(1);
        deque.push(2);
        deque.push(3);
        
        assert_eq!(deque.len(), 3);
        assert!(!deque.is_empty());
        
        assert_eq!(deque.pop(), Some(3));
        assert_eq!(deque.pop(), Some(2));
        assert_eq!(deque.pop(), Some(1));
        assert_eq!(deque.pop(), None);
        
        assert!(deque.is_empty());
    }

    #[test]
    fn test_chase_lev_deque_steal() {
        let deque: ChaseLevDeque<i32> = ChaseLevDeque::new(16);
        
        // Push some items
        for i in 1..=5 {
            deque.push(i);
        }
        
        // Steal from the top
        assert_eq!(deque.steal(), StealResult::Success(1));
        assert_eq!(deque.steal(), StealResult::Success(2));
        
        // Pop from the bottom
        assert_eq!(deque.pop(), Some(5));
        assert_eq!(deque.pop(), Some(4));
        
        // Steal the last item
        assert_eq!(deque.steal(), StealResult::Success(3));
        
        // Should be empty now
        assert_eq!(deque.steal(), StealResult::Empty);
        assert_eq!(deque.pop(), None);
    }

    #[test]
    fn test_work_stealing_scheduler() {
        let config = SchedulerConfig::default();
        let scheduler = WorkStealingScheduler::new(SchedulerId::new(0), config);
        
        // Test task scheduling
        let task = Box::new(TestTask::new(42));
        scheduler.schedule_task(task).unwrap();
        
        assert_eq!(scheduler.load(), 1);
        
        // Test task execution
        let executed = scheduler.try_execute_next_task().unwrap();
        assert!(executed);
        assert_eq!(scheduler.load(), 0);
        
        // No more tasks
        let executed = scheduler.try_execute_next_task().unwrap();
        assert!(!executed);
    }

    #[test]
    fn test_scheduler_stats() {
        let config = SchedulerConfig::default();
        let scheduler = WorkStealingScheduler::new(SchedulerId::new(1), config);
        
        // Schedule and execute some tasks
        for i in 0..5 {
            let task = Box::new(TestTask::new(i));
            scheduler.schedule_task(task).unwrap();
        }
        
        // Execute all tasks
        while scheduler.try_execute_next_task().unwrap() {}
        
        let stats = scheduler.stats();
        assert_eq!(stats.scheduler_id, SchedulerId::new(1));
        assert_eq!(stats.tasks_scheduled, 5);
        assert_eq!(stats.tasks_executed, 5);
        assert_eq!(stats.current_load, 0);
        assert!(stats.execution_time_ns > 0);
    }

    #[test]
    fn test_local_scheduler() {
        let config = SchedulerConfig {
            queue_type: QueueType::ChaseLev,
            ..Default::default()
        };
        let scheduler = WorkStealingScheduler::new(SchedulerId::new(2), config);
        
        // Test multiple task scheduling
        for i in 0..10 {
            let task = Box::new(TestTask::new(i));
            scheduler.schedule_task(task).unwrap();
        }
        
        assert_eq!(scheduler.load(), 10);
        
        // Execute some tasks
        let mut executed_count = 0;
        while scheduler.try_execute_next_task().unwrap() {
            executed_count += 1;
        }
        
        assert_eq!(executed_count, 10);
        assert_eq!(scheduler.load(), 0);
    }
}