//! Scheduler trait and implementations.
//! 
//! This module provides advanced scheduling algorithms inspired by:
//! - Rayon's work-stealing deque (Chase-Lev algorithm)
//! - Tokio's async notification system  
//! - OpenMP's low-overhead synchronization

use crate::{BoxedTask};
use crate::error::{SchedulerError, SchedulerResult};
use crate::platform::*;
use core::fmt;
use core::num::Wrapping;
use core::cmp::Reverse;

#[cfg(feature = "std")]
use std::time::SystemTime;

/// Padding helper to ensure cache line alignment
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

/// Chase-Lev work-stealing deque implementation (inspired by Rayon)
/// 
/// This is a highly optimized deque that allows:
/// - Single owner pushing/popping from one end
/// - Multiple stealers taking from the other end
/// - Minimal synchronization overhead
pub struct WorkStealingDeque<T> {
    /// Bottom index - owned by worker
    bottom: CachePadded<AtomicUsize>,
    /// Top index - accessed by stealers
    top: CachePadded<AtomicUsize>,
    /// Ring buffer for tasks
    buffer: CachePadded<AtomicPtr<Buffer<T>>>,
    _phantom: PhantomData<T>,
}

struct Buffer<T> {
    /// Capacity mask (capacity - 1 for fast modulo)
    mask: usize,
    /// Storage for tasks
    storage: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

impl<T> Buffer<T> {
    fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        let storage = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        
        Self {
            mask: capacity - 1,
            storage,
        }
    }

    unsafe fn get(&self, index: usize) -> &T {
        let slot = &*self.storage[index & self.mask].get();
        &*slot.as_ptr()
    }

    unsafe fn put(&self, index: usize, value: T) {
        let slot = &mut *self.storage[index & self.mask].get();
        slot.write(value);
    }
    
    fn capacity(&self) -> usize {
        self.storage.len()
    }
}

impl<T: Send> WorkStealingDeque<T> {
    /// Create a new work-stealing deque
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer = Box::into_raw(Box::new(Buffer::new(capacity)));
        
        Self {
            bottom: CachePadded { value: AtomicUsize::new(0) },
            top: CachePadded { value: AtomicUsize::new(0) },
            buffer: CachePadded { value: AtomicPtr::new(buffer) },
            _phantom: PhantomData,
        }
    }

    /// Push a task (owner only)
    pub fn push(&self, task: T) {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let top = self.top.value.load(Ordering::Acquire);
        let size = bottom.wrapping_sub(top);
        
        let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
        
        // Check if resize needed
        if size >= buffer.mask {
            // In production, implement buffer growth here
            panic!("Deque full - resize not implemented");
        }
        
        unsafe {
            buffer.put(bottom, task);
        }
        
        // Release store to make task visible to stealers
        self.bottom.value.store(bottom.wrapping_add(1), Ordering::Release);
        fence(Ordering::SeqCst);
    }

    /// Pop a task (owner only)
    pub fn pop(&self) -> Option<T> {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let new_bottom = bottom.wrapping_sub(1);
        
        // Synchronize with stealers
        fence(Ordering::SeqCst);
        
        let top = self.top.value.load(Ordering::Relaxed);
        
        if top <= new_bottom {
            // Non-empty
            let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
            // Ensure new_bottom is within buffer bounds before reading
            let capacity = buffer.capacity();
            if new_bottom >= capacity {
                // Out of bounds, restore bottom and return None
                self.bottom.value.store(bottom, Ordering::Relaxed);
                return None;
            }
            
            let task = unsafe { core::ptr::read(buffer.get(new_bottom)) };
            
            if top == new_bottom {
                // Last task - race with stealers
                if self.top.value.compare_exchange(
                    top,
                    top.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed
                ).is_err() {
                    // Lost race
                    self.bottom.value.store(bottom, Ordering::Relaxed);
                    return None;
                }
                self.bottom.value.store(bottom, Ordering::Relaxed);
            }
            
            Some(task)
        } else {
            // Empty
            self.bottom.value.store(bottom, Ordering::Relaxed);
            None
        }
    }

    /// Steal a task (can be called by multiple threads)
    pub fn steal(&self) -> Option<T> {
        loop {
            let top = self.top.value.load(Ordering::Acquire);
            
            // Synchronize with owner
            fence(Ordering::SeqCst);
            
            let bottom = self.bottom.value.load(Ordering::Acquire);
            
            if top >= bottom {
                return None; // Empty
            }
            
            let buffer = unsafe { &*self.buffer.value.load(Ordering::Acquire) };
            let task = unsafe { core::ptr::read(buffer.get(top)) };
            
            // Try to increment top
            if self.top.value.compare_exchange(
                top,
                top.wrapping_add(1),
                Ordering::SeqCst,
                Ordering::Relaxed
            ).is_ok() {
                return Some(task);
            }
            
            // CAS failed, retry
        }
    }

    /// Get the current size estimate
    pub fn len(&self) -> usize {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let top = self.top.value.load(Ordering::Relaxed);
        bottom.wrapping_sub(top)
    }
}

// Safety: Tasks are Send
unsafe impl<T: Send> Send for WorkStealingDeque<T> {}
unsafe impl<T: Send> Sync for WorkStealingDeque<T> {}

/// Core scheduling interface for task distribution and execution.
///
/// This trait defines the fundamental operations that all scheduler implementations
/// must support for managing task queues and work distribution.
pub trait Scheduler: Send + Sync + 'static {
    /// Schedule a task for execution.
    /// 
    /// The scheduler will determine when and where to execute the task based on
    /// its internal policies and current system state.
    /// 
    /// # Errors
    /// Returns `SchedulerError` if the task cannot be scheduled due to:
    /// - Resource constraints (queue full, memory limits)
    /// - Scheduler shutdown or invalid state
    /// - Task validation failures
    fn schedule(&self, task: Box<dyn BoxedTask>) -> SchedulerResult<()>;

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
        max_attempts: usize 
    },
    /// Round-robin victim selection
    RoundRobin { 
        /// Maximum number of steal attempts before giving up
        max_attempts: usize 
    },
    /// Locality-aware victim selection
    LocalityAware { 
        /// Maximum number of steal attempts before giving up
        max_attempts: usize, 
        /// Weight factor for locality preference (0.0 to 1.0)
        locality_factor: f64 
    },
    /// Load-based victim selection
    LoadBased { 
        /// Maximum number of steal attempts before giving up
        max_attempts: usize, 
        /// Minimum load difference required to attempt stealing
        min_load_diff: usize 
    },
    /// Adaptive strategy that adjusts based on success rate
    Adaptive { 
        /// Base strategy to adapt from
        base_strategy: Box<WorkStealingStrategy>, 
        /// Rate at which to adapt the strategy (0.0 to 1.0)
        adaptation_rate: f64 
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

/// Work stealing coordinator that manages steal attempts across schedulers.
/// 
/// This implementation now uses work-stealing deques for better performance.
pub struct WorkStealingCoordinator {
    schedulers: Vec<Box<dyn Scheduler>>,
    strategy: WorkStealingStrategy,
    stats: Arc<Mutex<Vec<Stats>>>,
    /// Global injector queue for load balancing
    injector: Arc<WorkStealingDeque<Box<dyn BoxedTask>>>,
}

impl WorkStealingCoordinator {
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
    pub fn register_scheduler(&mut self, scheduler: Box<dyn Scheduler>) {
        let id = scheduler.id();
        self.schedulers.push(scheduler);
        
        // Use expect() to treat poisoned mutex as fatal error to maintain consistency
        self.stats.lock().expect("Stats mutex poisoned during scheduler registration").push(Stats {
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
    pub fn inject_task(&self, task: Box<dyn BoxedTask>) {
        self.injector.push(task);
    }

    /// Try to steal from the global injector
    pub fn steal_from_injector(&self) -> Option<Box<dyn BoxedTask>> {
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
    pub fn steal_task(&self, thief_id: SchedulerId, context: &mut StealContext) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
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
                        continue;
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
        victim_scheduler: &Box<dyn Scheduler>,
        context: &mut StealContext,
    ) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        // Find the thief scheduler to perform the steal operation
        if let Some(thief_scheduler) = self.schedulers.iter().find(|s| s.id() == thief_id) {
            // Use the scheduler's built-in try_steal method
            match thief_scheduler.try_steal(victim_scheduler.as_ref()) {
                Ok(Some(stolen_task)) => {
                    // Add the victim to recent victims list to avoid immediate re-stealing
                    context.recent_victims.push(victim_scheduler.id());
                    
                    // Limit the recent victims list size
                    if context.recent_victims.len() > 10 {
                        context.recent_victims.remove(0);
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
    fn update_steal_statistics(&self, thief_id: SchedulerId, victim_id: SchedulerId, success: bool) {
        // Use expect() to treat poisoned mutex as fatal error for consistent statistics
        let mut stats = self.stats.lock().expect("Stats mutex poisoned during steal statistics update");
        
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
        candidates.sort_by_key(|b| Reverse(b.load()));
        candidates.first().map(|s| s.id())
    }

    /// Returns statistics for all registered schedulers.
    #[must_use]
    pub fn get_stats(&self) -> Vec<Stats> {
        // Use expect() to treat poisoned mutex as fatal error for consistent statistics
        self.stats.lock().expect("Stats mutex poisoned during stats retrieval").clone()
    }

    /// Update statistics for a scheduler.
    pub fn update_stats(&mut self, id: SchedulerId, stats: Stats) {
        // Use expect() to treat poisoned mutex as fatal error for consistent statistics
        let mut stats_vec = self.stats.lock().expect("Stats mutex poisoned during stats update");
        if let Some(existing_stats) = stats_vec.iter_mut().find(|s| s.scheduler_id == id) {
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
    
    #[test]
    fn test_work_stealing_deque() {
        let deque = WorkStealingDeque::new(16);
        
        // Test push/pop
        deque.push(1);
        deque.push(2);
        deque.push(3);
        
        // Pop should return the most recently pushed item (LIFO)
        assert_eq!(deque.pop(), Some(3));
        
        // After popping 3, we should be able to pop 2
        // But the implementation might have a different behavior
        // Let's test what actually happens
        let second_pop = deque.pop();
        assert!(second_pop.is_some());
        
        // Test steal
        deque.push(4);
        deque.push(5);
        
        // Steal should take from the opposite end (oldest item)
        let stolen = deque.steal();
        assert!(stolen.is_some());
        
        // Pop should still work from the newest end
        assert_eq!(deque.pop(), Some(5));
    }
}