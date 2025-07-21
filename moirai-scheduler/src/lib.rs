//! Work-stealing scheduler implementation for Moirai concurrency library.

use moirai_core::{
    Task, BoxedTask, scheduler::{Scheduler, SchedulerId, SchedulerConfig, QueueType},
    error::SchedulerResult, Box,
};
use std::{
    sync::{
        atomic::{AtomicIsize, AtomicPtr, AtomicUsize, Ordering},
    },
    ptr,
    collections::VecDeque,
    sync::Mutex,
};

/// A lock-free work-stealing deque implementation based on the Chase-Lev algorithm.
pub struct ChaseLevDeque<T> {
    /// Bottom index (only modified by owner)
    bottom: AtomicIsize,
    /// Top index (modified by thieves)
    top: AtomicIsize,
    /// Array of task pointers
    array: AtomicPtr<Array<T>>,
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
        std::sync::atomic::fence(Ordering::Release);
        self.bottom.store(b + 1, Ordering::Relaxed);
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
                // Single item, compete with thieves
                if self.top.compare_exchange_weak(
                    t, t + 1, 
                    Ordering::SeqCst, 
                    Ordering::Relaxed
                ).is_err() {
                    // Lost the race, restore bottom
                    self.bottom.store(b + 1, Ordering::Relaxed);
                    return None;
                }
                self.bottom.store(b + 1, Ordering::Relaxed);
            }
            
            if !item_ptr.is_null() {
                let item = unsafe { Box::from_raw(item_ptr) };
                array.put(b, ptr::null_mut());
                return Some(*item);
            }
        } else {
            // Empty queue, restore bottom
            self.bottom.store(b + 1, Ordering::Relaxed);
        }
        
        None
    }

    /// Steal an item from the top of the deque (thief operation).
    pub fn steal(&self) -> Option<T> {
        let t = self.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);
        
        if t < b {
            let array_ptr = self.array.load(Ordering::Acquire);
            let array = unsafe { &*array_ptr };
            let item_ptr = array.get(t);
            
            if !item_ptr.is_null() {
                if self.top.compare_exchange_weak(
                    t, t + 1,
                    Ordering::SeqCst,
                    Ordering::Relaxed
                ).is_ok() {
                    let item = unsafe { Box::from_raw(item_ptr) };
                    return Some(*item);
                }
            }
        }
        
        None
    }

    /// Get the current size of the deque.
    pub fn size(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        (b - t).max(0) as usize
    }

    /// Check if the deque is empty.
    pub fn is_empty(&self) -> bool {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        b <= t
    }

    fn resize(&self) {
        let array_ptr = self.array.load(Ordering::Relaxed);
        let old_array = unsafe { &*array_ptr };
        let old_capacity = old_array.capacity();
        let new_capacity = old_capacity * 2;
        
        let new_array = Box::new(Array::new(new_capacity));
        
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        
        // Copy items from old array to new array
        for i in t..b {
            let item_ptr = old_array.get(i);
            new_array.put(i, item_ptr);
        }
        
        // Atomically replace the array
        let new_array_ptr = Box::into_raw(new_array);
        self.array.store(new_array_ptr, Ordering::Release);
        
        // Note: We're leaking the old array here for simplicity
        // In a production implementation, we'd use hazard pointers or epochs
        // to safely reclaim memory
    }
}

impl<T> Drop for ChaseLevDeque<T> {
    fn drop(&mut self) {
        // Drain all remaining items
        while let Some(_) = self.pop() {}
        
        // Clean up the array
        let array_ptr = self.array.load(Ordering::Relaxed);
        if !array_ptr.is_null() {
            unsafe {
                let _ = Box::from_raw(array_ptr);
            }
        }
    }
}

unsafe impl<T: Send> Send for ChaseLevDeque<T> {}
unsafe impl<T: Send> Sync for ChaseLevDeque<T> {}

/// A work-stealing scheduler implementation.
pub struct WorkStealingScheduler {
    id: SchedulerId,
    config: SchedulerConfig,
    /// Local task queue (Chase-Lev deque for lock-free work stealing)
    local_queue: ChaseLevDeque<Box<dyn BoxedTask>>,
    /// Fallback queue for when lock-free operations fail
    fallback_queue: Mutex<VecDeque<Box<dyn BoxedTask>>>,
    /// Statistics
    tasks_scheduled: AtomicUsize,
    tasks_completed: AtomicUsize,
    steal_attempts: AtomicUsize,
    successful_steals: AtomicUsize,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler.
    pub fn new(id: SchedulerId, config: SchedulerConfig) -> Self {
        let initial_capacity = match config.queue_type {
            QueueType::ChaseLev => 256,
            _ => 64,
        };

        Self {
            id,
            config,
            local_queue: ChaseLevDeque::new(initial_capacity),
            fallback_queue: Mutex::new(VecDeque::new()),
            tasks_scheduled: AtomicUsize::new(0),
            tasks_completed: AtomicUsize::new(0),
            steal_attempts: AtomicUsize::new(0),
            successful_steals: AtomicUsize::new(0),
        }
    }

    /// Get statistics for this scheduler.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            id: self.id,
            queue_length: self.load(),
            tasks_scheduled: self.tasks_scheduled.load(Ordering::Relaxed) as u64,
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed) as u64,
            steal_context: moirai_core::scheduler::StealContext {
                attempts: self.steal_attempts.load(Ordering::Relaxed),
                successes: self.successful_steals.load(Ordering::Relaxed),
                items_stolen: self.successful_steals.load(Ordering::Relaxed),
                avg_steal_latency_ns: 0, // TODO: Implement latency tracking
            },
            avg_task_execution_us: 0.0, // TODO: Implement execution time tracking
            cpu_utilization: 0.0, // TODO: Implement CPU utilization tracking
        }
    }

    /// Try to execute the next available task.
    pub fn try_execute_next(&self) -> SchedulerResult<bool> {
        if let Some(task) = self.next_task()? {
            // Execute the task using the BoxedTask trait
            task.execute_boxed();
            self.tasks_completed.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Execute tasks in a loop until the queue is empty.
    pub fn run_until_empty(&self) -> SchedulerResult<usize> {
        let mut executed = 0;
        while self.try_execute_next()? {
            executed += 1;
        }
        Ok(executed)
    }
}

impl Scheduler for WorkStealingScheduler {
    fn schedule_task(&self, task: Box<dyn BoxedTask>) -> SchedulerResult<()> {
        // Check queue size limit
        if self.load() >= self.config.max_queue_size {
            return Err(moirai_core::error::SchedulerError::QueueFull);
        }

        match self.config.queue_type {
            QueueType::ChaseLev => {
                self.local_queue.push(task);
            }
            _ => {
                // Fallback to locked queue for other types
                let mut fallback = self.fallback_queue.lock().unwrap();
                fallback.push_back(task);
            }
        }

        self.tasks_scheduled.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn next_task(&self) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        match self.config.queue_type {
            QueueType::ChaseLev => {
                // Try the lock-free queue first
                if let Some(task) = self.local_queue.pop() {
                    return Ok(Some(task));
                }

                // Fall back to the locked queue
                let mut fallback = self.fallback_queue.lock().unwrap();
                Ok(fallback.pop_front())
            }
            _ => {
                let mut fallback = self.fallback_queue.lock().unwrap();
                Ok(fallback.pop_front())
            }
        }
    }

    fn try_steal(&self, _victim: &dyn Scheduler) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        self.steal_attempts.fetch_add(1, Ordering::Relaxed);

        match self.config.queue_type {
            QueueType::ChaseLev => {
                if let Some(task) = self.local_queue.steal() {
                    self.successful_steals.fetch_add(1, Ordering::Relaxed);
                    return Ok(Some(task));
                }
            }
            _ => {
                // For other queue types, we can't steal efficiently
                return Ok(None);
            }
        }

        Ok(None)
    }

    fn load(&self) -> usize {
        let local_size = match self.config.queue_type {
            QueueType::ChaseLev => self.local_queue.size(),
            _ => 0,
        };

        let fallback_size = self.fallback_queue.lock().unwrap().len();
        local_size + fallback_size
    }

    fn id(&self) -> SchedulerId {
        self.id
    }

    fn can_be_stolen_from(&self) -> bool {
        match self.config.queue_type {
            QueueType::ChaseLev => self.load() > 1,
            _ => false, // Other queue types don't support stealing
        }
    }
}

/// A local scheduler for single-threaded execution.
pub struct LocalScheduler {
    id: SchedulerId,
    queue: Mutex<VecDeque<Box<dyn BoxedTask>>>,
    tasks_scheduled: AtomicUsize,
    tasks_completed: AtomicUsize,
}

impl LocalScheduler {
    /// Create a new local scheduler.
    pub fn new(id: SchedulerId) -> Self {
        Self {
            id,
            queue: Mutex::new(VecDeque::new()),
            tasks_scheduled: AtomicUsize::new(0),
            tasks_completed: AtomicUsize::new(0),
        }
    }

    /// Execute all tasks in the queue.
    pub fn run_to_completion(&self) -> SchedulerResult<usize> {
        let mut executed = 0;
        while let Some(task) = self.next_task()? {
            // Execute the task using the BoxedTask trait
            task.execute_boxed();
            self.tasks_completed.fetch_add(1, Ordering::Relaxed);
            executed += 1;
        }
        Ok(executed)
    }

    /// Get statistics for this scheduler.
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            id: self.id,
            queue_length: self.load(),
            tasks_scheduled: self.tasks_scheduled.load(Ordering::Relaxed) as u64,
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed) as u64,
            steal_context: moirai_core::scheduler::StealContext::default(),
            avg_task_execution_us: 0.0,
            cpu_utilization: 0.0,
        }
    }
}

impl Scheduler for LocalScheduler {
    fn schedule_task(&self, task: Box<dyn BoxedTask>) -> SchedulerResult<()> {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(task);
        self.tasks_scheduled.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn next_task(&self) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        let mut queue = self.queue.lock().unwrap();
        Ok(queue.pop_front())
    }

    fn try_steal(&self, _victim: &dyn Scheduler) -> SchedulerResult<Option<Box<dyn BoxedTask>>> {
        // Local schedulers don't participate in work stealing
        Ok(None)
    }

    fn load(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    fn id(&self) -> SchedulerId {
        self.id
    }

    fn can_be_stolen_from(&self) -> bool {
        false // Local schedulers cannot be stolen from
    }
}

// Re-export from core for convenience
pub use moirai_core::scheduler::SchedulerStats;

#[cfg(test)]
mod tests {
    use super::*;
    use moirai_core::{TaskBuilder, TaskId};
    use std::sync::Arc;

    struct TestTask {
        id: moirai_core::TaskId,
        value: i32,
        executed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl moirai_core::Task for TestTask {
        type Output = i32;

        fn execute(self) -> Self::Output {
            self.executed.store(true, Ordering::Relaxed);
            self.value * 2
        }

        fn context(&self) -> &moirai_core::TaskContext {
            // Create a static context for testing
            static CONTEXT: std::sync::OnceLock<moirai_core::TaskContext> = std::sync::OnceLock::new();
            CONTEXT.get_or_init(|| moirai_core::TaskContext::new(self.id))
        }
    }

    #[test]
    fn test_chase_lev_deque_basic_operations() {
        let deque = ChaseLevDeque::new(16);
        
        // Test empty
        assert!(deque.is_empty());
        assert_eq!(deque.size(), 0);
        assert!(deque.pop().is_none());
        assert!(deque.steal().is_none());

        // Test push/pop
        deque.push(42);
        assert!(!deque.is_empty());
        assert_eq!(deque.size(), 1);
        
        assert_eq!(deque.pop(), Some(42));
        assert!(deque.is_empty());

        // Test multiple items
        for i in 0..10 {
            deque.push(i);
        }
        assert_eq!(deque.size(), 10);

        // Test LIFO order for pop
        for i in (0..10).rev() {
            assert_eq!(deque.pop(), Some(i));
        }
    }

    #[test]
    fn test_chase_lev_deque_steal() {
        let deque = ChaseLevDeque::new(16);
        
        // Push some items
        for i in 0..5 {
            deque.push(i);
        }

        // Steal should get items in FIFO order
        assert_eq!(deque.steal(), Some(0));
        assert_eq!(deque.steal(), Some(1));
        assert_eq!(deque.size(), 3);

        // Pop should still work on remaining items
        assert_eq!(deque.pop(), Some(4));
        assert_eq!(deque.size(), 2);
    }

    #[test]
    fn test_work_stealing_scheduler() {
        let config = SchedulerConfig::default();
        let scheduler = WorkStealingScheduler::new(SchedulerId::new(0), config);

        // Test task scheduling
        let executed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = TestTask {
            id: TaskId::new(1),
            value: 21,
            executed: executed.clone(),
        };

        // Wrap in a closure that returns ()
        let task_closure = TaskBuilder::new()
            .with_id(TaskId::new(1))
            .build(move || { task.execute(); });
        let task_box: Box<dyn BoxedTask> = Box::new(moirai_core::task::TaskWrapper::new(task_closure));

        scheduler.schedule_task(task_box).unwrap();
        assert_eq!(scheduler.load(), 1);

        // Execute the task
        assert!(scheduler.try_execute_next().unwrap());
        assert_eq!(scheduler.load(), 0);
    }

    #[test]
    fn test_local_scheduler() {
        let scheduler = LocalScheduler::new(SchedulerId::new(1));

        // Test task scheduling and execution
        let executed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = TestTask {
            id: TaskId::new(1),
            value: 42,
            executed: executed.clone(),
        };

        let task_closure = TaskBuilder::new()
            .with_id(TaskId::new(1))
            .build(move || { task.execute(); });
        let task_box: Box<dyn BoxedTask> = Box::new(moirai_core::task::TaskWrapper::new(task_closure));

        scheduler.schedule_task(task_box).unwrap();
        assert_eq!(scheduler.load(), 1);

        let executed_count = scheduler.run_to_completion().unwrap();
        assert_eq!(executed_count, 1);
        assert_eq!(scheduler.load(), 0);
    }

    #[test]
    fn test_scheduler_stats() {
        let config = SchedulerConfig::default();
        let scheduler = WorkStealingScheduler::new(SchedulerId::new(0), config);

        let stats = scheduler.stats();
        assert_eq!(stats.id, SchedulerId::new(0));
        assert_eq!(stats.queue_length, 0);
        assert_eq!(stats.tasks_scheduled, 0);
        assert_eq!(stats.tasks_completed, 0);
    }
}