//! Improved scheduler implementation with techniques from Rayon, Tokio, and OpenMP.
//! 
//! This module implements advanced scheduling techniques:
//! - Work-stealing deques (from Rayon)
//! - Async task notification (from Tokio)
//! - Low-overhead synchronization (from OpenMP/Fork Union)

use crate::{Task, BoxedTask, error::{SchedulerResult, SchedulerError}, Box, Vec};
use std::sync::atomic::{AtomicUsize, AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ptr;

/// Cache line size for padding to prevent false sharing
const CACHE_LINE_SIZE: usize = 64;

/// Padding helper to ensure cache line alignment
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

/// Chase-Lev work-stealing deque implementation (from Rayon)
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
    /// Stealer handle factory
    _phantom: std::marker::PhantomData<T>,
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
            _phantom: std::marker::PhantomData,
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
    }

    /// Pop a task (owner only)
    pub fn pop(&self) -> Option<T> {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let new_bottom = bottom.wrapping_sub(1);
        
        // Relaxed store is safe - only owner modifies bottom
        self.bottom.value.store(new_bottom, Ordering::Relaxed);
        
        // Synchronize with stealers
        std::sync::atomic::fence(Ordering::SeqCst);
        
        let top = self.top.value.load(Ordering::Relaxed);
        
        if top <= new_bottom {
            // Non-empty
            let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
            let task = unsafe { ptr::read(buffer.get(new_bottom)) };
            
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
            std::sync::atomic::fence(Ordering::SeqCst);
            
            let bottom = self.bottom.value.load(Ordering::Acquire);
            
            if top >= bottom {
                return None; // Empty
            }
            
            let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
            let task = unsafe { ptr::read(buffer.get(top)) };
            
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

/// Async task notification system (inspired by Tokio)
/// 
/// This provides efficient wakeup mechanisms for async tasks
/// without the overhead of condition variables
pub struct AsyncNotifier {
    /// State: 0 = idle, 1 = notified, 2 = polling
    state: AtomicUsize,
    /// Waker storage
    waker: Mutex<Option<std::task::Waker>>,
}

impl AsyncNotifier {
    pub fn new() -> Self {
        Self {
            state: AtomicUsize::new(0),
            waker: Mutex::new(None),
        }
    }

    /// Register a waker
    pub fn register(&self, waker: &std::task::Waker) {
        let mut guard = self.waker.lock().unwrap();
        
        // Only update if different
        if let Some(existing) = &*guard {
            if !existing.will_wake(waker) {
                *guard = Some(waker.clone());
            }
        } else {
            *guard = Some(waker.clone());
        }
    }

    /// Notify the task
    pub fn notify(&self) {
        // Fast path - try to set notified state
        match self.state.compare_exchange(
            0, // idle
            1, // notified
            Ordering::AcqRel,
            Ordering::Acquire
        ) {
            Ok(_) => {
                // Successfully notified idle task
                if let Some(waker) = self.waker.lock().unwrap().take() {
                    waker.wake();
                }
            }
            Err(current) => {
                if current == 2 {
                    // Task is polling - it will see the notification
                    self.state.store(1, Ordering::Release);
                }
                // Already notified - nothing to do
            }
        }
    }

    /// Check for notification
    pub fn is_notified(&self) -> bool {
        self.state.load(Ordering::Acquire) == 1
    }

    /// Clear notification
    pub fn clear(&self) {
        self.state.store(0, Ordering::Release);
    }
}

/// Low-overhead synchronization primitive (inspired by OpenMP/Fork Union)
/// 
/// This uses atomic increments instead of CAS operations where possible
/// to reduce contention and improve throughput
pub struct LowOverheadBarrier {
    /// Number of threads to synchronize
    num_threads: usize,
    /// Current generation
    generation: CachePadded<AtomicU64>,
    /// Number of threads that have arrived
    arrived: CachePadded<AtomicUsize>,
    /// Sense reversal flag
    sense: CachePadded<AtomicBool>,
}

impl LowOverheadBarrier {
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            generation: CachePadded { value: AtomicU64::new(0) },
            arrived: CachePadded { value: AtomicUsize::new(0) },
            sense: CachePadded { value: AtomicBool::new(false) },
        }
    }

    /// Wait at the barrier
    pub fn wait(&self) {
        let my_generation = self.generation.value.load(Ordering::Relaxed);
        let my_sense = self.sense.value.load(Ordering::Relaxed);
        
        // Increment arrived count
        let arrived = self.arrived.value.fetch_add(1, Ordering::AcqRel) + 1;
        
        if arrived == self.num_threads {
            // Last thread - reset and release others
            self.arrived.value.store(0, Ordering::Relaxed);
            self.generation.value.fetch_add(1, Ordering::Relaxed);
            self.sense.value.store(!my_sense, Ordering::Release);
        } else {
            // Wait for release
            while self.sense.value.load(Ordering::Acquire) == my_sense &&
                  self.generation.value.load(Ordering::Relaxed) == my_generation {
                std::hint::spin_loop();
            }
        }
    }
}

/// Improved scheduler with advanced techniques
pub struct ImprovedScheduler {
    /// Worker ID
    id: usize,
    /// Local work queue (work-stealing deque)
    local_queue: Arc<WorkStealingDeque<Box<dyn BoxedTask>>>,
    /// Global injector queue for external submissions
    injector: Arc<WorkStealingDeque<Box<dyn BoxedTask>>>,
    /// Notification system for async tasks
    notifier: Arc<AsyncNotifier>,
    /// Performance counters
    stats: CachePadded<SchedulerStats>,
}

#[derive(Default)]
struct SchedulerStats {
    tasks_executed: AtomicU64,
    tasks_stolen: AtomicU64,
    steal_attempts: AtomicU64,
    local_pushes: AtomicU64,
}

impl ImprovedScheduler {
    pub fn new(id: usize, capacity: usize) -> Self {
        Self {
            id,
            local_queue: Arc::new(WorkStealingDeque::new(capacity)),
            injector: Arc::new(WorkStealingDeque::new(capacity * 4)),
            notifier: Arc::new(AsyncNotifier::new()),
            stats: CachePadded { 
                value: SchedulerStats::default() 
            },
        }
    }

    /// Submit a task to this scheduler
    pub fn submit(&self, task: Box<dyn BoxedTask>) -> SchedulerResult<()> {
        // Try local queue first (fast path)
        self.local_queue.push(task);
        self.stats.value.local_pushes.fetch_add(1, Ordering::Relaxed);
        
        // Notify async runtime if needed
        self.notifier.notify();
        
        Ok(())
    }

    /// Get next task to execute
    pub fn next_task(&self) -> Option<Box<dyn BoxedTask>> {
        // Try local queue first
        if let Some(task) = self.local_queue.pop() {
            return Some(task);
        }
        
        // Try global injector
        if let Some(task) = self.injector.steal() {
            return Some(task);
        }
        
        // No tasks available
        None
    }

    /// Steal from another scheduler
    pub fn steal_from(&self, victim: &ImprovedScheduler) -> Option<Box<dyn BoxedTask>> {
        self.stats.value.steal_attempts.fetch_add(1, Ordering::Relaxed);
        
        if let Some(task) = victim.local_queue.steal() {
            self.stats.value.tasks_stolen.fetch_add(1, Ordering::Relaxed);
            Some(task)
        } else {
            None
        }
    }

    /// Get current load
    pub fn load(&self) -> usize {
        self.local_queue.len() + self.injector.len()
    }
}

// Safety: Tasks are Send + Sync
unsafe impl Send for ImprovedScheduler {}
unsafe impl Sync for ImprovedScheduler {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_stealing_deque() {
        let deque = WorkStealingDeque::new(16);
        
        // Test push/pop
        deque.push(1);
        deque.push(2);
        deque.push(3);
        
        assert_eq!(deque.pop(), Some(3));
        assert_eq!(deque.pop(), Some(2));
        
        // Test steal
        deque.push(4);
        deque.push(5);
        
        assert_eq!(deque.steal(), Some(1)); // Steals oldest
        assert_eq!(deque.pop(), Some(5));   // Pops newest
    }

    #[test]
    fn test_low_overhead_barrier() {
        use std::thread;
        use std::sync::Arc;
        
        let barrier = Arc::new(LowOverheadBarrier::new(4));
        let mut handles = vec![];
        
        for i in 0..4 {
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                println!("Thread {} waiting", i);
                barrier.wait();
                println!("Thread {} released", i);
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
}