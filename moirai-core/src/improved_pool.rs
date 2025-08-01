//! Improved thread pool implementation with techniques from Tokio and OpenMP.
//! 
//! This module implements:
//! - Per-thread executors (from Tokio)
//! - Low-overhead task dispatch (from OpenMP)
//! - NUMA-aware thread pinning
//! - Adaptive spinning vs blocking

use crate::{
    improved_scheduler::{ImprovedScheduler, WorkStealingDeque, LowOverheadBarrier},
    BoxedTask, Task,
    error::{ExecutorResult, ExecutorError},
};
use std::sync::{Arc, Mutex, Condvar};
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::cell::RefCell;

/// Cache line padding
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

/// Thread-local storage for worker state (inspired by Tokio)
thread_local! {
    static WORKER: RefCell<Option<WorkerHandle>> = RefCell::new(None);
}

/// Handle to access current worker thread
#[derive(Clone)]
struct WorkerHandle {
    id: usize,
    scheduler: Arc<ImprovedScheduler>,
}

/// Worker thread state
struct Worker {
    /// Worker ID
    id: usize,
    /// Local scheduler
    scheduler: Arc<ImprovedScheduler>,
    /// Shared pool state
    pool: Arc<PoolState>,
    /// Thread handle
    handle: Option<JoinHandle<()>>,
}

/// Shared state for the thread pool
struct PoolState {
    /// Worker threads
    workers: Mutex<Vec<Arc<ImprovedScheduler>>>,
    /// Global task injector
    injector: Arc<WorkStealingDeque<Box<dyn BoxedTask>>>,
    /// Shutdown flag
    shutdown: AtomicBool,
    /// Number of active workers
    active_workers: CachePadded<AtomicUsize>,
    /// Number of sleeping workers
    sleeping_workers: CachePadded<AtomicUsize>,
    /// Condition variable for worker sleep/wake
    worker_condvar: Condvar,
    /// Worker mutex for condvar
    worker_mutex: Mutex<()>,
    /// Barrier for synchronization
    barrier: Option<Arc<LowOverheadBarrier>>,
    /// Performance metrics
    metrics: CachePadded<PoolMetrics>,
}

#[derive(Default)]
struct PoolMetrics {
    total_tasks: AtomicU64,
    total_steals: AtomicU64,
    total_parks: AtomicU64,
    total_unparks: AtomicU64,
}

/// Improved thread pool with advanced scheduling
pub struct ImprovedThreadPool {
    /// Pool state
    state: Arc<PoolState>,
    /// Worker threads
    workers: Vec<Worker>,
    /// Number of threads
    num_threads: usize,
}

impl ImprovedThreadPool {
    /// Create a new thread pool with the specified number of threads
    pub fn new(num_threads: usize) -> ExecutorResult<Self> {
        let num_threads = num_threads.max(1);
        
        let state = Arc::new(PoolState {
            workers: Mutex::new(Vec::with_capacity(num_threads)),
            injector: Arc::new(WorkStealingDeque::new(1024)),
            shutdown: AtomicBool::new(false),
            active_workers: CachePadded { value: AtomicUsize::new(0) },
            sleeping_workers: CachePadded { value: AtomicUsize::new(0) },
            worker_condvar: Condvar::new(),
            worker_mutex: Mutex::new(()),
            barrier: Some(Arc::new(LowOverheadBarrier::new(num_threads))),
            metrics: CachePadded { value: PoolMetrics::default() },
        });
        
        let mut workers = Vec::with_capacity(num_threads);
        let mut schedulers = Vec::with_capacity(num_threads);
        
        // Create schedulers first
        for id in 0..num_threads {
            let scheduler = Arc::new(ImprovedScheduler::new(id, 256));
            schedulers.push(scheduler.clone());
            
            workers.push(Worker {
                id,
                scheduler,
                pool: state.clone(),
                handle: None,
            });
        }
        
        // Register all schedulers
        {
            let mut guard = state.workers.lock().unwrap();
            *guard = schedulers;
        }
        
        // Start worker threads
        for worker in &mut workers {
            let id = worker.id;
            let scheduler = worker.scheduler.clone();
            let state = state.clone();
            
            let handle = thread::Builder::new()
                .name(format!("moirai-worker-{}", id))
                .spawn(move || {
                    // Set thread-local worker handle
                    WORKER.with(|w| {
                        *w.borrow_mut() = Some(WorkerHandle { id, scheduler: scheduler.clone() });
                    });
                    
                    // Pin to CPU if supported
                    #[cfg(target_os = "linux")]
                    {
                        unsafe {
                            let mut cpu_set: libc::cpu_set_t = std::mem::zeroed();
                            libc::CPU_SET(id % num_cpus::get(), &mut cpu_set);
                            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpu_set);
                        }
                    }
                    
                    // Run worker loop
                    worker_loop(id, scheduler, state);
                })
                .map_err(|_| ExecutorError::ResourceExhausted("Failed to spawn worker thread".into()))?;
            
            worker.handle = Some(handle);
        }
        
        Ok(Self {
            state,
            workers,
            num_threads,
        })
    }
    
    /// Submit a task to the pool
    pub fn submit<T>(&self, task: T) -> ExecutorResult<()>
    where
        T: Task + Send + 'static,
    {
        if self.state.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShutdownInProgress);
        }
        
        // Try to submit to current worker's queue (fast path)
        if let Some(handle) = WORKER.with(|w| w.borrow().clone()) {
            return handle.scheduler.submit(Box::new(task));
        }
        
        // Fall back to global injector
        self.state.injector.push(Box::new(task));
        self.state.metrics.value.total_tasks.fetch_add(1, Ordering::Relaxed);
        
        // Wake a sleeping worker if needed
        self.wake_worker();
        
        Ok(())
    }
    
    /// Synchronize all workers (barrier)
    pub fn barrier(&self) {
        if let Some(barrier) = &self.state.barrier {
            barrier.wait();
        }
    }
    
    /// Wake a sleeping worker
    fn wake_worker(&self) {
        let sleeping = self.state.sleeping_workers.load(Ordering::Acquire);
        if sleeping > 0 {
            self.state.metrics.value.total_unparks.fetch_add(1, Ordering::Relaxed);
            self.state.worker_condvar.notify_one();
        }
    }
    
    /// Get pool metrics
    pub fn metrics(&self) -> PoolMetricsSnapshot {
        PoolMetricsSnapshot {
            total_tasks: self.state.metrics.value.total_tasks.load(Ordering::Relaxed),
            total_steals: self.state.metrics.value.total_steals.load(Ordering::Relaxed),
            total_parks: self.state.metrics.value.total_parks.load(Ordering::Relaxed),
            total_unparks: self.state.metrics.value.total_unparks.load(Ordering::Relaxed),
            active_workers: self.state.active_workers.value.load(Ordering::Relaxed),
            sleeping_workers: self.state.sleeping_workers.value.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of pool metrics
#[derive(Debug, Clone)]
pub struct PoolMetricsSnapshot {
    pub total_tasks: u64,
    pub total_steals: u64,
    pub total_parks: u64,
    pub total_unparks: u64,
    pub active_workers: usize,
    pub sleeping_workers: usize,
}

/// Worker thread main loop
fn worker_loop(id: usize, scheduler: Arc<ImprovedScheduler>, state: Arc<PoolState>) {
    state.active_workers.value.fetch_add(1, Ordering::AcqRel);
    
    let mut spin_count = 0;
    const MAX_SPINS: u32 = 100;
    const PARK_TIMEOUT: Duration = Duration::from_millis(1);
    
    while !state.shutdown.load(Ordering::Acquire) {
        let mut found_work = false;
        
        // Try to get work from local queue
        if let Some(task) = scheduler.next_task() {
            execute_task(task);
            found_work = true;
            spin_count = 0;
            continue;
        }
        
        // Try to steal from global injector
        if let Some(task) = state.injector.steal() {
            execute_task(task);
            found_work = true;
            spin_count = 0;
            continue;
        }
        
        // Try to steal from other workers
        if let Some(task) = try_steal_from_others(id, &scheduler, &state) {
            execute_task(task);
            found_work = true;
            spin_count = 0;
            state.metrics.value.total_steals.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        
        // No work found - adaptive spinning vs parking
        if spin_count < MAX_SPINS {
            spin_count += 1;
            std::hint::spin_loop();
        } else {
            // Park the thread
            park_worker(&state);
            spin_count = 0;
        }
    }
    
    state.active_workers.value.fetch_sub(1, Ordering::AcqRel);
}

/// Try to steal work from other workers
fn try_steal_from_others(
    my_id: usize,
    my_scheduler: &Arc<ImprovedScheduler>,
    state: &Arc<PoolState>
) -> Option<Box<dyn BoxedTask>> {
    let workers = state.workers.lock().unwrap();
    let num_workers = workers.len();
    
    // Start from a random position to avoid patterns
    let start = fastrand::usize(0..num_workers);
    
    for i in 0..num_workers {
        let victim_id = (start + i) % num_workers;
        if victim_id == my_id {
            continue;
        }
        
        if let Some(task) = my_scheduler.steal_from(&workers[victim_id]) {
            return Some(task);
        }
    }
    
    None
}

/// Park the worker thread
fn park_worker(state: &Arc<PoolState>) {
    state.sleeping_workers.value.fetch_add(1, Ordering::AcqRel);
    state.metrics.value.total_parks.fetch_add(1, Ordering::Relaxed);
    
    let guard = state.worker_mutex.lock().unwrap();
    let _ = state.worker_condvar.wait_timeout(guard, Duration::from_millis(10)).unwrap();
    
    state.sleeping_workers.value.fetch_sub(1, Ordering::AcqRel);
}

/// Execute a task
fn execute_task(mut task: Box<dyn BoxedTask>) {
    task.execute();
}

impl Drop for ImprovedThreadPool {
    fn drop(&mut self) {
        // Signal shutdown
        self.state.shutdown.store(true, Ordering::Release);
        
        // Wake all workers
        self.state.worker_condvar.notify_all();
        
        // Join all threads
        for worker in self.workers.drain(..) {
            if let Some(handle) = worker.handle {
                let _ = handle.join();
            }
        }
    }
}

/// Fast random number generation (xorshift)
mod fastrand {
    use std::cell::Cell;
    
    thread_local! {
        static RNG: Cell<u64> = Cell::new({
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&std::thread::current().id(), &mut hasher);
            std::hash::Hasher::finish(&hasher)
        });
    }
    
    pub fn usize(range: std::ops::Range<usize>) -> usize {
        let mut x = RNG.with(|r| r.get());
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        RNG.with(|r| r.set(x));
        
        let span = range.end - range.start;
        range.start + (x as usize % span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    
    struct TestTask {
        counter: Arc<AtomicUsize>,
    }
    
    impl Task for TestTask {
        type Output = ();
        
        fn execute(self) -> Self::Output {
            self.counter.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    #[test]
    fn test_thread_pool_execution() {
        let pool = ImprovedThreadPool::new(4).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        
        // Submit tasks
        for _ in 0..100 {
            let task = TestTask { counter: counter.clone() };
            pool.submit(task).unwrap();
        }
        
        // Wait for completion
        thread::sleep(Duration::from_millis(100));
        
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }
    
    #[test]
    fn test_work_stealing() {
        let pool = ImprovedThreadPool::new(4).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        
        // Submit many tasks to trigger work stealing
        for _ in 0..1000 {
            let task = TestTask { counter: counter.clone() };
            pool.submit(task).unwrap();
        }
        
        // Wait for completion
        thread::sleep(Duration::from_millis(200));
        
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
        
        // Check that stealing occurred
        let metrics = pool.metrics();
        assert!(metrics.total_steals > 0);
    }
}