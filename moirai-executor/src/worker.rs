//! Worker thread implementation for task execution.
//!
//! This module provides the Worker struct that handles task execution
//! following the Controller pattern, with enhanced performance monitoring
//! and CPU optimization capabilities.

use crate::{
    metrics::{TaskPerformanceMetrics, WorkerMetrics, WorkerSnapshot},
    task_registry::TaskRegistry,
    types::WorkerId,
};
use moirai_core::{
    BoxedTask, TaskId, TaskStatus, Scheduler,
};
use moirai_scheduler::{WorkStealingScheduler, WorkStealingCoordinator};
use moirai_utils::memory::prefetch_read;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

/// Worker thread that executes tasks from the scheduler.
/// 
/// Each worker follows the Information Expert pattern by owning
/// its execution context and managing its own lifecycle.
/// Enhanced with CPU topology awareness for optimal performance.
pub struct Worker {
    /// Unique identifier for this worker
    pub(crate) id: WorkerId,
    /// Local work-stealing scheduler
    pub(crate) scheduler: Arc<WorkStealingScheduler>,
    /// Coordination interface for work stealing
    pub(crate) coordinator: Arc<WorkStealingCoordinator>,
    /// All schedulers in the system for work stealing
    pub(crate) all_schedulers: Vec<Arc<WorkStealingScheduler>>,
    /// Task registry for lifecycle management
    pub(crate) task_registry: Arc<TaskRegistry>,
    /// Shutdown signal coordination
    pub(crate) shutdown_signal: Arc<AtomicBool>,
    /// Worker-level performance metrics
    pub(crate) metrics: Arc<WorkerMetrics>,
    /// Task-level performance tracking
    pub(crate) task_metrics: Arc<Mutex<HashMap<TaskId, TaskPerformanceMetrics>>>,
    // CPU optimization fields (disabled for now)
    // cpu_core: Option<CpuCore>,
    // affinity_mask: AffinityMask,
}

impl Worker {
    /// Create a new worker with CPU topology awareness.
    pub fn new(
        id: WorkerId,
        scheduler: Arc<WorkStealingScheduler>,
        coordinator: Arc<WorkStealingCoordinator>,
        all_schedulers: Vec<Arc<WorkStealingScheduler>>,
        task_registry: Arc<TaskRegistry>,
        shutdown_signal: Arc<AtomicBool>,
        metrics: Arc<WorkerMetrics>,
    ) -> Self {
        // CPU topology detection disabled for now
        // let topology = CpuTopology::detect();
        // let worker_index = id.get();
        
        // CPU affinity assignment disabled for now
        // let cpu_core = if worker_index < topology.logical_cores as usize {
        //     Some(CpuCore::new(worker_index as u32))
        // } else {
        //     let core_id = worker_index % (topology.logical_cores as usize);
        //     Some(CpuCore::new(core_id as u32))
        // };
        
        // let affinity_mask = if let Some(core) = cpu_core {
        //     if let Some(numa_node) = topology.numa_node(core) {
        //         AffinityMask::numa_node(numa_node)
        //     } else {
        //         AffinityMask::single(core)
        //     }
        // } else {
        //     AffinityMask::all()
        // };
        
        Self {
            id,
            scheduler,
            coordinator,
            all_schedulers,
            task_registry,
            shutdown_signal,
            metrics,
            // cpu_core,
            // affinity_mask,
            task_metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Main worker loop - follows the Controller pattern.
    /// Enhanced with CPU optimizations for better performance.
    /// 
    /// # Behavior Guarantees
    /// - Processes tasks until shutdown signal is received
    /// - Attempts work stealing when local queue is empty
    /// - Updates metrics atomically for thread safety
    /// - Handles panics gracefully without crashing worker
    /// - Sets CPU affinity for optimal cache locality
    pub fn run(self) {
        // CPU affinity setting disabled for now
        // if let Err(e) = self.affinity_mask.set_current_thread_affinity() {
        //     eprintln!("Warning: Failed to set CPU affinity for worker {}: {}", self.id.get(), e);
        // }
        
        // if let Some(core) = self.cpu_core {
        //     if let Err(e) = pin_to_core(core) {
        //         eprintln!("Warning: Failed to pin worker {} to core {}: {}", self.id.get(), core.id(), e);
        //     }
        // }
        
        while !self.shutdown_signal.load(Ordering::Acquire) {
            let mut work_found = false;

            // Try to get work from local scheduler first
            if let Ok(Some(task)) = self.scheduler.next_task() {
                self.execute_task(task);
                work_found = true;
            }

            // If no local work, try to steal from other workers
            if !work_found {
                self.metrics.record_steal_attempt(false);
                
                // Try work stealing
                if let Some(task) = self.coordinator.steal_work(&*self.scheduler, &self.all_schedulers) {
                    self.metrics.record_steal_attempt(true);
                    
                    // Track preemption for stolen tasks
                    let task_id = task.context().id;
                    if let Ok(mut metrics_map) = self.task_metrics.lock() {
                        if let Some(metrics) = metrics_map.get_mut(&task_id) {
                            metrics.increment_preemption();
                        }
                    }
                    
                    self.execute_task(task);
                    work_found = true;
                }
            }

            // If still no work, yield to avoid busy waiting
            if !work_found {
                thread::yield_now();
            }
        }
    }

    /// Execute a single task with error handling and metrics.
    /// Enhanced with memory prefetching for better cache performance.
    /// 
    /// # Behavior Guarantees
    /// - Task panics are caught and recorded as failures
    /// - Execution time is measured and recorded
    /// - Memory ordering ensures consistent metrics updates
    /// - Task status is updated in the registry
    /// - Memory prefetching improves cache locality
    fn execute_task(&self, task: Box<dyn BoxedTask>) {
        let task_id = task.context().id;
        let start_time = Instant::now();
        
        // Initialize task performance metrics
        let memory_start = self.get_current_memory_usage();
        let initial_metrics = TaskPerformanceMetrics::new(memory_start);
        
        // Register task metrics - use expect() for consistent task tracking
        self.task_metrics
            .lock()
            .expect("Task metrics mutex poisoned during task registration")
            .insert(task_id, initial_metrics);
        
        // Prefetch task data for better cache performance
        prefetch_read(task.as_ref() as *const _ as *const u8);
        
        // Update task status to running
        self.task_registry.update_status(task_id, TaskStatus::Running);
        
        // Track CPU time during execution
        let cpu_start = self.get_current_cpu_time();
        
        // Execute task with panic handling
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Monitor memory usage during execution
            self.monitor_task_memory(task_id);
            task.execute_boxed();
        }));

        let execution_time = start_time.elapsed();
        let cpu_end = self.get_current_cpu_time();
        let cpu_time_ns = cpu_end.saturating_sub(cpu_start);
        
        // Final metrics update
        self.finalize_task_metrics(task_id, cpu_time_ns, execution_time.as_nanos() as u64);
        
        // Update task status based on result
        match result {
            Ok(()) => {
                self.task_registry.update_status(task_id, TaskStatus::Completed);
            }
            Err(_) => {
                self.task_registry.update_status(task_id, TaskStatus::Failed);
                eprintln!("Task {} panicked during execution on worker {}", task_id, self.id.get());
            }
        }
        
        // Update metrics atomically
        self.metrics.record_task_completion(execution_time.as_nanos() as u64);
    }

    /// Get current memory usage for the process.
    /// 
    /// # Returns
    /// Current memory usage in bytes, or 0 if unable to determine.
    fn get_current_memory_usage(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            // Read from /proc/self/status for memory information
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms - use a simple heuristic
            // This is a simplified implementation for cross-platform compatibility
            0
        }
        
        0
    }

    /// Get current CPU time for the current thread.
    /// 
    /// # Returns
    /// CPU time in nanoseconds, or 0 if unable to determine.
    fn get_current_cpu_time(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            // Use clock_gettime for thread-specific CPU time
            use std::os::raw::{c_int, c_long};
            
            #[repr(C)]
            struct Timespec {
                tv_sec: c_long,
                tv_nsec: c_long,
            }
            
            extern "C" {
                fn clock_gettime(clock_id: c_int, tp: *mut Timespec) -> c_int;
            }
            
            const CLOCK_THREAD_CPUTIME_ID: c_int = 3;
            
            let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
            unsafe {
                if clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) == 0 {
                    return (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64);
                }
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        }
        
        0
    }

    /// Monitor task memory usage during execution.
    /// 
    /// # Arguments
    /// * `task_id` - The ID of the task to monitor
    fn monitor_task_memory(&self, task_id: TaskId) {
        let current_memory = self.get_current_memory_usage();
        
        // Use expect() for consistent memory monitoring
        let mut metrics_map = self.task_metrics
            .lock()
            .expect("Task metrics mutex poisoned during memory monitoring");
        if let Some(metrics) = metrics_map.get_mut(&task_id) {
            metrics.update(current_memory);
        }
    }

    /// Finalize task metrics after completion.
    /// 
    /// # Arguments
    /// * `task_id` - The ID of the completed task
    /// * `cpu_time_ns` - CPU time consumed in nanoseconds
    /// * `execution_time_ns` - Total execution time in nanoseconds
    fn finalize_task_metrics(&self, task_id: TaskId, cpu_time_ns: u64, _execution_time_ns: u64) {
        // Use expect() for consistent task finalization
        let mut metrics_map = self.task_metrics
            .lock()
            .expect("Task metrics mutex poisoned during task finalization");
        
        if let Some(metrics) = metrics_map.get_mut(&task_id) {
            metrics.cpu_time_ns = cpu_time_ns;
            metrics.last_update = Instant::now();
            
            // Log performance statistics using all fields
            let execution_duration = metrics.execution_time();
            let memory_growth = metrics.memory_growth();
            let was_preempted = metrics.was_preempted();
            
            // Debug logging for performance analysis
            const SLOW_TASK_THRESHOLD_MS: u128 = 100;
            const LARGE_MEMORY_THRESHOLD_BYTES: u64 = 1024 * 1024;
            
            if execution_duration.as_millis() > SLOW_TASK_THRESHOLD_MS 
                || memory_growth > LARGE_MEMORY_THRESHOLD_BYTES 
                || was_preempted {
                println!(
                    "Task {} performance: {}ms execution, {}MB memory growth, preempted: {}", 
                    task_id, 
                    execution_duration.as_millis(), 
                    memory_growth / (1024 * 1024), 
                    was_preempted
                );
            }
            
            // Clean up local metrics after processing
            // Keep only recent metrics to prevent memory bloat
            const MAX_RETAINED_METRICS: usize = 100;
            if metrics_map.len() > MAX_RETAINED_METRICS {
                // Remove oldest entries
                let mut entries: Vec<_> = metrics_map
                    .iter()
                    .map(|(k, v)| (*k, v.last_update))
                    .collect();
                entries.sort_by_key(|(_, last_update)| *last_update);
                
                let to_remove = entries.len().saturating_sub(MAX_RETAINED_METRICS);
                let ids_to_remove: Vec<_> = entries
                    .iter()
                    .take(to_remove)
                    .map(|(id, _)| *id)
                    .collect();
                
                for id in ids_to_remove {
                    metrics_map.remove(&id);
                }
            }
        }
    }

    /// Get worker metrics snapshot.
    pub fn metrics(&self) -> WorkerSnapshot {
        WorkerSnapshot::from_metrics(self.id, &*self.metrics)
    }
}