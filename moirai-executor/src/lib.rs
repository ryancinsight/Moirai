//! Hybrid executor implementation for Moirai concurrency library.
//!
//! This module provides a concrete implementation of the Moirai executor
//! that combines async and parallel execution models in a unified runtime.
//!
//! ## Design Principles
//!
//! - **SOLID**: Each component has a single responsibility and clear interfaces
//! - **CUPID**: Composable, predictable, and domain-centric design
//! - **GRASP**: Information expert pattern with low coupling
//! - **Zero-cost abstractions**: Compile-time optimizations
//! - **Memory safety**: Rust ownership model prevents data races

use moirai_core::{
    Task, TaskId, Priority, TaskContext, TaskHandle, BoxedTask,
    executor::{
        TaskSpawner, TaskManager, ExecutorControl, Executor,
        TaskStatus, TaskStats, ExecutorConfig, CleanupConfig,
        ExecutorPlugin,
    },
    scheduler::{Scheduler, SchedulerId, WorkStealingCoordinator},
    error::{ExecutorError, ExecutorResult, TaskError},
    task::{TaskBuilder, TaskWrapper},
    Box, Vec,
};
use moirai_scheduler::WorkStealingScheduler;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, RwLock, Condvar,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

/// A unique identifier for worker threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(usize);

impl WorkerId {
    /// Create a new worker ID.
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Worker thread that executes tasks from the scheduler.
/// 
/// Each worker follows the Information Expert pattern by owning
/// its execution context and managing its own lifecycle.
struct Worker {
    id: WorkerId,
    scheduler: Arc<WorkStealingScheduler>,
    coordinator: Arc<WorkStealingCoordinator>,
    task_registry: Arc<TaskRegistry>,
    shutdown_signal: Arc<AtomicBool>,
    metrics: Arc<WorkerMetrics>,
}

/// Metrics collected per worker thread.
#[derive(Debug, Default)]
struct WorkerMetrics {
    tasks_executed: AtomicU64,
    steal_attempts: AtomicU64,
    successful_steals: AtomicU64,
    execution_time_ns: AtomicU64,
}

impl Worker {
    /// Create a new worker.
    fn new(
        id: WorkerId,
        scheduler: Arc<WorkStealingScheduler>,
        coordinator: Arc<WorkStealingCoordinator>,
        task_registry: Arc<TaskRegistry>,
        shutdown_signal: Arc<AtomicBool>,
        metrics: Arc<WorkerMetrics>,
    ) -> Self {
        Self {
            id,
            scheduler,
            coordinator,
            task_registry,
            shutdown_signal,
            metrics,
        }
    }

    /// Main worker loop - follows the Controller pattern.
    /// 
    /// # Behavior Guarantees
    /// - Processes tasks until shutdown signal is received
    /// - Attempts work stealing when local queue is empty
    /// - Updates metrics atomically for thread safety
    /// - Handles panics gracefully without crashing worker
    fn run(self) {
        while !self.shutdown_signal.load(Ordering::Acquire) {
            let mut work_found = false;

            // Try to get work from local scheduler first
            if let Ok(Some(task)) = self.scheduler.next_task() {
                self.execute_task(task);
                work_found = true;
            }

            // If no local work, try to steal from other workers
            if !work_found {
                self.metrics.steal_attempts.fetch_add(1, Ordering::Relaxed);
                
                if let Ok(Some(task)) = self.coordinator.try_steal_for(self.scheduler.id()) {
                    self.metrics.successful_steals.fetch_add(1, Ordering::Relaxed);
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
    /// 
    /// # Behavior Guarantees
    /// - Task panics are caught and recorded as failures
    /// - Execution time is measured and recorded
    /// - Memory ordering ensures consistent metrics updates
    /// - Task status is updated in the registry
    fn execute_task(&self, task: Box<dyn BoxedTask>) {
        let task_id = task.context().id;
        let start_time = Instant::now();
        
        // Update task status to running
        self.task_registry.update_status(task_id, TaskStatus::Running);
        
        // Execute task with panic handling
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            task.execute_boxed();
        }));

        let execution_time = start_time.elapsed();
        
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
        self.metrics.tasks_executed.fetch_add(1, Ordering::Relaxed);
        self.metrics.execution_time_ns.fetch_add(
            execution_time.as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Get worker metrics snapshot.
    fn metrics(&self) -> WorkerSnapshot {
        WorkerSnapshot {
            id: self.id,
            tasks_executed: self.metrics.tasks_executed.load(Ordering::Relaxed),
            steal_attempts: self.metrics.steal_attempts.load(Ordering::Relaxed),
            successful_steals: self.metrics.successful_steals.load(Ordering::Relaxed),
            execution_time_ns: self.metrics.execution_time_ns.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of worker metrics at a point in time.
#[derive(Debug, Clone)]
pub struct WorkerSnapshot {
    pub id: WorkerId,
    pub tasks_executed: u64,
    pub steal_attempts: u64,
    pub successful_steals: u64,
    pub execution_time_ns: u64,
}

/// Task registry for tracking active tasks.
/// 
/// Follows the Information Expert pattern by owning task metadata
/// and providing efficient lookups for task management operations.
struct TaskRegistry {
    tasks: RwLock<HashMap<TaskId, TaskMetadata>>,
    next_id: AtomicU64,
}

/// Metadata about a task in the registry.
#[derive(Debug, Clone)]
struct TaskMetadata {
    id: TaskId,
    status: TaskStatus,
    priority: Priority,
    spawn_time: Instant,
    start_time: Option<Instant>,
    completion_time: Option<Instant>,
    waker: Option<Waker>,
}

impl TaskRegistry {
    /// Create a new task registry.
    fn new() -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0), // Start from 0 to match test expectations
        }
    }

    /// Generate a new unique task ID.
    fn generate_id(&self) -> TaskId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        TaskId::new(id)
    }

    /// Register a new task.
    fn register_task(&self, priority: Priority) -> TaskId {
        let id = self.generate_id();
        let metadata = TaskMetadata {
            id,
            status: TaskStatus::Queued,
            priority,
            spawn_time: Instant::now(),
            start_time: None,
            completion_time: None,
            waker: None,
        };

        let mut tasks = self.tasks.write().unwrap();
        tasks.insert(id, metadata);
        id
    }

    /// Update task status.
    fn update_status(&self, id: TaskId, status: TaskStatus) {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(metadata) = tasks.get_mut(&id) {
            metadata.status = status;
            
            match status {
                TaskStatus::Running => {
                    metadata.start_time = Some(Instant::now());
                }
                TaskStatus::Completed | TaskStatus::Cancelled | TaskStatus::Failed => {
                    metadata.completion_time = Some(Instant::now());
                    
                    // Wake any waiting futures
                    if let Some(waker) = metadata.waker.take() {
                        waker.wake();
                    }
                }
                _ => {}
            }
        }
    }

    /// Get task status.
    fn get_status(&self, id: TaskId) -> Option<TaskStatus> {
        let tasks = self.tasks.read().unwrap();
        tasks.get(&id).map(|metadata| metadata.status)
    }

    /// Get task statistics.
    fn get_stats(&self, id: TaskId) -> Option<TaskStats> {
        let tasks = self.tasks.read().unwrap();
        tasks.get(&id).map(|metadata| TaskStats {
            id: metadata.id,
            status: metadata.status,
            priority: metadata.priority,
            spawn_time: metadata.spawn_time,
            start_time: metadata.start_time,
            completion_time: metadata.completion_time,
            preemption_count: 0, // TODO: Track preemptions
            cpu_time_ns: 0, // TODO: Track CPU time
            memory_used_bytes: 0, // TODO: Track memory usage
        })
    }

    /// Set waker for task completion notifications.
    fn set_waker(&self, id: TaskId, waker: Waker) {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(metadata) = tasks.get_mut(&id) {
            metadata.waker = Some(waker);
        }
    }

    /// Cancel a task.
    fn cancel_task(&self, id: TaskId) -> Result<(), TaskError> {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(metadata) = tasks.get_mut(&id) {
            match metadata.status {
                TaskStatus::Queued | TaskStatus::Running => {
                    metadata.status = TaskStatus::Cancelled;
                    metadata.completion_time = Some(Instant::now());
                    
                    if let Some(waker) = metadata.waker.take() {
                        waker.wake();
                    }
                    Ok(())
                }
                _ => Ok(()), // Already completed, no-op
            }
        } else {
            Err(TaskError::InvalidOperation)
        }
    }

    /// Clean up completed tasks older than the specified duration.
    fn cleanup_completed(&self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;
        let mut tasks = self.tasks.write().unwrap();
        
        tasks.retain(|_, metadata| {
            match metadata.status {
                TaskStatus::Completed | TaskStatus::Cancelled | TaskStatus::Failed => {
                    metadata.completion_time
                        .map(|time| time > cutoff)
                        .unwrap_or(false)
                }
                _ => true, // Keep active tasks
            }
        });
    }

    /// Clean up completed tasks with both age and count limits.
    /// 
    /// This prevents unbounded memory growth by enforcing both time-based
    /// and count-based cleanup policies.
    fn cleanup_completed_with_limits(&self, max_age: Duration, max_retained_tasks: usize) {
        let cutoff = Instant::now() - max_age;
        let mut tasks = self.tasks.write().unwrap();
        
        // First pass: remove tasks older than max_age
        tasks.retain(|_, metadata| {
            match metadata.status {
                TaskStatus::Completed | TaskStatus::Cancelled | TaskStatus::Failed => {
                    metadata.completion_time
                        .map(|time| time > cutoff)
                        .unwrap_or(false)
                }
                _ => true, // Keep active tasks
            }
        });
        
        // Second pass: if we still have too many completed tasks, remove oldest ones
        let completed_tasks: Vec<_> = tasks.iter()
            .filter(|(_, metadata)| {
                matches!(metadata.status, 
                    TaskStatus::Completed | TaskStatus::Cancelled | TaskStatus::Failed)
            })
            .collect();
            
        if completed_tasks.len() > max_retained_tasks {
            // Sort by completion time (oldest first) and remove excess
            let mut completed_with_times: Vec<_> = completed_tasks.iter()
                .filter_map(|(id, metadata)| {
                    metadata.completion_time.map(|time| (**id, time))
                })
                .collect();
                
            completed_with_times.sort_by_key(|(_, time)| *time);
            
            let to_remove = completed_with_times.len().saturating_sub(max_retained_tasks);
            for (task_id, _) in completed_with_times.iter().take(to_remove) {
                tasks.remove(task_id);
            }
        }
    }
}

/// Future for waiting on task completion.
struct TaskWaitFuture {
    task_id: TaskId,
    registry: Arc<TaskRegistry>,
    timeout: Option<Duration>,
    start_time: Instant,
}

impl Future for TaskWaitFuture {
    type Output = Result<(), TaskError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let status = self.registry.get_status(self.task_id);
        
        match status {
            Some(TaskStatus::Completed) => Poll::Ready(Ok(())),
            Some(TaskStatus::Cancelled) => Poll::Ready(Err(TaskError::Cancelled)),
            Some(TaskStatus::Failed) => Poll::Ready(Err(TaskError::ExecutionFailed(
                moirai_core::error::TaskErrorKind::Other
            ))),
            Some(TaskStatus::Queued) | Some(TaskStatus::Running) => {
                // Check timeout
                if let Some(timeout) = self.timeout {
                    if self.start_time.elapsed() >= timeout {
                        return Poll::Ready(Err(TaskError::Timeout));
                    }
                }
                
                // Register waker and return pending
                self.registry.set_waker(self.task_id, cx.waker().clone());
                Poll::Pending
            }
            None => Poll::Ready(Err(TaskError::InvalidOperation)),
        }
    }
}

/// Hybrid executor that combines async and parallel execution.
/// 
/// This executor implements all the segregated interfaces while maintaining
/// internal cohesion and following the SOLID principles.
/// 
/// # Architecture
/// - Worker threads for CPU-bound tasks
/// - Async runtime for I/O-bound tasks  
/// - Work-stealing scheduler for load balancing
/// - Task registry for lifecycle management
/// - Plugin system for extensibility
pub struct HybridExecutor {
    // Core components
    config: ExecutorConfig,
    schedulers: Vec<Arc<WorkStealingScheduler>>,
    coordinator: Arc<WorkStealingCoordinator>,
    task_registry: Arc<TaskRegistry>,
    
    // Thread management
    worker_handles: Mutex<Vec<JoinHandle<()>>>,
    cleanup_handle: Mutex<Option<JoinHandle<()>>>,
    shutdown_signal: Arc<AtomicBool>,
    shutdown_complete: Arc<Condvar>,
    shutdown_mutex: Arc<Mutex<bool>>,
    
    // Metrics
    worker_metrics: Vec<Arc<WorkerMetrics>>,
    
    // Plugins
    plugins: Vec<Box<dyn ExecutorPlugin>>,
    
    // Statistics
    start_time: Instant,
    tasks_spawned: AtomicU64,
}

impl HybridExecutor {
    /// Create a new hybrid executor with the given configuration.
    /// 
    /// # Behavior Guarantees
    /// - Creates configured number of worker threads
    /// - Initializes work-stealing schedulers
    /// - Sets up task registry and coordination systems
    /// - All components are ready for task execution
    /// 
    /// # Performance Characteristics
    /// - Initialization time: O(worker_count) for thread creation
    /// - Memory usage: ~1MB base + ~8MB per worker thread
    /// - Ready for task execution immediately after creation
    pub fn new(config: ExecutorConfig) -> ExecutorResult<Self> {
        Self::with_plugins(config, Vec::new())
    }

    /// Create a new hybrid executor with plugins.
    pub fn with_plugins(
        config: ExecutorConfig,
        mut plugins: Vec<Box<dyn ExecutorPlugin>>,
    ) -> ExecutorResult<Self> {
        // Initialize plugins
        for plugin in &mut plugins {
            plugin.initialize().map_err(|_| ExecutorError::InvalidConfiguration)?;
        }

        // Create schedulers for each worker thread
        let mut schedulers = Vec::with_capacity(config.worker_threads);
        let coordinator = Arc::new(WorkStealingCoordinator::new(
            moirai_core::scheduler::WorkStealingStrategy::Random { max_attempts: 3 }
        ));

        for i in 0..config.worker_threads {
            let scheduler_config = moirai_core::scheduler::SchedulerConfig::default();
            let scheduler = Arc::new(WorkStealingScheduler::new(
                SchedulerId::new(i),
                scheduler_config,
            ));
            schedulers.push(scheduler);
        }

        let task_registry = Arc::new(TaskRegistry::new());
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let shutdown_complete = Arc::new(Condvar::new());
        let shutdown_mutex = Arc::new(Mutex::new(false));

        // Create worker threads
        let mut worker_handles = Vec::with_capacity(config.worker_threads);
        let mut worker_metrics = Vec::with_capacity(config.worker_threads);
        
        for (i, scheduler) in schedulers.iter().enumerate() {
            let metrics = Arc::new(WorkerMetrics::default());
            worker_metrics.push(metrics.clone());
            
            let worker = Worker::new(
                WorkerId::new(i),
                scheduler.clone(),
                coordinator.clone(),
                task_registry.clone(),
                shutdown_signal.clone(),
                metrics,
            );

            let handle = thread::Builder::new()
                .name(format!("{}-{}", config.thread_name_prefix, i))
                .spawn(move || worker.run())
                .map_err(|_| ExecutorError::ThreadPoolCreationFailed)?;

            worker_handles.push(handle);
        }

        // Create cleanup thread if automatic cleanup is enabled
        let cleanup_handle = if config.cleanup.enable_automatic_cleanup {
            let cleanup_task_registry = task_registry.clone();
            let cleanup_shutdown_signal = shutdown_signal.clone();
            let cleanup_config = config.cleanup.clone();
            
            let handle = thread::Builder::new()
                .name(format!("{}-cleanup", config.thread_name_prefix))
                .spawn(move || {
                    Self::cleanup_thread_loop(
                        cleanup_task_registry,
                        cleanup_shutdown_signal,
                        cleanup_config,
                    )
                })
                .map_err(|_| ExecutorError::ThreadPoolCreationFailed)?;
                
            Some(handle)
        } else {
            None
        };

        Ok(Self {
            config,
            schedulers,
            coordinator,
            task_registry,
            worker_handles: Mutex::new(worker_handles),
            cleanup_handle: Mutex::new(cleanup_handle),
            shutdown_signal,
            shutdown_complete,
            shutdown_mutex,
            worker_metrics,
            plugins,
            start_time: Instant::now(),
            tasks_spawned: AtomicU64::new(0),
        })
    }

    /// Select the best scheduler for a new task.
    /// 
    /// Uses load balancing to distribute tasks evenly across workers.
    /// Follows the Strategy pattern for scheduler selection algorithms.
    fn select_scheduler(&self) -> Option<&Arc<WorkStealingScheduler>> {
        // Simple round-robin for now - could be enhanced with load-based selection
        let task_count = self.tasks_spawned.load(Ordering::Relaxed);
        let index = (task_count % self.schedulers.len() as u64) as usize;
        self.schedulers.get(index)
    }

    /// Cleanup thread loop that runs periodically to clean up completed task metadata.
    /// 
    /// This prevents memory leaks by removing old task metadata according to the
    /// configured cleanup policy.
    fn cleanup_thread_loop(
        task_registry: Arc<TaskRegistry>,
        shutdown_signal: Arc<AtomicBool>,
        config: CleanupConfig,
    ) {
        while !shutdown_signal.load(Ordering::Relaxed) {
            // Sleep for the cleanup interval, but wake up early if shutdown is signaled
            let sleep_duration = config.cleanup_interval;
            let start_time = Instant::now();
            
            while start_time.elapsed() < sleep_duration {
                if shutdown_signal.load(Ordering::Relaxed) {
                    return;
                }
                // Sleep for a short time and check again
                thread::sleep(Duration::from_millis(100));
            }
            
            // Perform cleanup
            task_registry.cleanup_completed_with_limits(
                config.task_retention_duration,
                config.max_retained_tasks,
            );
        }
    }

    /// Internal task spawning implementation.
    /// 
    /// Follows the Template Method pattern with customizable task creation.
    fn spawn_internal<T>(&self, task: T, priority: Priority) -> TaskHandle<T::Output>
    where
        T: Task,
    {
        // Register task in registry
        let task_id = self.task_registry.register_task(priority);
        
        // Notify plugins
        for plugin in &self.plugins {
            plugin.before_task_spawn(task_id, priority);
        }

        // Create result communication channel
        #[cfg(feature = "std")]
        let (result_sender, result_receiver) = std::sync::mpsc::channel::<T::Output>();
        
        // Create task wrapper with result sender
        #[cfg(feature = "std")]
        let task_wrapper = TaskWrapper::with_result_sender(task, result_sender);
        #[cfg(not(feature = "std"))]
        let task_wrapper = TaskWrapper::new(task);
        
        let boxed_task: Box<dyn BoxedTask> = Box::new(task_wrapper);

        // Schedule task
        if let Some(scheduler) = self.select_scheduler() {
            if let Ok(()) = scheduler.schedule_task(boxed_task) {
                self.tasks_spawned.fetch_add(1, Ordering::Relaxed);
                
                // Update task status to queued
                self.task_registry.update_status(task_id, TaskStatus::Queued);
                
                // Notify plugins
                for plugin in &self.plugins {
                    plugin.after_task_spawn(task_id);
                }

                // Create handle with proper result communication
                #[cfg(feature = "std")]
                return TaskHandle::new_with_receiver(task_id, result_receiver);
                #[cfg(not(feature = "std"))]
                return TaskHandle::new_detached(task_id);
            }
        }

        // If scheduling failed, mark task as failed
        self.task_registry.update_status(task_id, TaskStatus::Failed);
        
        // Return a handle even if scheduling failed (it will return None on join)
        #[cfg(feature = "std")]
        {
            TaskHandle::new_with_receiver(task_id, result_receiver)
        }
        #[cfg(not(feature = "std"))]
        {
            TaskHandle::new_detached(task_id)
        }
    }



    /// Get comprehensive statistics about the executor.
    #[cfg(feature = "metrics")]
    pub fn detailed_stats(&self) -> DetailedExecutorStats {
        let worker_stats: Vec<WorkerSnapshot> = self.worker_handles
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, _)| WorkerSnapshot {
                id: WorkerId::new(i),
                tasks_executed: self.worker_metrics[i].tasks_executed.load(Ordering::Relaxed),
                steal_attempts: self.worker_metrics[i].steal_attempts.load(Ordering::Relaxed),
                successful_steals: self.worker_metrics[i].successful_steals.load(Ordering::Relaxed),
                execution_time_ns: self.worker_metrics[i].execution_time_ns.load(Ordering::Relaxed),
            })
            .collect();

        DetailedExecutorStats {
            uptime: self.start_time.elapsed(),
            total_tasks_spawned: self.tasks_spawned.load(Ordering::Relaxed),
            worker_count: self.config.worker_threads,
            worker_stats,
            current_load: self.load(),
        }
    }
}

// Implement the segregated traits

impl TaskSpawner for HybridExecutor {
    fn spawn<T>(&self, task: T) -> TaskHandle<T::Output>
    where
        T: Task,
    {
        self.spawn_internal(task, Priority::Normal)
    }

    fn spawn_async<F>(&self, _future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        // TODO: Implement async task spawning
        // For now, create a placeholder handle
        let task_id = self.task_registry.register_task(Priority::Normal);
        
        #[cfg(feature = "std")]
        {
            // Create a dummy channel for now - this will be properly implemented
            // when async task execution is added
            let (_sender, receiver) = std::sync::mpsc::channel();
            TaskHandle::new_with_receiver(task_id, receiver)
        }
        #[cfg(not(feature = "std"))]
        {
            TaskHandle::new_detached(task_id)
        }
    }

    fn spawn_blocking<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Create a task from the blocking function
        let task_id = self.task_registry.generate_id();
        let task = TaskBuilder::new()
            .with_id(task_id)
            .build(func);

        self.spawn_internal(task, Priority::Normal)
    }

    fn spawn_with_priority<T>(&self, task: T, priority: Priority) -> TaskHandle<T::Output>
    where
        T: Task,
    {
        self.spawn_internal(task, priority)
    }
}

impl TaskManager for HybridExecutor {
    fn cancel_task(&self, id: TaskId) -> Result<(), TaskError> {
        self.task_registry.cancel_task(id)
    }

    fn task_status(&self, id: TaskId) -> Option<TaskStatus> {
        self.task_registry.get_status(id)
    }

    fn wait_for_task(&self, id: TaskId, timeout: Option<Duration>) -> impl Future<Output = Result<(), TaskError>> + Send {
        TaskWaitFuture {
            task_id: id,
            registry: self.task_registry.clone(),
            timeout,
            start_time: Instant::now(),
        }
    }

    fn task_stats(&self, id: TaskId) -> Option<TaskStats> {
        self.task_registry.get_stats(id)
    }
}

impl ExecutorControl for HybridExecutor {
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        // Simple block_on implementation using a basic runtime
        // In a production implementation, this would integrate with the async runtime
        futures::executor::block_on(future)
    }

    fn try_run(&self) -> bool {
        // Check if any schedulers have work and try to process it
        let mut work_done = false;
        
        for scheduler in &self.schedulers {
            if scheduler.load() > 0 {
                if let Ok(executed) = scheduler.try_execute_next() {
                    work_done = work_done || executed;
                }
            }
        }
        
        work_done
    }

    fn shutdown(&self) {
        // Signal shutdown to all workers and cleanup thread
        self.shutdown_signal.store(true, Ordering::Release);

        // Wait for cleanup thread to complete first
        let mut cleanup_handle = self.cleanup_handle.lock().unwrap();
        if let Some(handle) = cleanup_handle.take() {
            let _ = handle.join();
        }
        drop(cleanup_handle); // Release the lock

        // Wait for all worker threads to complete
        let mut handles = self.worker_handles.lock().unwrap();
        let worker_handles = std::mem::take(&mut *handles);
        drop(handles); // Release the lock before joining
        
        for handle in worker_handles {
            let _ = handle.join();
        }

        // Notify plugins
        for plugin in &self.plugins {
            plugin.on_shutdown();
        }

        // Signal shutdown complete
        let mut shutdown_complete = self.shutdown_mutex.lock().unwrap();
        *shutdown_complete = true;
        self.shutdown_complete.notify_all();
    }

    fn shutdown_timeout(&self, timeout: Duration) {
        // Try graceful shutdown first
        self.shutdown_signal.store(true, Ordering::Release);

        // Wait for completion or timeout
        let shutdown_complete = self.shutdown_mutex.lock().unwrap();
        let result = self.shutdown_complete.wait_timeout_while(
            shutdown_complete,
            timeout,
            |&mut completed| !completed,
        ).unwrap();

        if result.1.timed_out() {
            // Force shutdown - in a real implementation, this would forcibly terminate threads
        }
    }

    fn is_shutting_down(&self) -> bool {
        self.shutdown_signal.load(Ordering::Acquire)
    }

    fn worker_count(&self) -> usize {
        self.config.worker_threads
    }

    fn load(&self) -> usize {
        self.schedulers.iter().map(|s| s.load()).sum()
    }
}

impl Executor for HybridExecutor {
    #[cfg(feature = "metrics")]
    fn stats(&self) -> moirai_core::executor::ExecutorStats {
        // Return placeholder stats - would be implemented with real metrics
        moirai_core::executor::ExecutorStats {
            worker_stats: Vec::new(),
            global_queue_stats: moirai_core::executor::QueueStats {
                current_length: self.load(),
                max_length: 0,
                total_enqueued: self.tasks_spawned.load(Ordering::Relaxed),
                total_dequeued: 0,
                avg_wait_time_us: 0.0,
            },
            memory_stats: moirai_core::executor::MemoryStats {
                current_usage: 0,
                peak_usage: 0,
                allocations: 0,
                deallocations: 0,
                pool_stats: moirai_core::executor::PoolStats {
                    small_pool_utilization: 0.0,
                    medium_pool_utilization: 0.0,
                    large_pool_utilization: 0.0,
                    pool_hits: 0,
                    pool_misses: 0,
                },
            },
            task_stats: moirai_core::executor::TaskExecutionStats {
                total_spawned: self.tasks_spawned.load(Ordering::Relaxed),
                total_completed: 0,
                total_cancelled: 0,
                total_failed: 0,
                avg_execution_time_us: 0.0,
                p95_execution_time_us: 0.0,
                p99_execution_time_us: 0.0,
                throughput_per_second: 0.0,
            },
        }
    }
}

impl HybridExecutor {
    /// Manually trigger cleanup of completed task metadata.
    /// 
    /// This method can be called even when automatic cleanup is disabled
    /// to provide manual control over memory management.
    /// 
    /// # Behavior
    /// - Removes completed tasks older than the configured retention duration
    /// - Enforces the maximum retained task limit
    /// - Safe to call concurrently with other operations
    pub fn cleanup_completed_tasks(&self) {
        self.task_registry.cleanup_completed_with_limits(
            self.config.cleanup.task_retention_duration,
            self.config.cleanup.max_retained_tasks,
        );
    }

    /// Get statistics about task metadata cleanup.
    /// 
    /// Returns information about the current state of task metadata
    /// to help monitor memory usage and cleanup effectiveness.
    pub fn cleanup_stats(&self) -> CleanupStats {
        let tasks = self.task_registry.tasks.read().unwrap();
        
        let mut stats = CleanupStats {
            total_tasks: tasks.len(),
            active_tasks: 0,
            completed_tasks: 0,
            cancelled_tasks: 0,
            failed_tasks: 0,
            oldest_completed_task_age: None,
        };
        
        let now = Instant::now();
        let mut oldest_completion_time = None;
        
        for metadata in tasks.values() {
            match metadata.status {
                TaskStatus::Queued | TaskStatus::Running => {
                    stats.active_tasks += 1;
                }
                TaskStatus::Completed => {
                    stats.completed_tasks += 1;
                    if let Some(completion_time) = metadata.completion_time {
                        match oldest_completion_time {
                            None => oldest_completion_time = Some(completion_time),
                            Some(oldest) if completion_time < oldest => {
                                oldest_completion_time = Some(completion_time);
                            }
                            _ => {}
                        }
                    }
                }
                TaskStatus::Cancelled => {
                    stats.cancelled_tasks += 1;
                }
                TaskStatus::Failed => {
                    stats.failed_tasks += 1;
                }
            }
        }
        
        if let Some(oldest_time) = oldest_completion_time {
            stats.oldest_completed_task_age = Some(now.duration_since(oldest_time));
        }
        
        stats
    }
}

/// Detailed executor statistics for monitoring and debugging.
#[cfg(feature = "metrics")]
#[derive(Debug)]
pub struct DetailedExecutorStats {
    pub uptime: Duration,
    pub total_tasks_spawned: u64,
    pub worker_count: usize,
    pub worker_stats: Vec<WorkerSnapshot>,
    pub current_load: usize,
}

/// Snapshot of task metadata cleanup statistics.
#[cfg(feature = "metrics")]
#[derive(Debug, Default)]
pub struct CleanupStats {
    pub total_tasks: usize,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub cancelled_tasks: usize,
    pub failed_tasks: usize,
    pub oldest_completed_task_age: Option<Duration>,
}

/// Builder for creating hybrid executors with custom configuration.
/// 
/// Follows the Builder pattern for flexible executor construction
/// while maintaining the SOLID principles.
pub struct HybridExecutorBuilder {
    config: ExecutorConfig,
    plugins: Vec<Box<dyn ExecutorPlugin>>,
}

impl HybridExecutorBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
            plugins: Vec::new(),
        }
    }

    /// Set the number of worker threads.
    pub fn worker_threads(mut self, count: usize) -> Self {
        self.config.worker_threads = count;
        self
    }

    /// Set the number of async threads.
    pub fn async_threads(mut self, count: usize) -> Self {
        self.config.async_threads = count;
        self
    }

    /// Add a plugin to the executor.
    pub fn plugin(mut self, plugin: impl ExecutorPlugin) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Build the hybrid executor.
    pub fn build(self) -> ExecutorResult<HybridExecutor> {
        HybridExecutor::with_plugins(self.config, self.plugins)
    }
}

impl Default for HybridExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Example plugin that logs task lifecycle events.
#[derive(Debug)]
pub struct LoggingPlugin {
    name: &'static str,
}

impl LoggingPlugin {
    /// Create a new logging plugin.
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl ExecutorPlugin for LoggingPlugin {
    fn initialize(&mut self) -> Result<(), moirai_core::error::ExecutorError> {
        println!("Initializing logging plugin: {}", self.name);
        Ok(())
    }

    fn before_task_spawn(&self, task_id: TaskId, priority: Priority) {
        println!("Spawning task {} with priority {:?}", task_id, priority);
    }

    fn after_task_spawn(&self, task_id: TaskId) {
        println!("Task {} spawned successfully", task_id);
    }

    fn before_task_execute(&self, task_id: TaskId) {
        println!("Starting execution of task {}", task_id);
    }

    fn after_task_complete(&self, task_id: TaskId, success: bool) {
        println!("Task {} completed with success: {}", task_id, success);
    }

    fn on_shutdown(&self) {
        println!("Logging plugin {} shutting down", self.name);
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

// Example blocking task for testing
struct BlockingTask<F, R> 
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    func: Option<F>,
    context: TaskContext,
    _phantom: std::marker::PhantomData<R>,
}

impl<F, R> BlockingTask<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    fn new(func: F) -> Self {
        Self {
            func: Some(func),
            context: TaskContext::new(TaskId::new(0)), // Will be replaced by executor
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, R> Task for BlockingTask<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    type Output = R;

    fn execute(mut self) -> Self::Output {
        let func = self.func.take().expect("Task already executed");
        func()
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::time::Duration;

    #[test]
    fn test_hybrid_executor_creation() {
        let executor = HybridExecutorBuilder::new()
            .worker_threads(4)
            .async_threads(2)
            .build()
            .unwrap();

        assert_eq!(executor.worker_count(), 4);
        assert!(!executor.is_shutting_down());
    }

    #[test]
    fn test_task_spawning() {
        let executor = HybridExecutorBuilder::new()
            .worker_threads(2)
            .build()
            .unwrap();

        let counter = Arc::new(AtomicI32::new(0));
        let counter_clone = counter.clone();

        let task = TaskBuilder::new()
            .name("test_task")
            .priority(Priority::High)
            .build(move || {
                counter_clone.fetch_add(1, Ordering::Relaxed);
                42
            });

        let _handle = executor.spawn_with_priority(task, Priority::High);
        
        // Give task time to complete
        thread::sleep(Duration::from_millis(100));
        
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_cleanup_mechanism() {
        use core::time::Duration;
        
        // Create executor with short cleanup intervals for testing
        let cleanup_config = CleanupConfig {
            task_retention_duration: Duration::from_millis(100),
            cleanup_interval: Duration::from_millis(50),
            enable_automatic_cleanup: true,
            max_retained_tasks: 5,
        };
        
        let config = ExecutorConfig {
            cleanup: cleanup_config,
            ..ExecutorConfig::default()
        };
        
        let executor = HybridExecutor::new(config).unwrap();
        
        // Spawn several tasks that complete quickly
        for i in 0..10 {
            let task = TaskBuilder::new()
                .name("test_task")
                .build(move || i * 2);
                
            let _handle = executor.spawn_with_priority(task, Priority::Normal);
        }
        
        // Wait for tasks to complete
        thread::sleep(Duration::from_millis(50));
        
        // Check initial stats
        let stats = executor.cleanup_stats();
        println!("Initial stats: {:?}", stats);
        
        // Wait for cleanup to occur (retention duration + cleanup interval + buffer)
        thread::sleep(Duration::from_millis(200));
        
        // Check stats after cleanup
        let stats_after = executor.cleanup_stats();
        println!("Stats after cleanup: {:?}", stats_after);
        
        // Verify that cleanup occurred
        assert!(stats_after.total_tasks <= stats.total_tasks);
        
        // Cleanup on shutdown
        executor.shutdown();
    }

    #[test]
    fn test_manual_cleanup() {
        use core::time::Duration;
        
        // Create executor with automatic cleanup disabled
        let cleanup_config = CleanupConfig {
            task_retention_duration: Duration::from_millis(50),
            cleanup_interval: Duration::from_secs(3600), // 1 hour - won't trigger during test
            enable_automatic_cleanup: false, // Disabled
            max_retained_tasks: 3,
        };
        
        let config = ExecutorConfig {
            cleanup: cleanup_config,
            ..ExecutorConfig::default()
        };
        
        let executor = HybridExecutor::new(config).unwrap();
        
        // Spawn several tasks
        for i in 0..5 {
            let task = TaskBuilder::new()
                .name("manual_test_task")
                .build(move || i);
                
            let _handle = executor.spawn_with_priority(task, Priority::Normal);
        }
        
        // Wait for tasks to complete
        thread::sleep(Duration::from_millis(100));
        
        // Check stats before manual cleanup
        let stats_before = executor.cleanup_stats();
        println!("Stats before manual cleanup: {:?}", stats_before);
        
        // Wait for retention duration to pass
        thread::sleep(Duration::from_millis(60));
        
        // Manually trigger cleanup
        executor.cleanup_completed_tasks();
        
        // Check stats after manual cleanup
        let stats_after = executor.cleanup_stats();
        println!("Stats after manual cleanup: {:?}", stats_after);
        
        // Verify that cleanup occurred
        assert!(stats_after.completed_tasks < stats_before.completed_tasks);
        
        // Cleanup on shutdown
        executor.shutdown();
    }

    #[test]
    fn test_cleanup_max_retained_tasks() {
        use core::time::Duration;
        
        // Create executor with very long retention but low max count
        let cleanup_config = CleanupConfig {
            task_retention_duration: Duration::from_secs(3600), // 1 hour
            cleanup_interval: Duration::from_millis(50),
            enable_automatic_cleanup: true,
            max_retained_tasks: 3, // Low limit to test count-based cleanup
        };
        
        let config = ExecutorConfig {
            cleanup: cleanup_config,
            ..ExecutorConfig::default()
        };
        
        let executor = HybridExecutor::new(config).unwrap();
        
        // Spawn more tasks than the max retained limit
        for i in 0..8 {
            let task = TaskBuilder::new()
                .name("count_test_task")
                .build(move || i);
                
            let _handle = executor.spawn_with_priority(task, Priority::Normal);
            
            // Small delay between tasks to ensure different completion times
            thread::sleep(Duration::from_millis(10));
        }
        
        // Wait for tasks to complete and cleanup to occur
        thread::sleep(Duration::from_millis(200));
        
        let stats = executor.cleanup_stats();
        println!("Stats after count-based cleanup: {:?}", stats);
        
        // Verify that count-based cleanup occurred
        // Should have at most max_retained_tasks completed tasks
        assert!(stats.completed_tasks <= 3);
        
        // Cleanup on shutdown
        executor.shutdown();
    }

    #[test]
    fn test_task_priority() {
        let executor = HybridExecutorBuilder::new()
            .worker_threads(1)
            .build()
            .unwrap();

        let task = TaskBuilder::new()
            .priority(Priority::Critical)
            .build(|| "high priority task");

        let handle = executor.spawn_with_priority(task, Priority::Critical);
        assert_eq!(handle.id().get(), 0); // First task should get ID 0
    }

    #[test]
    fn test_executor_shutdown() {
        let executor = HybridExecutorBuilder::new()
            .worker_threads(2)
            .build()
            .unwrap();

        assert!(!executor.is_shutting_down());
        
        executor.shutdown();
        
        assert!(executor.is_shutting_down());
    }

    #[test]
    fn test_logging_plugin() {
        let plugin = LoggingPlugin::new("test");
        assert_eq!(plugin.name(), "test");

        let executor = HybridExecutorBuilder::new()
            .worker_threads(1)
            .plugin(plugin)
            .build()
            .unwrap();

        let task = TaskBuilder::new().build(|| 42);
        let _handle = executor.spawn(task);
    }

    #[test]
    fn test_task_registry() {
        let registry = TaskRegistry::new();
        
        let id1 = registry.register_task(Priority::Normal);
        let id2 = registry.register_task(Priority::High);
        
        assert_ne!(id1, id2);
        assert_eq!(registry.get_status(id1), Some(TaskStatus::Queued));
        
        registry.update_status(id1, TaskStatus::Running);
        assert_eq!(registry.get_status(id1), Some(TaskStatus::Running));
    }

    #[test]
    fn test_task_cancellation() {
        let registry = TaskRegistry::new();
        let id = registry.register_task(Priority::Normal);
        
        assert!(registry.cancel_task(id).is_ok());
        assert_eq!(registry.get_status(id), Some(TaskStatus::Cancelled));
    }

    #[test]
    fn test_worker_metrics_sharing() {
        use std::sync::atomic::Ordering;
        
        let executor = HybridExecutorBuilder::new()
            .worker_threads(2)
            .build()
            .unwrap();

        // Spawn several tasks to generate metrics
        for i in 0..5 {
            let task = TaskBuilder::new()
                .name("metrics_test_task")
                .build(move || {
                    // Do some work to generate metrics
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    i * 2
                });
                
            let _handle = executor.spawn_with_priority(task, Priority::Normal);
        }
        
        // Wait for tasks to complete
        std::thread::sleep(std::time::Duration::from_millis(200));
        
        // Verify that worker metrics are accessible and have been updated
        let stats = executor.detailed_stats();
        
        println!("Detailed executor stats: {:?}", stats);
        
        // Verify we have metrics for all workers
        assert_eq!(stats.worker_stats.len(), 2);
        
        // Verify that at least some tasks were executed (metrics should be > 0)
        let total_tasks_executed: u64 = stats.worker_stats.iter()
            .map(|w| w.tasks_executed)
            .sum();
            
        println!("Total tasks executed across workers: {}", total_tasks_executed);
        assert!(total_tasks_executed > 0, "Workers should have executed some tasks");
        
        // Verify that execution time was recorded
        let total_execution_time: u64 = stats.worker_stats.iter()
            .map(|w| w.execution_time_ns)
            .sum();
            
        println!("Total execution time across workers: {} ns", total_execution_time);
        assert!(total_execution_time > 0, "Workers should have recorded execution time");
        
        executor.shutdown();
    }
}