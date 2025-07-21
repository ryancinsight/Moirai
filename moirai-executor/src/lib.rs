//! Hybrid executor implementation for Moirai concurrency library.

use moirai_core::{
    Task, BoxedTask, TaskHandle, Priority, TaskId, TaskContext,
    executor::{
        Executor, ExecutorConfig, TaskSpawner, TaskManager, ExecutorControl, 
        TaskStatus, TaskStats,
    },
    scheduler::{Scheduler, SchedulerId, SchedulerConfig, QueueType},
    error::{ExecutorResult, TaskError, ExecutorError},
};

use moirai_scheduler::{WorkStealingScheduler, LocalScheduler};

#[cfg(feature = "metrics")]
use moirai_core::executor::ExecutorStats;

use std::{
    future::Future,
    sync::{
        Arc, Mutex, RwLock, 
        atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Condvar,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
    collections::HashMap,
    pin::Pin,
    task::{Context, Poll, Waker},
};

/// A unique identifier for worker threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(usize);

impl WorkerId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn get(self) -> usize {
        self.0
    }
}

/// Messages sent to worker threads.
enum WorkerMessage {
    /// Execute a task
    Task(Box<dyn BoxedTask>),
    /// Shutdown the worker
    Shutdown,
}

/// Information about a spawned task.
#[derive(Debug)]
struct TaskInfo {
    id: TaskId,
    status: TaskStatus,
    priority: Priority,
    spawn_time: Instant,
    start_time: Option<Instant>,
    completion_time: Option<Instant>,
    waker: Option<Waker>,
}

/// A worker thread that executes tasks.
struct Worker {
    id: WorkerId,
    scheduler: Arc<WorkStealingScheduler>,
    receiver: Receiver<WorkerMessage>,
    shutdown: Arc<AtomicBool>,
    task_registry: Arc<RwLock<HashMap<TaskId, TaskInfo>>>,
}

impl Worker {
    fn new(
        id: WorkerId,
        scheduler: Arc<WorkStealingScheduler>,
        receiver: Receiver<WorkerMessage>,
        shutdown: Arc<AtomicBool>,
        task_registry: Arc<RwLock<HashMap<TaskId, TaskInfo>>>,
    ) -> Self {
        Self {
            id,
            scheduler,
            receiver,
            shutdown,
            task_registry,
        }
    }

    fn run(self) {
        let thread_name = format!("moirai-worker-{}", self.id.get());
        let _ = thread::Builder::new()
            .name(thread_name)
            .spawn(move || self.worker_loop());
    }

    fn worker_loop(self) {
        while !self.shutdown.load(Ordering::Relaxed) {
            // Try to receive a message from the channel
            match self.receiver.try_recv() {
                Ok(WorkerMessage::Task(task)) => {
                    self.execute_task(task);
                }
                Ok(WorkerMessage::Shutdown) => {
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // No messages, try to execute from local queue or steal work
                    if !self.scheduler.try_execute_next().unwrap_or(false) {
                        // Try to steal work from other workers
                        // For now, just yield to avoid busy waiting
                        thread::yield_now();
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    break;
                }
            }
        }

        // Execute remaining tasks in the local queue
        let _ = self.scheduler.run_until_empty();
    }

    fn execute_task(&self, task: Box<dyn BoxedTask>) {
        let task_id = TaskId::new(0); // TODO: Get actual task ID from context
        
        // Update task status to running
        if let Ok(mut registry) = self.task_registry.write() {
            if let Some(info) = registry.get_mut(&task_id) {
                info.status = TaskStatus::Running;
                info.start_time = Some(Instant::now());
            }
        }

        // Execute the task
        task.execute_boxed();

        // Update task status to completed
        if let Ok(mut registry) = self.task_registry.write() {
            if let Some(info) = registry.get_mut(&task_id) {
                info.status = TaskStatus::Completed;
                info.completion_time = Some(Instant::now());
                
                // Wake up any waiting futures
                if let Some(waker) = info.waker.take() {
                    waker.wake();
                }
            }
        }
    }
}

/// A hybrid executor that supports both async and parallel task execution.
pub struct HybridExecutor {
    config: ExecutorConfig,
    worker_handles: Vec<JoinHandle<()>>,
    worker_senders: Vec<Sender<WorkerMessage>>,
    schedulers: Vec<Arc<WorkStealingScheduler>>,
    shutdown: Arc<AtomicBool>,
    task_counter: AtomicU64,
    task_registry: Arc<RwLock<HashMap<TaskId, TaskInfo>>>,
    
    // For async support
    #[cfg(feature = "async")]
    async_runtime: Option<Arc<tokio::runtime::Runtime>>,
}

impl HybridExecutor {
    /// Create a new hybrid executor with the given configuration.
    pub fn new(config: ExecutorConfig) -> ExecutorResult<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let task_registry = Arc::new(RwLock::new(HashMap::new()));
        
        // Create schedulers for each worker thread
        let mut schedulers = Vec::new();
        let mut worker_handles = Vec::new();
        let mut worker_senders = Vec::new();

        for i in 0..config.worker_threads {
            let scheduler_config = SchedulerConfig {
                max_queue_size: config.max_local_queue_size,
                priority_scheduling: true,
                work_stealing: moirai_core::scheduler::WorkStealingStrategy::default(),
                queue_type: QueueType::ChaseLev,
            };

            let scheduler = Arc::new(WorkStealingScheduler::new(
                SchedulerId::new(i),
                scheduler_config,
            ));

            let (sender, receiver) = mpsc::channel();

            // Spawn the worker thread
            let worker_handle = thread::Builder::new()
                .name(format!("{}-{}", config.thread_name_prefix, i))
                .spawn({
                    let worker = Worker::new(
                        WorkerId::new(i),
                        scheduler.clone(),
                        receiver,
                        shutdown.clone(),
                        task_registry.clone(),
                    );
                    move || worker.worker_loop()
                })
                .map_err(|_| ExecutorError::ThreadPoolCreationFailed)?;

            schedulers.push(scheduler);
            worker_handles.push(worker_handle);
            worker_senders.push(sender);
        }

        // Create async runtime if needed
        #[cfg(feature = "async")]
        let async_runtime = if config.async_threads > 0 {
            Some(Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(config.async_threads)
                    .thread_name(&config.thread_name_prefix)
                    .enable_all()
                    .build()
                    .map_err(|_| ExecutorError::ThreadPoolCreationFailed)?
            ))
        } else {
            None
        };
        
        #[cfg(not(feature = "async"))]
        let async_runtime = None;

        Ok(Self {
            config,
            worker_handles,
            worker_senders,
            schedulers,
            shutdown,
            task_counter: AtomicU64::new(0),
            task_registry,
            async_runtime,
        })
    }

    /// Get the next task ID.
    fn next_task_id(&self) -> TaskId {
        TaskId::new(self.task_counter.fetch_add(1, Ordering::Relaxed))
    }

    /// Register a new task in the registry.
    fn register_task(&self, id: TaskId, priority: Priority) -> TaskHandle<()> {
        let info = TaskInfo {
            id,
            status: TaskStatus::Queued,
            priority,
            spawn_time: Instant::now(),
            start_time: None,
            completion_time: None,
            waker: None,
        };

        if let Ok(mut registry) = self.task_registry.write() {
            registry.insert(id, info);
        }

        TaskHandle::new(id)
    }

    /// Select the best scheduler for a new task based on load balancing.
    fn select_scheduler(&self) -> Option<&Arc<WorkStealingScheduler>> {
        // Simple round-robin selection for now
        // In a more sophisticated implementation, we'd consider load, NUMA topology, etc.
        if self.schedulers.is_empty() {
            return None;
        }

        let min_load_scheduler = self.schedulers
            .iter()
            .min_by_key(|s| s.load())
            .unwrap();

        Some(min_load_scheduler)
    }

    /// Submit a task to a worker via channel.
    fn submit_task(&self, task: Box<dyn BoxedTask>) -> ExecutorResult<()> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(ExecutorError::ShuttingDown);
        }

        // Find the worker with the least load
        let min_load_idx = self.worker_senders
            .iter()
            .enumerate()
            .min_by_key(|(i, _)| self.schedulers[*i].load())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.worker_senders[min_load_idx]
            .send(WorkerMessage::Task(task))
            .map_err(|_| ExecutorError::SpawnFailed(TaskError::ExecutionFailed(
                moirai_core::error::TaskErrorKind::Other
            )))?;

        Ok(())
    }
}

impl TaskSpawner for HybridExecutor {
    fn spawn<T>(&self, task: T) -> TaskHandle<T::Output>
    where
        T: Task,
    {
        // For now, we only support tasks that return ()
        // In a full implementation, we'd need a more sophisticated approach
        // to handle different return types
        let task_id = self.next_task_id();
        
        // Convert the task to a BoxedTask
        let boxed_task: Box<dyn BoxedTask> = Box::new(TaskWrapper::new(task));
        
        // Submit to a worker
        if let Err(_) = self.submit_task(boxed_task) {
            // If submission fails, return a handle that will immediately resolve to an error
            return TaskHandle::new(task_id);
        }

        self.register_task(task_id, Priority::Normal);
        TaskHandle::new(task_id)
    }

    fn spawn_async<F>(&self, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let task_id = self.next_task_id();

        #[cfg(feature = "async")]
        if let Some(runtime) = &self.async_runtime {
            // Spawn on the async runtime
            runtime.spawn(future);
        }

        TaskHandle::new(task_id)
    }

    fn spawn_blocking<F, R>(&self, func: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task_id = self.next_task_id();
        
        // Convert the function to a task
        let task = moirai_core::TaskBuilder::new(func, task_id).build();
        
        // Convert to BoxedTask
        let boxed_task: Box<dyn BoxedTask> = Box::new(TaskWrapper::new(task));
        
        // Submit to a worker
        if let Err(_) = self.submit_task(boxed_task) {
            return TaskHandle::new(task_id);
        }

        self.register_task(task_id, Priority::Normal);
        TaskHandle::new(task_id)
    }

    fn spawn_with_priority<T>(&self, task: T, priority: Priority) -> TaskHandle<T::Output>
    where
        T: Task,
    {
        let task_id = self.next_task_id();
        
        // Convert the task to a BoxedTask
        let boxed_task: Box<dyn BoxedTask> = Box::new(TaskWrapper::new(task));
        
        // Submit to a worker
        if let Err(_) = self.submit_task(boxed_task) {
            return TaskHandle::new(task_id);
        }

        self.register_task(task_id, priority);
        TaskHandle::new(task_id)
    }
}

impl TaskManager for HybridExecutor {
    fn cancel_task(&self, id: TaskId) -> Result<(), TaskError> {
        if let Ok(mut registry) = self.task_registry.write() {
            if let Some(info) = registry.get_mut(&id) {
                if matches!(info.status, TaskStatus::Queued | TaskStatus::Running) {
                    info.status = TaskStatus::Cancelled;
                    info.completion_time = Some(Instant::now());
                    
                    // Wake up any waiting futures
                    if let Some(waker) = info.waker.take() {
                        waker.wake();
                    }
                    
                    return Ok(());
                }
            }
        }
        
        Err(TaskError::InvalidOperation)
    }

    fn task_status(&self, id: TaskId) -> Option<TaskStatus> {
        self.task_registry
            .read()
            .ok()?
            .get(&id)
            .map(|info| info.status)
    }

    async fn wait_for_task(&self, id: TaskId, timeout: Option<Duration>) -> Result<(), TaskError> {
        let start_time = Instant::now();
        
        loop {
            // Check if task is completed
            if let Some(status) = self.task_status(id) {
                match status {
                    TaskStatus::Completed => return Ok(()),
                    TaskStatus::Cancelled => return Err(TaskError::Cancelled),
                    TaskStatus::Failed => return Err(TaskError::ExecutionFailed(
                        moirai_core::error::TaskErrorKind::Other
                    )),
                    _ => {
                        // Check timeout
                        if let Some(timeout) = timeout {
                            if start_time.elapsed() >= timeout {
                                return Err(TaskError::Timeout);
                            }
                        }
                        
                                                 // Yield and continue waiting
                         #[cfg(feature = "async")]
                         tokio::task::yield_now().await;
                         #[cfg(not(feature = "async"))]
                         std::future::ready(()).await;
                    }
                }
            } else {
                return Err(TaskError::InvalidOperation);
            }
        }
    }

    fn task_stats(&self, id: TaskId) -> Option<TaskStats> {
        let registry = self.task_registry.read().ok()?;
        let info = registry.get(&id)?;
        
        Some(TaskStats {
            id: info.id,
            status: info.status,
            priority: info.priority,
            spawn_time: info.spawn_time,
            start_time: info.start_time,
            completion_time: info.completion_time,
            preemption_count: 0, // TODO: Track preemptions
            cpu_time_ns: 0, // TODO: Track CPU time
            memory_used_bytes: 0, // TODO: Track memory usage
        })
    }
}

impl ExecutorControl for HybridExecutor {
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        #[cfg(feature = "async")]
        if let Some(runtime) = &self.async_runtime {
            runtime.block_on(future)
        } else {
            // Fallback to a simple block_on implementation
            futures::executor::block_on(future)
        }
        
        #[cfg(not(feature = "async"))]
        futures::executor::block_on(future)
    }

    fn try_run(&self) -> bool {
        // Try to execute one task from each scheduler
        let mut executed_any = false;
        
        for scheduler in &self.schedulers {
            if scheduler.try_execute_next().unwrap_or(false) {
                executed_any = true;
            }
        }
        
        executed_any
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Send shutdown messages to all workers
        for sender in &self.worker_senders {
            let _ = sender.send(WorkerMessage::Shutdown);
        }
        
        // Note: Async runtime will be dropped and shut down automatically
    }

    fn shutdown_timeout(&self, timeout: Duration) {
        self.shutdown();
        
        // Wait for workers to finish with timeout
        let start = Instant::now();
        for handle in &self.worker_handles {
            if start.elapsed() >= timeout {
                break;
            }
            // Note: We can't actually wait on the handles here because we don't own them
            // In a real implementation, we'd need to restructure this
        }
    }

    fn is_shutting_down(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
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
    fn stats(&self) -> ExecutorStats {
        let worker_stats = self.schedulers
            .iter()
            .enumerate()
            .map(|(i, scheduler)| {
                let stats = scheduler.stats();
                moirai_core::executor::WorkerStats {
                    thread_id: i,
                    tasks_executed: stats.tasks_completed,
                    successful_steals: stats.steal_context.successes as u64,
                    failed_steals: (stats.steal_context.attempts - stats.steal_context.successes) as u64,
                    stolen_from: 0, // TODO: Track this
                    local_queue_length: stats.queue_length,
                    cpu_utilization: stats.cpu_utilization,
                    total_execution_time_ns: 0, // TODO: Track this
                }
            })
            .collect();

        ExecutorStats {
            worker_stats,
            global_queue_stats: moirai_core::executor::QueueStats {
                current_length: self.load(),
                max_length: 0, // TODO: Track this
                total_enqueued: 0, // TODO: Track this
                total_dequeued: 0, // TODO: Track this
                avg_wait_time_us: 0.0, // TODO: Track this
            },
            memory_stats: moirai_core::executor::MemoryStats {
                current_usage: 0, // TODO: Track this
                peak_usage: 0, // TODO: Track this
                allocations: 0, // TODO: Track this
                deallocations: 0, // TODO: Track this
                pool_stats: moirai_core::executor::PoolStats {
                    small_pool_utilization: 0.0,
                    medium_pool_utilization: 0.0,
                    large_pool_utilization: 0.0,
                    pool_hits: 0,
                    pool_misses: 0,
                },
            },
            task_stats: moirai_core::executor::TaskExecutionStats {
                total_spawned: self.task_counter.load(Ordering::Relaxed),
                total_completed: 0, // TODO: Track this
                total_cancelled: 0, // TODO: Track this
                total_failed: 0, // TODO: Track this
                avg_execution_time_us: 0.0, // TODO: Track this
                p95_execution_time_us: 0.0, // TODO: Track this
                p99_execution_time_us: 0.0, // TODO: Track this
                throughput_per_second: 0.0, // TODO: Track this
            },
        }
    }
}

/// Wrapper to convert any Task to a BoxedTask.
struct TaskWrapper<T> {
    task: Option<T>,
}

impl<T> TaskWrapper<T> {
    fn new(task: T) -> Self {
        Self { task: Some(task) }
    }
}

impl<T> BoxedTask for TaskWrapper<T>
where
    T: Task + Send + 'static,
    T::Output: Send + 'static,
{
    fn execute_boxed(mut self: Box<Self>) {
        if let Some(task) = self.task.take() {
            let _ = task.execute(); // Ignore the output for now
        }
    }

    fn context(&self) -> &TaskContext {
        if let Some(task) = &self.task {
            task.context()
        } else {
            // Return a default context if task was already executed
            static DEFAULT_CONTEXT: std::sync::OnceLock<TaskContext> = std::sync::OnceLock::new();
            DEFAULT_CONTEXT.get_or_init(|| TaskContext::new(TaskId::new(0)))
        }
    }

    fn is_stealable(&self) -> bool {
        self.task.as_ref().map(|t| t.is_stealable()).unwrap_or(false)
    }

    fn estimated_cost(&self) -> u32 {
        self.task.as_ref().map(|t| t.estimated_cost()).unwrap_or(1)
    }
}

/// A handle to the executor for managing its lifecycle.
pub struct ExecutorHandle {
    join_handles: Vec<JoinHandle<()>>,
}

impl ExecutorHandle {
    /// Create a new executor handle.
    pub fn new(join_handles: Vec<JoinHandle<()>>) -> Self {
        Self { join_handles }
    }

    /// Wait for the executor to shut down.
    pub fn join(self) {
        for handle in self.join_handles {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moirai_core::{TaskBuilder, Priority};
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_executor_creation() {
        let config = ExecutorConfig::default();
        let executor = HybridExecutor::new(config).unwrap();
        assert!(executor.worker_count() > 0);
    }

    #[test]
    fn test_task_spawning() {
        let config = ExecutorConfig {
            worker_threads: 2,
            async_threads: 1,
            ..ExecutorConfig::default()
        };
        let executor = HybridExecutor::new(config).unwrap();

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let task = TaskBuilder::new(
            move || {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            },
            TaskId::new(1)
        ).build();

        let handle = executor.spawn(task);
        assert_eq!(handle.id().get(), 0); // First task gets ID 0

        // Give the task some time to execute
        std::thread::sleep(Duration::from_millis(100));
        
        // Note: In a real test, we'd wait for the task to complete properly
        // For now, we just verify the handle was created
    }

    #[test]
    fn test_task_priorities() {
        let config = ExecutorConfig::default();
        let executor = HybridExecutor::new(config).unwrap();

        let task = TaskBuilder::new(|| {}, TaskId::new(1)).build();
        let handle = executor.spawn_with_priority(task, Priority::High);
        
        // Verify the task was registered with high priority
        if let Some(stats) = executor.task_stats(handle.id()) {
            assert_eq!(stats.priority, Priority::High);
        }
    }

    #[test]
    fn test_executor_shutdown() {
        let config = ExecutorConfig::default();
        let executor = HybridExecutor::new(config).unwrap();
        
        assert!(!executor.is_shutting_down());
        executor.shutdown();
        assert!(executor.is_shutting_down());
    }

    #[test]
    fn test_load_tracking() {
        let config = ExecutorConfig::default();
        let executor = HybridExecutor::new(config).unwrap();
        
        // Initially, load should be 0
        assert_eq!(executor.load(), 0);
        
        // After spawning tasks, load might increase
        // (though it might be executed immediately in tests)
        let task = TaskBuilder::new(|| {
            std::thread::sleep(Duration::from_millis(10));
        }, TaskId::new(1)).build();
        
        let _handle = executor.spawn(task);
        // Note: Load tracking depends on timing and might be 0 if task executes immediately
    }
}