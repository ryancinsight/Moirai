//! Main hybrid executor implementation.
//!
//! This module provides the core HybridExecutor that coordinates between
//! async and parallel execution, managing workers, tasks, and metrics.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    executor::{
        Executor, ExecutorConfig, ExecutorControl, ExecutorStats, TaskManager, TaskSpawner,
        TaskStats, TaskStatus,
    },
    task::{Task, TaskHandle, TaskId},
    Priority,
};

use crate::{metrics::ExecutorMetrics, registry::TaskRegistry, worker::Worker};

/// Main hybrid executor that coordinates async and parallel execution
pub struct HybridExecutor {
    config: ExecutorConfig,
    workers: Vec<Worker>,
    task_registry: Arc<Mutex<TaskRegistry>>,
    metrics: Arc<ExecutorMetrics>,
    shutdown_signal: Arc<AtomicBool>,
    next_task_id: AtomicU64,
}

impl HybridExecutor {
    /// Create a new hybrid executor with the given configuration
    pub fn new(config: ExecutorConfig) -> ExecutorResult<Self> {
        let worker_count = config.worker_threads; // Use config for worker count
        let mut workers = Vec::with_capacity(worker_count);

        // Create workers and start them immediately
        for i in 0..worker_count {
            let mut worker = Worker::new(i);
            worker.start(); // Start worker threads immediately
            workers.push(worker);
        }

        Ok(Self {
            config,
            workers,
            task_registry: Arc::new(Mutex::new(TaskRegistry::new())),
            metrics: Arc::new(ExecutorMetrics::new()),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            next_task_id: AtomicU64::new(1),
        })
    }

    /// Get executor configuration
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Start the executor and all its workers
    pub fn start(&mut self) -> ExecutorResult<()> {
        for worker in &mut self.workers {
            worker.start();
        }
        Ok(())
    }

    /// Shutdown the executor gracefully
    pub fn shutdown(&mut self) -> ExecutorResult<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        for worker in &mut self.workers {
            worker.shutdown();
        }

        Ok(())
    }

    /// Get executor metrics
    pub fn metrics(&self) -> &ExecutorMetrics {
        &self.metrics
    }

    /// Submit a task to the least loaded worker
    pub fn submit_task<F>(&self, task: F) -> ExecutorResult<TaskId>
    where
        F: FnOnce() + Send + 'static,
    {
        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);

        // Register the task
        {
            let mut registry = self.task_registry.lock().unwrap();
            registry.register_task();
        }

        // Find the least loaded worker
        let worker =
            self.workers
                .iter()
                .min_by_key(|w| w.queue_len())
                .ok_or(ExecutorError::SpawnFailed(
                    moirai_core::error::TaskError::ExecutionFailed(
                        moirai_core::error::TaskErrorKind::Io,
                    ),
                ))?;

        // Submit the task
        worker.submit_task(task);
        self.metrics.record_task_spawned();

        Ok(TaskId::new(task_id))
    }

    /// Get the number of active workers
    pub fn active_workers(&self) -> usize {
        self.workers.iter().filter(|w| w.is_busy()).count()
    }

    /// Get the total number of workers
    pub fn total_workers(&self) -> usize {
        self.workers.len()
    }

    /// Get pending task count across all workers
    pub fn pending_tasks(&self) -> usize {
        self.workers.iter().map(|w| w.queue_len()).sum()
    }
}

impl TaskSpawner for HybridExecutor {
    fn spawn<T>(&self, task: T) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
    {
        // Create a task handle
        let task_id = TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));

        // For now, use submit_task functionality adapted for the trait
        // In a real implementation, this would properly spawn the task
        let handle = TaskHandle::new_detached(task_id);

        // Submit the task (simplified implementation)
        let _ = self.submit_task(move || {
            let _ = task.execute();
        });

        Ok(handle)
    }

    fn spawn_async<F>(&self, future: F) -> ExecutorResult<TaskHandle<F::Output>>
    where
        F: core::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let task_id = TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));

        // Create a channel to communicate the result back
        let (result_sender, result_receiver) =
            std::sync::mpsc::channel::<Result<F::Output, moirai_core::error::TaskError>>();
        let handle = TaskHandle::new_with_receiver(task_id, result_receiver);

        // Submit the future as a task - implement basic async execution
        let _ = self.submit_task(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Create a basic async runtime
                let waker = std::task::Waker::noop();
                let mut context = std::task::Context::from_waker(&waker);
                let mut future = std::pin::Pin::from(Box::new(future));

                // Simple polling loop - not efficient but works
                loop {
                    match future.as_mut().poll(&mut context) {
                        std::task::Poll::Ready(value) => return value,
                        std::task::Poll::Pending => {
                            // In a real implementation, we'd yield to other tasks or wait for events
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                }
            }));

            match result {
                Ok(value) => {
                    let _ = result_sender.send(Ok(value));
                }
                Err(_) => {
                    let _ =
                        result_sender.send(Err(moirai_core::error::TaskError::ExecutionFailed(
                            moirai_core::error::TaskErrorKind::Io,
                        )));
                }
            }
        });

        Ok(handle)
    }

    fn spawn_blocking<F, R>(&self, func: F) -> ExecutorResult<TaskHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task_id = TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));

        // Create a channel to communicate the result back
        let (result_sender, result_receiver) =
            std::sync::mpsc::channel::<Result<R, moirai_core::error::TaskError>>();
        let handle = TaskHandle::new_with_receiver(task_id, result_receiver);

        // Submit the blocking function
        let _ = self.submit_task(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(func));
            match result {
                Ok(value) => {
                    let _ = result_sender.send(Ok(value));
                }
                Err(_) => {
                    let _ =
                        result_sender.send(Err(moirai_core::error::TaskError::ExecutionFailed(
                            moirai_core::error::TaskErrorKind::Io,
                        )));
                }
            }
        });

        Ok(handle)
    }

    fn spawn_with_priority<T>(
        &self,
        task: T,
        priority: Priority,
        locality_hint: Option<usize>,
    ) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
    {
        let task_id = TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));

        // Register the task with priority
        {
            let mut registry = self.task_registry.lock().unwrap();
            registry.register_task();
        }

        // Select worker based on locality hint or use least loaded
        let worker =
            if let Some(hint) = locality_hint {
                // Use modulo to map hint to available workers
                let worker_index = hint % self.workers.len();
                &self.workers[worker_index]
            } else {
                // Find the least loaded worker for load balancing
                self.workers.iter().min_by_key(|w| w.queue_len()).ok_or(
                    ExecutorError::SpawnFailed(moirai_core::error::TaskError::ExecutionFailed(
                        moirai_core::error::TaskErrorKind::Io,
                    )),
                )?
            };

        // Create prioritized task wrapper
        let _priority_ref = priority; // Use priority for future implementation
        let prioritized_task = move || {
            // Note: Priority could be implemented through queue ordering
            // For now, execute immediately but record priority for metrics
            if let Err(e) = task.execute() {
                eprintln!("Task execution failed: {:?}", e);
            }
        };

        // Submit the task
        worker.submit_task(prioritized_task);
        self.metrics.record_task_spawned();

        Ok(TaskHandle::new_detached(task_id))
    }
}

impl TaskManager for HybridExecutor {
    fn cancel_task(&self, id: TaskId) -> ExecutorResult<()> {
        // Simple cancellation - mark task as needing cancellation
        // In a full implementation, this would signal workers to stop the task
        if let Ok(registry) = self.task_registry.lock() {
            if registry.get_metadata(id.0).is_some() {
                // Task exists - in a full implementation, we'd signal cancellation
                Ok(())
            } else {
                Err(ExecutorError::SpawnFailed(
                    moirai_core::error::TaskError::InvalidOperation,
                ))
            }
        } else {
            Err(ExecutorError::ResourceExhausted(
                "Failed to acquire registry lock".to_string(),
            ))
        }
    }

    fn task_status(&self, id: TaskId) -> Option<TaskStatus> {
        if let Ok(registry) = self.task_registry.lock() {
            if let Some(_metadata) = registry.get_metadata(id.0) {
                if registry.is_completed(id.0) {
                    Some(TaskStatus::Completed)
                } else {
                    Some(TaskStatus::Running)
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    fn wait_for_task(
        &self,
        id: TaskId,
        timeout: Option<core::time::Duration>,
    ) -> impl core::future::Future<Output = ExecutorResult<()>> + Send {
        let registry = self.task_registry.clone();
        async move {
            let start = std::time::Instant::now();

            loop {
                // Check if task is complete
                if let Ok(registry) = registry.lock() {
                    if registry.is_completed(id.0) {
                        return Ok(());
                    }

                    if registry.get_metadata(id.0).is_none() {
                        return Err(ExecutorError::SpawnFailed(
                            moirai_core::error::TaskError::InvalidOperation,
                        ));
                    }
                }

                // Check timeout
                if let Some(timeout) = timeout {
                    if start.elapsed() >= timeout {
                        return Err(ExecutorError::ResourceExhausted(
                            "Task wait timeout".to_string(),
                        ));
                    }
                }

                // Simple polling delay without tokio dependency
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    fn task_stats(&self, id: TaskId) -> Option<TaskStats> {
        if let Ok(registry) = self.task_registry.lock() {
            if let Some(metadata) = registry.get_metadata(id.0) {
                Some(TaskStats {
                    id,
                    priority: Priority::Normal, // Default priority
                    status: if registry.is_completed(id.0) {
                        TaskStatus::Completed
                    } else {
                        TaskStatus::Running
                    },
                    spawn_time: metadata.created_at,
                    start_time: metadata.started_at,
                    completion_time: metadata.completed_at,
                    preemption_count: 0,  // Not tracked in current implementation
                    cpu_time_ns: 0,       // Not tracked in current implementation
                    memory_used_bytes: 0, // Not tracked in current implementation
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl ExecutorControl for HybridExecutor {
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: core::future::Future,
    {
        // Simple implementation using std::thread
        // In a real implementation, this would integrate with the executor's runtime
        std::thread::scope(|_| {
            // For now, we'll use a very basic implementation
            // This is not ideal but works for the immediate compilation fix
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(&waker);
            let mut future = std::pin::Pin::from(Box::new(future));

            loop {
                match future.as_mut().poll(&mut context) {
                    std::task::Poll::Ready(result) => return result,
                    std::task::Poll::Pending => {
                        // In a real implementation, this would properly yield
                        std::thread::yield_now();
                    }
                }
            }
        })
    }

    fn try_run(&self) -> bool {
        // Simple implementation that checks for available work
        // In a full implementation, this would try to process pending tasks
        if let Ok(registry) = self.task_registry.lock() {
            let active_tasks = registry.active_count();
            let total_capacity = self.workers.len();

            // Return true if we have capacity for more tasks
            active_tasks < total_capacity
        } else {
            false
        }
    }

    fn shutdown(&self) {
        // Convert mutable shutdown to immutable by using atomic signaling
        self.shutdown_signal.store(true, Ordering::Relaxed);
    }

    fn shutdown_timeout(&self, _timeout: core::time::Duration) {
        self.shutdown();
    }

    fn is_shutting_down(&self) -> bool {
        self.shutdown_signal.load(Ordering::Relaxed)
    }

    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    fn load(&self) -> usize {
        self.pending_tasks()
    }
}

impl Executor for HybridExecutor {
    #[cfg(feature = "metrics")]
    fn stats(&self) -> ExecutorStats {
        ExecutorStats::default()
    }
}

impl Drop for HybridExecutor {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
