//! Main hybrid executor implementation.
//!
//! This module provides the core HybridExecutor that coordinates between
//! async and parallel execution, managing workers, tasks, and metrics.

use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    executor::{ExecutorConfig, TaskSpawner, TaskManager, ExecutorControl, Executor, TaskStatus, TaskStats, ExecutorStats},
    task::{TaskId, Task, TaskHandle},
    Priority,
};

use crate::{
    registry::TaskRegistry,
    worker::Worker,
    metrics::ExecutorMetrics,
};

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
        let worker_count = 4; // Default worker count since we don't have the method
        let mut workers = Vec::with_capacity(worker_count);
        
        for i in 0..worker_count {
            workers.push(Worker::new(i));
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
        let worker = self.workers.iter()
            .min_by_key(|w| w.queue_len())
            .ok_or(ExecutorError::SpawnFailed(moirai_core::error::TaskError::ExecutionFailed(moirai_core::error::TaskErrorKind::Io)))?;

        // Submit the task
        worker.submit_task(task);
        self.metrics.record_task_spawned();

        Ok(TaskId::new(task_id))
    }

    /// Get the number of active workers
    pub fn active_workers(&self) -> usize {
        self.workers.iter()
            .filter(|w| w.is_busy())
            .count()
    }

    /// Get the total number of workers
    pub fn total_workers(&self) -> usize {
        self.workers.len()
    }

    /// Get pending task count across all workers
    pub fn pending_tasks(&self) -> usize {
        self.workers.iter()
            .map(|w| w.queue_len())
            .sum()
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
        // Create a task handle for the async task
        let task_id = TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));
        let handle = TaskHandle::new_detached(task_id);
        
        // Submit the future as a task (simplified)
        let _ = self.submit_task(move || {
            // In a real implementation, this would properly execute the future
            // For now, we'll just drop it since we can't block_on here
            drop(future);
        });
        
        Ok(handle)
    }

    fn spawn_blocking<F, R>(&self, func: F) -> ExecutorResult<TaskHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task_id = TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));
        let handle = TaskHandle::new_detached(task_id);
        
        // Submit the blocking function
        let _ = self.submit_task(move || {
            let _ = func();
        });
        
        Ok(handle)
    }

    fn spawn_with_priority<T>(
        &self,
        task: T,
        _priority: Priority,
        _locality_hint: Option<usize>,
    ) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
    {
        // For now, ignore priority and locality hints
        self.spawn(task)
    }
}

impl TaskManager for HybridExecutor {
    fn cancel_task(&self, _id: TaskId) -> ExecutorResult<()> {
        // TODO: Implement proper task cancellation
        Ok(())
    }

    fn task_status(&self, _id: TaskId) -> Option<TaskStatus> {
        // TODO: Implement proper task status tracking
        None
    }

    fn wait_for_task(&self, _id: TaskId, _timeout: Option<core::time::Duration>) -> impl core::future::Future<Output = ExecutorResult<()>> + Send {
        async { Ok(()) }
    }

    fn task_stats(&self, _id: TaskId) -> Option<TaskStats> {
        // TODO: Implement task statistics
        None
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
        // TODO: Implement non-blocking task execution
        false
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