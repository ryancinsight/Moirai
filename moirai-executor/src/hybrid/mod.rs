//! Main hybrid executor implementation.
//!
//! This module provides the core HybridExecutor that coordinates between
//! async and parallel execution, managing workers, tasks, and metrics.

use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    executor::ExecutorConfig,
    task::TaskId,
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

impl Drop for HybridExecutor {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}