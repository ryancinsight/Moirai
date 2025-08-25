//! Worker thread management and metrics.
//!
//! This module provides worker thread abstractions and performance monitoring
//! for the hybrid executor system.

use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

/// Worker thread state and configuration
pub struct Worker {
    pub id: usize,
    pub thread_handle: Option<thread::JoinHandle<()>>,
    pub task_queue: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send>>>>,
    pub is_shutdown: Arc<AtomicBool>,
    pub metrics: WorkerMetrics,
}

/// Performance metrics for individual workers
#[derive(Debug)]
pub struct WorkerMetrics {
    pub tasks_executed: AtomicUsize,
    pub total_execution_time: Duration,
    pub last_task_completion: Option<Instant>,
    pub idle_time: Duration,
    pub created_at: Instant,
}

impl Clone for WorkerMetrics {
    fn clone(&self) -> Self {
        Self {
            tasks_executed: AtomicUsize::new(self.tasks_executed.load(Ordering::Relaxed)),
            total_execution_time: self.total_execution_time,
            last_task_completion: self.last_task_completion,
            idle_time: self.idle_time,
            created_at: self.created_at,
        }
    }
}

impl Default for WorkerMetrics {
    fn default() -> Self {
        Self {
            tasks_executed: AtomicUsize::new(0),
            total_execution_time: Duration::ZERO,
            last_task_completion: None,
            idle_time: Duration::ZERO,
            created_at: Instant::now(),
        }
    }
}

impl Worker {
    /// Create a new worker with the given ID
    pub fn new(id: usize) -> Self {
        Self {
            id,
            thread_handle: None,
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            is_shutdown: Arc::new(AtomicBool::new(false)),
            metrics: WorkerMetrics::default(),
        }
    }

    /// Start the worker thread
    pub fn start(&mut self) {
        let id = self.id;
        let queue = self.task_queue.clone();
        let shutdown = self.is_shutdown.clone();

        let handle = thread::Builder::new()
            .name(format!("moirai-worker-{}", id))
            .spawn(move || {
                Self::worker_loop(id, queue, shutdown);
            })
            .expect("Failed to spawn worker thread");

        self.thread_handle = Some(handle);
    }

    /// Worker thread main loop
    fn worker_loop(
        _worker_id: usize,
        queue: Arc<Mutex<VecDeque<Box<dyn FnOnce() + Send>>>>,
        shutdown: Arc<AtomicBool>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            let task = {
                let mut q = queue.lock().unwrap();
                q.pop_front()
            };

            match task {
                Some(task) => {
                    let start = Instant::now();
                    task();
                    let _elapsed = start.elapsed();
                    // Metrics would be updated here in a real implementation
                }
                None => {
                    // No tasks available, briefly yield to avoid busy waiting
                    thread::yield_now();
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Submit a task to this worker
    pub fn submit_task<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let mut queue = self.task_queue.lock().unwrap();
        queue.push_back(Box::new(task));
    }

    /// Get the current queue length
    pub fn queue_len(&self) -> usize {
        self.task_queue.lock().unwrap().len()
    }

    /// Check if the worker is busy (has tasks in queue)
    pub fn is_busy(&self) -> bool {
        self.queue_len() > 0
    }

    /// Shutdown the worker
    pub fn shutdown(&mut self) {
        self.is_shutdown.store(true, Ordering::Relaxed);

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Get worker utilization percentage
    pub fn utilization(&self) -> f64 {
        let total_time = self.metrics.created_at.elapsed();
        if total_time.is_zero() {
            0.0
        } else {
            let active_time = total_time - self.metrics.idle_time;
            (active_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}
