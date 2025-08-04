//! Hybrid concurrency primitives combining coroutines and work-stealing.
//!
//! This module provides advanced concurrency types that combine the benefits
//! of different execution models for optimal performance.

use crate::TaskId;
use crate::scheduler::{ZeroCopyWorkStealingDeque, TaskSlot};
use crate::error::ExecutorResult;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::future::Future;
use std::collections::VecDeque;

/// A hybrid task that can switch between coroutine and work-stealing execution
pub struct HybridTask<T> {
    #[allow(dead_code)]
    id: TaskId,
    state: TaskState,
    future: Option<Pin<Box<dyn Future<Output = T> + Send>>>,
    work: Option<Box<dyn FnOnce() -> T + Send>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TaskState {
    /// Task is ready to run
    Ready,
    /// Task is running as a coroutine
    Coroutine,
    /// Task is running in work-stealing mode
    WorkStealing,
    /// Task has completed
    Completed,
}

impl<T: Send + 'static> HybridTask<T> {
    /// Create a new hybrid task from a future
    pub fn from_future<F>(id: TaskId, future: F) -> Self
    where
        F: Future<Output = T> + Send + 'static,
    {
        Self {
            id,
            state: TaskState::Ready,
            future: Some(Box::pin(future)),
            work: None,
        }
    }
    
    /// Create a new hybrid task from a closure
    pub fn from_work<F>(id: TaskId, work: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
    {
        Self {
            id,
            state: TaskState::Ready,
            future: None,
            work: Some(Box::new(work)),
        }
    }
    
    /// Execute the task in the appropriate mode
    pub fn execute(mut self, waker: &Waker) -> Option<T> {
        match self.state {
            TaskState::Ready => {
                if let Some(future) = &mut self.future {
                    // Try to run as coroutine first
                    self.state = TaskState::Coroutine;
                    let mut cx = Context::from_waker(waker);
                    match future.as_mut().poll(&mut cx) {
                        Poll::Ready(result) => {
                            self.state = TaskState::Completed;
                            Some(result)
                        }
                        Poll::Pending => None,
                    }
                } else if let Some(work) = self.work.take() {
                    // Run as work-stealing task
                    self.state = TaskState::WorkStealing;
                    let result = work();
                    self.state = TaskState::Completed;
                    Some(result)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Hybrid executor that combines coroutine and work-stealing execution
pub struct HybridExecutor {
    /// Work-stealing deques for CPU-bound tasks using zero-allocation slots
    work_queues: Vec<Arc<ZeroCopyWorkStealingDeque<TaskSlot>>>,
    /// Coroutine queue for I/O-bound tasks
    #[allow(dead_code)]
    coroutine_queue: Arc<Mutex<VecDeque<TaskSlot>>>,
    /// Number of worker threads
    num_workers: usize,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Active task count
    active_tasks: Arc<AtomicUsize>,
}

impl HybridExecutor {
    /// Create a new hybrid executor
    pub fn new(num_workers: usize) -> Self {
        let mut work_queues = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            work_queues.push(Arc::new(ZeroCopyWorkStealingDeque::new(1024)));
        }
        
        Self {
            work_queues,
            coroutine_queue: Arc::new(Mutex::new(VecDeque::new())),
            num_workers,
            shutdown: Arc::new(AtomicBool::new(false)),
            active_tasks: Arc::new(AtomicUsize::new(0)),
        }
    }
    
    /// Spawn a closure task with automatic mode selection
    pub fn spawn_closure<F>(&self, f: F) -> ExecutorResult<()>
    where
        F: FnOnce() + Send + 'static,
    {
        self.active_tasks.fetch_add(1, Ordering::Relaxed);
        
        // Create a zero-allocation task slot
        let task_slot = TaskSlot::new_closure(f);
        
        // Distribute round-robin for load balancing
        let worker_id = self.active_tasks.load(Ordering::Relaxed) % self.num_workers;
        self.work_queues[worker_id].push(task_slot);
        
        Ok(())
    }
    
    /// Shutdown the executor
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
    
    /// Check if the executor is shutting down
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

/// Adaptive task that can switch execution modes based on runtime behavior
#[allow(dead_code)]
pub struct AdaptiveTask<F> {
    id: TaskId,
    /// Execution history for adaptation
    history: ExecutionHistory,
    /// Current execution mode
    mode: ExecutionMode,
    /// The actual task closure
    inner: Option<F>,
}

#[derive(Debug, Default)]
struct ExecutionHistory {
    /// Number of times run as coroutine
    coroutine_runs: u32,
    /// Number of times run as work-stealing
    work_stealing_runs: u32,
    /// Average execution time in coroutine mode (microseconds)
    avg_coroutine_time: u64,
    /// Average execution time in work-stealing mode (microseconds)
    avg_work_stealing_time: u64,
}

/// Execution mode for adaptive tasks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionMode {
    /// Prefer coroutine execution
    Coroutine,
    /// Prefer work-stealing execution
    WorkStealing,
    /// Automatically choose based on history
    Adaptive,
}

impl<F: FnOnce() + Send> AdaptiveTask<F> {
    /// Create a new adaptive task
    pub fn new(id: TaskId, task: F) -> Self {
        Self {
            id,
            history: ExecutionHistory::default(),
            mode: ExecutionMode::Adaptive,
            inner: Some(task),
        }
    }
    
    /// Update execution history
    pub fn update_history(&mut self, mode: ExecutionMode, duration_us: u64) {
        match mode {
            ExecutionMode::Coroutine => {
                self.history.coroutine_runs += 1;
                self.history.avg_coroutine_time = 
                    (self.history.avg_coroutine_time * (self.history.coroutine_runs - 1) as u64 
                     + duration_us) / self.history.coroutine_runs as u64;
            }
            ExecutionMode::WorkStealing => {
                self.history.work_stealing_runs += 1;
                self.history.avg_work_stealing_time = 
                    (self.history.avg_work_stealing_time * (self.history.work_stealing_runs - 1) as u64 
                     + duration_us) / self.history.work_stealing_runs as u64;
            }
            _ => {}
        }
    }
    
    /// Choose the best execution mode based on history
    pub fn choose_mode(&self) -> ExecutionMode {
        if self.history.coroutine_runs == 0 && self.history.work_stealing_runs == 0 {
            // No history, default to coroutine for I/O tasks
            ExecutionMode::Coroutine
        } else if self.history.avg_coroutine_time < self.history.avg_work_stealing_time {
            ExecutionMode::Coroutine
        } else {
            ExecutionMode::WorkStealing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hybrid_task_creation() {
        let task = HybridTask::from_work(TaskId::new(1), || 42);
        assert_eq!(task.state, TaskState::Ready);
    }
    
    #[test]
    fn test_adaptive_task_mode_selection() {
        let dummy_task = || {};
        let mut adaptive = AdaptiveTask::new(TaskId::new(1), dummy_task);
        
        // Initially should choose coroutine
        assert_eq!(adaptive.choose_mode(), ExecutionMode::Coroutine);
        
        // Update history to favor work-stealing
        adaptive.update_history(ExecutionMode::Coroutine, 1000);
        adaptive.update_history(ExecutionMode::WorkStealing, 100);
        
        assert_eq!(adaptive.choose_mode(), ExecutionMode::WorkStealing);
    }
}