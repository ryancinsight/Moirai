//! Hybrid concurrency primitives combining coroutines and work-stealing.
//!
//! This module provides advanced concurrency types that combine the benefits
//! of different execution models for optimal performance.

use crate::{Task, TaskId, TaskContext};
use crate::scheduler::ZeroCopyWorkStealingDeque;
use crate::error::ExecutorResult;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::future::Future;
use std::collections::VecDeque;

/// A hybrid task that can switch between coroutine and work-stealing execution
pub struct HybridTask<T> {
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
    /// Work-stealing deques for CPU-bound tasks
    work_queues: Vec<Arc<ZeroCopyWorkStealingDeque<Box<dyn Task<Output = ()>>>>>,
    /// Coroutine queue for I/O-bound tasks
    coroutine_queue: Arc<Mutex<VecDeque<Box<dyn Task<Output = ()>>>>>,
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
    
    /// Spawn a task with automatic mode selection
    pub fn spawn<T>(&self, task: T) -> ExecutorResult<()>
    where
        T: Task<Output = ()>,
    {
        self.active_tasks.fetch_add(1, Ordering::Relaxed);
        
        // Heuristic: Use coroutine mode for tasks with async nature
        // Use work-stealing for CPU-bound tasks
        let boxed_task = Box::new(task) as Box<dyn Task<Output = ()>>;
        
        // For now, distribute round-robin
        let worker_id = self.active_tasks.load(Ordering::Relaxed) % self.num_workers;
        self.work_queues[worker_id].push(boxed_task);
        
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
pub struct AdaptiveTask {
    id: TaskId,
    /// Execution history for adaptation
    history: ExecutionHistory,
    /// Current execution mode
    mode: ExecutionMode,
    /// The actual task
    inner: Box<dyn Task<Output = ()>>,
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum ExecutionMode {
    /// Prefer coroutine execution
    Coroutine,
    /// Prefer work-stealing execution
    WorkStealing,
    /// Automatically choose based on history
    Adaptive,
}

impl AdaptiveTask {
    /// Create a new adaptive task
    pub fn new(id: TaskId, task: Box<dyn Task<Output = ()>>) -> Self {
        Self {
            id,
            history: ExecutionHistory::default(),
            mode: ExecutionMode::Adaptive,
            inner: task,
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
        struct DummyTask {
            context: TaskContext,
        }
        impl Task for DummyTask {
            type Output = ();
            fn execute(self) -> Self::Output {}
            fn context(&self) -> &TaskContext {
                &self.context
            }
        }
        let dummy_task = Box::new(DummyTask {
            context: TaskContext::new(TaskId::new(1)),
        }) as Box<dyn Task<Output = ()>>;
        let mut adaptive = AdaptiveTask::new(TaskId::new(1), dummy_task);
        
        // Initially should choose coroutine
        assert_eq!(adaptive.choose_mode(), ExecutionMode::Coroutine);
        
        // Update history to favor work-stealing
        adaptive.update_history(ExecutionMode::Coroutine, 1000);
        adaptive.update_history(ExecutionMode::WorkStealing, 100);
        
        assert_eq!(adaptive.choose_mode(), ExecutionMode::WorkStealing);
    }
}