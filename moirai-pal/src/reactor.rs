//! Core reactor implementation for managing async I/O operations.
//!
//! This module provides the central event loop and task scheduling for
//! platform-specific async I/O operations.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};
use std::io;

use crate::{Reactor, Event, RawFd, Interest, create_reactor};

/// Central async I/O reactor managing all platform-specific operations.
pub struct IoReactor {
    /// Platform-specific reactor implementation
    platform_reactor: Box<dyn Reactor>,
    /// Event loop control
    running: Arc<AtomicBool>,
    /// Registered file descriptor tracking
    registered_fds: Arc<Mutex<HashMap<RawFd, FdInfo>>>,
    /// Pending task queue
    task_queue: Arc<Mutex<VecDeque<BoxedTask>>>,
    /// Waker registry for efficient task resumption
    waker_registry: Arc<Mutex<HashMap<TaskId, Waker>>>,
    /// Performance metrics
    metrics: Arc<ReactorMetrics>,
}

/// Information about registered file descriptors
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for future telemetry/debugging per ADR requirements
struct FdInfo {
    interest: Interest,
    registered_at: Instant,
    event_count: u64,
}

/// Task identifier for tracking async operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// Boxed async task for the reactor queue
type BoxedTask = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Performance metrics for the reactor
#[derive(Debug, Default)]
pub struct ReactorMetrics {
    /// Total events processed
    pub events_processed: AtomicU64,
    /// Total tasks executed
    pub tasks_executed: AtomicU64,
    /// Average event processing time (nanoseconds)
    pub avg_event_time_ns: AtomicU64,
    /// Peak number of registered file descriptors
    pub peak_fd_count: AtomicU64,
    /// Reactor uptime
    pub start_time: std::sync::OnceLock<Instant>,
}

impl IoReactor {
    /// Create a new I/O reactor with platform-optimal implementation.
    pub fn new() -> io::Result<Self> {
        let platform_reactor = create_reactor()?;
        
        Ok(Self {
            platform_reactor,
            running: Arc::new(AtomicBool::new(false)),
            registered_fds: Arc::new(Mutex::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            waker_registry: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(ReactorMetrics::default()),
        })
    }
    
    /// Register a file descriptor for async I/O operations.
    pub fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        // Register with platform reactor
        self.platform_reactor.register_fd(fd, interest)?;
        
        // Track registration
        let mut fds = self.registered_fds.lock().unwrap();
        fds.insert(fd, FdInfo {
            interest,
            registered_at: Instant::now(),
            event_count: 0,
        });
        
        // Update peak FD count metric
        let current_count = fds.len() as u64;
        let peak = self.metrics.peak_fd_count.load(Ordering::Relaxed);
        if current_count > peak {
            self.metrics.peak_fd_count.store(current_count, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// Unregister a file descriptor.
    pub fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        self.platform_reactor.unregister_fd(fd)?;
        self.registered_fds.lock().unwrap().remove(&fd);
        Ok(())
    }
    
    /// Spawn an async task on the reactor.
    pub fn spawn<F>(&self, future: F) -> TaskHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task_id = TaskId::new();
        
        // Box the future for storage
        let boxed_task: BoxedTask = Box::pin(future);
        
        // Add to task queue
        self.task_queue.lock().unwrap().push_back(boxed_task);
        
        TaskHandle {
            task_id,
            waker_registry: self.waker_registry.clone(),
        }
    }
    
    /// Run the event loop until stopped.
    pub fn run(&self) -> io::Result<()> {
        self.running.store(true, Ordering::SeqCst);
        self.metrics.start_time.set(Instant::now()).map_err(|_| {
            io::Error::other("Reactor already started")
        })?;
        
        while self.running.load(Ordering::SeqCst) {
            self.run_iteration(Some(Duration::from_millis(10)))?;
        }
        
        Ok(())
    }
    
    /// Run a single iteration of the event loop.
    pub fn run_iteration(&self, timeout: Option<Duration>) -> io::Result<()> {
        let iteration_start = Instant::now();
        
        // Process pending tasks first
        self.process_pending_tasks();
        
        // Poll for I/O events
        let events = self.platform_reactor.poll_events(timeout)?;
        
        // Process I/O events
        for event in events {
            self.handle_event(event)?;
        }
        
        // Update metrics
        let iteration_time = iteration_start.elapsed().as_nanos() as u64;
        self.metrics.avg_event_time_ns.store(iteration_time, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Stop the event loop.
    pub fn stop(&self) -> io::Result<()> {
        self.running.store(false, Ordering::SeqCst);
        self.platform_reactor.wake()
    }
    
    /// Process all pending tasks in the queue.
    fn process_pending_tasks(&self) {
        let mut tasks = self.task_queue.lock().unwrap();
        let mut completed_tasks = Vec::new();
        
        for (index, task) in tasks.iter_mut().enumerate() {
            // Create a simple noop waker compatible with MSRV 1.75.0
            // Using standard library patterns per Rust Book Ch.16
            use std::task::{RawWaker, RawWakerVTable, Waker};
            
            const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
                |_| RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE),
                |_| {},
                |_| {},
                |_| {},
            );
            
            let waker = unsafe {
                Waker::from_raw(RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE))
            };
            let mut context = Context::from_waker(&waker);
            
            match task.as_mut().poll(&mut context) {
                Poll::Ready(()) => {
                    completed_tasks.push(index);
                    self.metrics.tasks_executed.fetch_add(1, Ordering::Relaxed);
                }
                Poll::Pending => {
                    // Task is still pending, keep it in queue
                }
            }
        }
        
        // Remove completed tasks (in reverse order to maintain indices)
        for &index in completed_tasks.iter().rev() {
            tasks.remove(index);
        }
    }
    
    /// Handle a single I/O event.
    fn handle_event(&self, event: Event) -> io::Result<()> {
        // Update FD event count
        if let Ok(mut fds) = self.registered_fds.lock() {
            if let Some(fd_info) = fds.get_mut(&event.fd) {
                fd_info.event_count += 1;
            }
        }
        
        // Update metrics
        self.metrics.events_processed.fetch_add(1, Ordering::Relaxed);
        
        // Wake any tasks waiting on this file descriptor
        self.wake_fd_waiters(event.fd);
        
        Ok(())
    }
    
    /// Wake tasks waiting on a specific file descriptor.
    fn wake_fd_waiters(&self, _fd: RawFd) {
        // This would wake specific tasks waiting on the FD
        // For now, this is a simplified implementation
        // In a complete implementation, we'd track FD->Task mappings
    }
    
    /// Get current reactor metrics.
    pub fn metrics(&self) -> ReactorMetrics {
        ReactorMetrics {
            events_processed: AtomicU64::new(self.metrics.events_processed.load(Ordering::Relaxed)),
            tasks_executed: AtomicU64::new(self.metrics.tasks_executed.load(Ordering::Relaxed)),
            avg_event_time_ns: AtomicU64::new(self.metrics.avg_event_time_ns.load(Ordering::Relaxed)),
            peak_fd_count: AtomicU64::new(self.metrics.peak_fd_count.load(Ordering::Relaxed)),
            start_time: std::sync::OnceLock::new(),
        }
    }
}

/// Handle for tracking spawned tasks.
pub struct TaskHandle {
    task_id: TaskId,
    waker_registry: Arc<Mutex<HashMap<TaskId, Waker>>>,
}

impl TaskHandle {
    /// Get the task ID.
    pub fn id(&self) -> TaskId {
        self.task_id
    }
}

impl Future for TaskHandle {
    type Output = ();
    
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Register waker for this task
        self.waker_registry
            .lock()
            .unwrap()
            .insert(self.task_id, cx.waker().clone());
            
        // For now, always return pending
        // In a complete implementation, this would check if the task completed
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reactor_creation() {
        let reactor = IoReactor::new();
        assert!(reactor.is_ok());
    }
    
    #[test]
    fn test_task_id_generation() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }
    
    #[test]
    fn test_reactor_metrics() {
        let reactor = IoReactor::new().unwrap();
        let metrics = reactor.metrics();
        assert_eq!(metrics.events_processed.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.tasks_executed.load(Ordering::Relaxed), 0);
    }
}