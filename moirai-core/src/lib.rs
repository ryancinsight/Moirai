//! # Moirai Core
//!
//! Core abstractions and traits for the Moirai concurrency library.
//! 
//! Named after the Greek Fates who controlled the threads of life, Moirai provides
//! a unified concurrency framework that weaves together async and parallel execution.
//!
//! ## Design Principles
//!
//! - **Zero-cost abstractions**: All abstractions compile away to optimal code
//! - **Memory safety**: Leverage Rust's ownership system for safe concurrency
//! - **Composability**: Small, focused components that work together
//! - **Performance**: Designed for maximum throughput and minimal latency

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "std")]
extern crate std;

use core::{
    fmt,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

pub mod task;
pub mod executor;
pub mod scheduler;
pub mod error;

#[cfg(feature = "metrics")]
pub mod metrics;

/// A unique identifier for tasks within the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    /// Create a new task ID.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Task({})", self.0)
    }
}

/// Priority levels for task scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Low priority tasks (background work)
    Low = 0,
    /// Normal priority tasks (default)
    Normal = 1,
    /// High priority tasks (interactive work)
    High = 2,
    /// Critical priority tasks (system-level work)
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Task execution context and metadata.
#[derive(Debug)]
pub struct TaskContext {
    /// Unique identifier for this task
    pub id: TaskId,
    /// Priority level for scheduling
    pub priority: Priority,
    /// Optional name for debugging
    pub name: Option<&'static str>,
}

impl TaskContext {
    /// Create a new task context.
    #[must_use]
    pub const fn new(id: TaskId) -> Self {
        Self {
            id,
            priority: Priority::Normal,
            name: None,
        }
    }

    /// Set the priority for this task.
    #[must_use]
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the name for this task.
    #[must_use]
    pub const fn with_name(mut self, name: &'static str) -> Self {
        self.name = Some(name);
        self
    }
}

/// The core trait for executable tasks in the Moirai runtime.
pub trait Task: Send + 'static {
    /// The output type produced by this task.
    type Output: Send + 'static;

    /// Execute this task to completion.
    fn execute(self) -> Self::Output;

    /// Get the task context for scheduling and debugging.
    fn context(&self) -> &TaskContext;

    /// Check if this task can be stolen by another thread.
    fn is_stealable(&self) -> bool {
        true
    }

    /// Estimate the computational cost of this task (for load balancing).
    fn estimated_cost(&self) -> u32 {
        1
    }
}

/// A task that can be polled to completion (async task).
pub trait AsyncTask: Send + 'static {
    /// The output type produced by this task.
    type Output: Send + 'static;

    /// Poll this task for completion.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;

    /// Get the task context for scheduling and debugging.
    fn context(&self) -> &TaskContext;

    /// Check if this task is ready to make progress.
    fn is_ready(&self) -> bool {
        false
    }
}

/// Marker trait for tasks that perform blocking operations.
pub trait BlockingTask: Task {}

/// Marker trait for tasks that are CPU-intensive.
pub trait CpuTask: Task {}

/// Marker trait for tasks that perform I/O operations.
pub trait IoTask: AsyncTask {}

/// A handle that can be used to await the completion of a spawned task.
pub struct TaskHandle<T> {
    id: TaskId,
    _phantom: core::marker::PhantomData<T>,
}

impl<T> TaskHandle<T> {
    /// Create a new task handle.
    #[must_use]
    pub const fn new(id: TaskId) -> Self {
        Self {
            id,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Get the task ID.
    #[must_use]
    pub const fn id(&self) -> TaskId {
        self.id
    }
}

impl<T> Future for TaskHandle<T>
where
    T: Send + 'static,
{
    type Output = Result<T, crate::error::TaskError>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // This is a placeholder - actual implementation would check task completion
        Poll::Pending
    }
}

unsafe impl<T: Send> Send for TaskHandle<T> {}
unsafe impl<T: Send> Sync for TaskHandle<T> {}

/// Configuration for task execution behavior.
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Maximum time a task can run before being preempted (microseconds)
    pub max_execution_time: Option<u64>,
    /// Whether to enable task metrics collection
    pub enable_metrics: bool,
    /// Stack size for blocking tasks (bytes)
    pub stack_size: Option<usize>,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Some(10_000), // 10ms default
            enable_metrics: cfg!(feature = "metrics"),
            stack_size: None, // Use system default
        }
    }
}

/// Builder pattern for creating tasks with configuration.
pub struct TaskBuilder<F> {
    func: F,
    context: TaskContext,
    config: TaskConfig,
}

impl<F> TaskBuilder<F> {
    /// Create a new task builder.
    #[must_use]
    pub fn new(func: F, id: TaskId) -> Self {
        Self {
            func,
            context: TaskContext::new(id),
            config: TaskConfig::default(),
        }
    }

    /// Set the task priority.
    #[must_use]
    pub fn priority(mut self, priority: Priority) -> Self {
        self.context.priority = priority;
        self
    }

    /// Set the task name.
    #[must_use]
    pub fn name(mut self, name: &'static str) -> Self {
        self.context.name = Some(name);
        self
    }

    /// Set the task configuration.
    #[must_use]
    pub fn config(mut self, config: TaskConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the task.
    #[must_use]
    pub fn build(self) -> impl Task<Output = F::Output>
    where
        F: FnOnce() -> F::Output + Send + 'static,
        F::Output: Send + 'static,
    {
        ClosureTask {
            func: Some(self.func),
            context: self.context,
            config: self.config,
        }
    }
}

/// A task implementation that wraps a closure.
struct ClosureTask<F> {
    func: Option<F>,
    context: TaskContext,
    config: TaskConfig,
}

impl<F> Task for ClosureTask<F>
where
    F: FnOnce() -> F::Output + Send + 'static,
    F::Output: Send + 'static,
{
    type Output = F::Output;

    fn execute(mut self) -> Self::Output {
        let func = self.func.take().expect("Task already executed");
        func()
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn estimated_cost(&self) -> u32 {
        // Default cost estimation - could be made configurable
        match self.context.priority {
            Priority::Low => 1,
            Priority::Normal => 2,
            Priority::High => 4,
            Priority::Critical => 8,
        }
    }
}

impl<F> fmt::Debug for ClosureTask<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClosureTask")
            .field("context", &self.context)
            .field("config", &self.config)
            .field("executed", &self.func.is_none())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_id() {
        let id = TaskId::new(42);
        assert_eq!(id.get(), 42);
        assert_eq!(format!("{}", id), "Task(42)");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_task_context() {
        let id = TaskId::new(1);
        let ctx = TaskContext::new(id)
            .with_priority(Priority::High)
            .with_name("test_task");

        assert_eq!(ctx.id, id);
        assert_eq!(ctx.priority, Priority::High);
        assert_eq!(ctx.name, Some("test_task"));
    }

    #[test]
    fn test_task_builder() {
        let id = TaskId::new(1);
        let task = TaskBuilder::new(|| 42, id)
            .priority(Priority::High)
            .name("test")
            .build();

        assert_eq!(task.context().id, id);
        assert_eq!(task.context().priority, Priority::High);
        assert_eq!(task.context().name, Some("test"));
        assert_eq!(task.execute(), 42);
    }
}