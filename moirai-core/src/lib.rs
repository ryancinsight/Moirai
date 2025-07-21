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

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use core::{
    fmt,
    pin::Pin,
    task::{Context, Poll},
};

// Re-export commonly used types from alloc for convenience
pub use alloc::{boxed::Box, string::String, vec::Vec};

pub mod task;
pub mod executor;
pub mod scheduler;
pub mod error;

#[cfg(feature = "metrics")]
pub mod metrics;

// Re-export key types from task module
pub use task::{
    Task, TaskExt, TaskFuture, TaskHandle, TaskBuilder, TaskWrapper,
    ClosureTask, ChainedTask, MappedTask, CatchTask
};

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
#[derive(Debug, Clone)]
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

/// A trait for tasks that can be executed from a Box<dyn ...>
pub trait BoxedTask: Send + 'static {
    /// Execute this task and return a boxed result.
    fn execute_boxed(self: Box<Self>);

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

// Implement BoxedTask for any TaskWrapper (which always returns ())
impl<T> BoxedTask for TaskWrapper<T> 
where 
    T: Task + Send + 'static,
{
    fn execute_boxed(self: Box<Self>) {
        (*self).execute();
    }

    fn context(&self) -> &TaskContext {
        Task::context(self)
    }

    fn is_stealable(&self) -> bool {
        Task::is_stealable(self)
    }

    fn estimated_cost(&self) -> u32 {
        Task::estimated_cost(self)
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
}