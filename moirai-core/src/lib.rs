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
pub mod security;

#[cfg(feature = "metrics")]
pub mod metrics;

// Re-export key types from task module
pub use task::{
    Task, TaskBuilder, TaskExt,
    Closure, Chained, Mapped, Catch, Parameterized, Group, Spawner
};

/// A handle to a spawned task that allows monitoring and control.
#[derive(Debug, Clone)]
pub struct TaskHandle {
    /// The unique identifier for this task
    pub id: TaskId,
    /// Whether the task can be cancelled
    pub cancellable: bool,
}

impl TaskHandle {
    /// Create a new task handle.
    #[must_use]
    pub const fn new(id: TaskId, cancellable: bool) -> Self {
        Self { id, cancellable }
    }
}

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

/// Real-time scheduling policies for tasks that require deterministic execution.
/// 
/// # Design Principles Applied
/// - **SOLID**: Single responsibility for scheduling policy definition
/// - **CUPID**: Composable with existing Priority system
/// - **GRASP**: Information expert for real-time requirements
/// - **ADP**: Adapts to different real-time scheduling needs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtSchedulingPolicy {
    /// First-In-First-Out scheduling (non-preemptive within priority)
    Fifo,
    /// Round-Robin scheduling with time slicing
    RoundRobin {
        /// Time slice in microseconds
        time_slice_us: u32,
    },
    /// Deadline-driven scheduling (Earliest Deadline First)
    DeadlineDriven,
    /// Rate-monotonic scheduling (fixed priority based on period)
    RateMonotonic,
    /// Energy-efficient scheduling (reduces CPU frequency when possible)
    EnergyEfficient {
        /// Target CPU utilization percentage (0-100)
        target_utilization: u8,
    },
    /// Fair scheduling with proportional share
    ProportionalShare {
        /// Scheduling weight for this task
        weight: u32,
    },
}

impl Default for RtSchedulingPolicy {
    fn default() -> Self {
        Self::Fifo
    }
}

impl fmt::Display for RtSchedulingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fifo => write!(f, "FIFO"),
            Self::RoundRobin { time_slice_us } => write!(f, "RR({time_slice_us}μs)"),
            Self::DeadlineDriven => write!(f, "EDF"),
            Self::RateMonotonic => write!(f, "RM"),
            Self::EnergyEfficient { target_utilization } => write!(f, "EE({target_utilization}%)"),
            Self::ProportionalShare { weight } => write!(f, "PS({weight})"),
        }
    }
}

/// Real-time task constraints and timing requirements.
/// 
/// # Design Principles Applied
/// - **SOLID**: Single responsibility for timing constraints
/// - **CUPID**: Composable with task scheduling system
/// - **GRASP**: Information expert for real-time timing
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RtConstraints {
    /// Deadline in nanoseconds from task creation
    pub deadline_ns: Option<u64>,
    /// Period for periodic tasks in nanoseconds
    pub period_ns: Option<u64>,
    /// Worst-case execution time in nanoseconds
    pub wcet_ns: Option<u64>,
    /// CPU quota as percentage (0-100)
    pub cpu_quota_percent: Option<u8>,
    /// Priority ceiling for priority inheritance
    pub priority_ceiling: Option<Priority>,
    /// Execution time slice in nanoseconds
    pub time_slice_ns: Option<u64>,
}

impl RtConstraints {
    /// Create constraints for a deadline-driven task.
    #[must_use]
    pub const fn deadline(deadline_ns: u64) -> Self {
        Self {
            deadline_ns: Some(deadline_ns),
            period_ns: None,
            wcet_ns: None,
            cpu_quota_percent: None,
            priority_ceiling: None,
            time_slice_ns: None,
        }
    }

    /// Create constraints for a periodic task.
    #[must_use]
    pub const fn periodic(period_ns: u64, wcet_ns: u64) -> Self {
        Self {
            deadline_ns: Some(period_ns), // Deadline equals period for periodic tasks
            period_ns: Some(period_ns),
            wcet_ns: Some(wcet_ns),
            cpu_quota_percent: None,
            priority_ceiling: None,
            time_slice_ns: None,
        }
    }

    /// Create constraints for a round-robin task.
    #[must_use]
    pub const fn round_robin(time_slice_ns: u64) -> Self {
        Self {
            deadline_ns: None,
            period_ns: None,
            wcet_ns: None,
            cpu_quota_percent: None,
            priority_ceiling: None,
            time_slice_ns: Some(time_slice_ns),
        }
    }

    /// Create constraints for energy-efficient scheduling.
    #[must_use]
    pub const fn energy_efficient(target_utilization: u8) -> Self {
        Self {
            deadline_ns: None,
            period_ns: None,
            wcet_ns: None,
            cpu_quota_percent: Some(target_utilization),
            priority_ceiling: None,
            time_slice_ns: None,
        }
    }

    /// Create constraints for proportional-share scheduling.
    #[must_use]
    pub const fn proportional_share(weight: u32) -> Self {
        Self {
            deadline_ns: None,
            period_ns: None,
            wcet_ns: None,
            cpu_quota_percent: None,
            priority_ceiling: None,
            time_slice_ns: Some(weight as u64), // Use time_slice_ns to store weight
        }
    }

    /// Check if this constraint has a deadline.
    #[must_use]
    pub const fn has_deadline(&self) -> bool {
        self.deadline_ns.is_some()
    }

    /// Check if this is a periodic task.
    #[must_use]
    pub const fn is_periodic(&self) -> bool {
        self.period_ns.is_some()
    }

    /// Check if this constraint has priority inheritance.
    #[must_use]
    pub const fn has_priority_inheritance(&self) -> bool {
        self.priority_ceiling.is_some()
    }

    /// Check if this constraint has a CPU quota.
    #[must_use]
    pub const fn has_cpu_quota(&self) -> bool {
        self.cpu_quota_percent.is_some()
    }

    /// Get the utilization factor (WCET / Period) for schedulability analysis.
    #[must_use]
    pub fn utilization(&self) -> Option<f64> {
        match (self.wcet_ns, self.period_ns) {
            (Some(wcet), Some(period)) if period > 0 => {
                // Intentional precision loss for utilization calculation
                #[allow(clippy::cast_precision_loss)]
                {
                    Some(wcet as f64 / period as f64)
                }
            }
            _ => None,
        }
    }

    /// Create RT constraints with priority inheritance ceiling.
    #[must_use]
    pub const fn with_priority_ceiling(mut self, ceiling: Priority) -> Self {
        self.priority_ceiling = Some(ceiling);
        self
    }

    /// Create RT constraints with CPU quota (0-100 percent).
    #[must_use]
    pub const fn with_cpu_quota(mut self, quota_percent: u8) -> Self {
        let quota = if quota_percent > 100 { 100 } else { quota_percent };
        self.cpu_quota_percent = Some(quota);
        self
    }

    /// Create RT constraints with execution time slice.
    #[must_use]
    pub const fn with_time_slice(mut self, slice_ns: u64) -> Self {
        self.time_slice_ns = Some(slice_ns);
        self
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
    /// Real-time scheduling constraints
    pub rt_constraints: Option<RtConstraints>,
}

impl TaskContext {
    /// Create a new task context.
    #[must_use]
    pub const fn new(id: TaskId) -> Self {
        Self {
            id,
            priority: Priority::Normal,
            name: None,
            rt_constraints: None,
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

    /// Set real-time scheduling constraints for this task.
    #[must_use]
    pub const fn with_rt_constraints(mut self, constraints: RtConstraints) -> Self {
        self.rt_constraints = Some(constraints);
        self
    }

    /// Set a deadline for this task (convenience method).
    #[must_use]
    pub const fn with_deadline(mut self, deadline_ns: u64) -> Self {
        self.rt_constraints = Some(RtConstraints::deadline(deadline_ns));
        self
    }

    /// Make this task periodic with the given period and WCET.
    #[must_use]
    pub const fn with_period(mut self, period_ns: u64, wcet_ns: u64) -> Self {
        self.rt_constraints = Some(RtConstraints::periodic(period_ns, wcet_ns));
        self
    }

    /// Check if this task has real-time constraints.
    #[must_use]
    pub const fn is_realtime(&self) -> bool {
        self.rt_constraints.is_some()
    }

    /// Get the deadline for this task, if any.
    #[must_use]
    pub const fn deadline_ns(&self) -> Option<u64> {
        match &self.rt_constraints {
            Some(constraints) => constraints.deadline_ns,
            None => None,
        }
    }

    /// Check if this task is periodic.
    #[must_use]
    pub const fn is_periodic(&self) -> bool {
        match &self.rt_constraints {
            Some(constraints) => constraints.is_periodic(),
            None => false,
        }
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
impl<T> BoxedTask for T 
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
        assert_eq!(format!("{id}"), "Task(42)");
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
        assert!(!ctx.is_realtime());
    }

    #[test]
    fn test_rt_scheduling_policy() {
        let fifo = RtSchedulingPolicy::Fifo;
        let rr = RtSchedulingPolicy::RoundRobin { time_slice_us: 1000 };
        let edf = RtSchedulingPolicy::DeadlineDriven;
        let rm = RtSchedulingPolicy::RateMonotonic;

        assert_eq!(format!("{fifo}"), "FIFO");
        assert_eq!(format!("{rr}"), "RR(1000μs)");
        assert_eq!(format!("{edf}"), "EDF");
        assert_eq!(format!("{rm}"), "RM");

        assert_eq!(RtSchedulingPolicy::default(), RtSchedulingPolicy::Fifo);
    }

    #[test]
    fn test_rt_constraints() {
        // Test deadline constraint
        let deadline_constraint = RtConstraints::deadline(1_000_000); // 1ms
        assert_eq!(deadline_constraint.deadline_ns, Some(1_000_000));
        assert!(deadline_constraint.has_deadline());
        assert!(!deadline_constraint.is_periodic());

        // Test periodic constraint
        let periodic_constraint = RtConstraints::periodic(10_000_000, 2_000_000); // 10ms period, 2ms WCET
        assert_eq!(periodic_constraint.period_ns, Some(10_000_000));
        assert_eq!(periodic_constraint.wcet_ns, Some(2_000_000));
        assert_eq!(periodic_constraint.deadline_ns, Some(10_000_000)); // Deadline = period
        assert!(periodic_constraint.has_deadline());
        assert!(periodic_constraint.is_periodic());

        // Test utilization calculation
        let utilization = periodic_constraint.utilization().unwrap();
        assert!((utilization - 0.2).abs() < f64::EPSILON); // 2ms / 10ms = 0.2

        // Test round-robin constraint
        let rr_constraint = RtConstraints::round_robin(500_000); // 500 microseconds in nanoseconds
        assert_eq!(rr_constraint.time_slice_ns, Some(500_000));
        assert!(!rr_constraint.has_deadline());
        assert!(!rr_constraint.is_periodic());
    }

    #[test]
    fn test_task_context_with_rt_constraints() {
        let id = TaskId::new(1);
        
        // Test deadline task
        let deadline_task = TaskContext::new(id)
            .with_deadline(5_000_000); // 5ms deadline
        
        assert!(deadline_task.is_realtime());
        assert_eq!(deadline_task.deadline_ns(), Some(5_000_000));
        assert!(!deadline_task.is_periodic());

        // Test periodic task
        let periodic_task = TaskContext::new(id)
            .with_period(20_000_000, 3_000_000); // 20ms period, 3ms WCET
        
        assert!(periodic_task.is_realtime());
        assert_eq!(periodic_task.deadline_ns(), Some(20_000_000));
        assert!(periodic_task.is_periodic());

        // Test custom RT constraints
        let custom_constraint = RtConstraints {
            deadline_ns: Some(1_000_000),
            period_ns: None,
            wcet_ns: Some(500_000),
            cpu_quota_percent: None,
            priority_ceiling: None,
            time_slice_ns: None,
        };

        let custom_task = TaskContext::new(id)
            .with_rt_constraints(custom_constraint);
        
        assert!(custom_task.is_realtime());
        assert_eq!(custom_task.deadline_ns(), Some(1_000_000));
        assert!(!custom_task.is_periodic());
    }

    #[test]
    fn test_rt_constraints_utilization() {
        // Test valid utilization
        let constraint = RtConstraints::periodic(10_000_000, 3_000_000);
        assert_eq!(constraint.utilization(), Some(0.3));

        // Test invalid utilization (no period)
        let constraint = RtConstraints::deadline(1_000_000);
        assert_eq!(constraint.utilization(), None);

        // Test zero period
        let constraint = RtConstraints {
            wcet_ns: Some(1_000_000),
            period_ns: Some(0),
            deadline_ns: None,
            cpu_quota_percent: None,
            priority_ceiling: None,
            time_slice_ns: None,
        };
        assert_eq!(constraint.utilization(), None);
    }

    #[test]
    fn test_advanced_rt_scheduling_policies() {
        // Test energy-efficient scheduling
        let ee_constraint = RtConstraints::energy_efficient(75);
        assert_eq!(ee_constraint.cpu_quota_percent, Some(75));
        assert!(!ee_constraint.has_deadline());
        assert!(!ee_constraint.is_periodic());

        // Test proportional share scheduling
        let ps_constraint = RtConstraints::proportional_share(100);
        assert_eq!(ps_constraint.time_slice_ns, Some(100));

        // Test policy display
        assert_eq!(format!("{}", RtSchedulingPolicy::EnergyEfficient { target_utilization: 80 }), "EE(80%)");
        assert_eq!(format!("{}", RtSchedulingPolicy::ProportionalShare { weight: 50 }), "PS(50)");
    }

    #[test]
    fn test_priority_inheritance_and_cpu_quota() {
        let constraint = RtConstraints::deadline(5_000_000)
            .with_priority_ceiling(Priority::Critical)
            .with_cpu_quota(75)
            .with_time_slice(1000);

        assert!(constraint.has_priority_inheritance());
        assert!(constraint.has_cpu_quota());
        assert_eq!(constraint.priority_ceiling, Some(Priority::Critical));
        assert_eq!(constraint.cpu_quota_percent, Some(75));
        assert_eq!(constraint.time_slice_ns, Some(1000));

        // Test CPU quota clamping
        let constraint_clamped = RtConstraints::deadline(1_000_000).with_cpu_quota(150);
        assert_eq!(constraint_clamped.cpu_quota_percent, Some(100));
    }

    #[test]
    fn test_rt_constraints_default() {
        let default_constraints = RtConstraints::default();
        assert_eq!(default_constraints.deadline_ns, None);
        assert_eq!(default_constraints.period_ns, None);
        assert_eq!(default_constraints.wcet_ns, None);
        assert_eq!(default_constraints.cpu_quota_percent, None);
        assert_eq!(default_constraints.priority_ceiling, None);
        assert_eq!(default_constraints.time_slice_ns, None);
        assert_eq!(default_constraints.utilization(), None);
    }
}