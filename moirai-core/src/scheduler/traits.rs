//! Core `Scheduler` trait and `SchedulerId` identifier newtype.

use super::task::ScheduledTask;
use crate::error::SchedulerResult;
use core::fmt;

/// Core scheduling interface for task distribution and execution.
///
/// This trait defines the fundamental operations that all scheduler implementations
/// must support for managing task queues and work distribution.
pub trait Scheduler: Send + Sync + 'static {
    /// Schedule a task for execution.
    ///
    /// The scheduler will determine when and where to execute the task based on
    /// its internal policies and current system state.
    ///
    /// # Errors
    /// Returns `SchedulerError` if the task cannot be scheduled due to:
    /// - Resource constraints (queue full, memory limits)
    /// - Scheduler shutdown or invalid state
    /// - Task validation failures
    fn schedule(&self, task: ScheduledTask) -> SchedulerResult<()>;

    /// Schedule a concrete task without exposing a runtime task trait object.
    fn schedule_task<T>(&self, task: T) -> SchedulerResult<()>
    where
        T: crate::task::Task,
    {
        self.schedule(ScheduledTask::new(task))
    }

    /// Retrieves the next available task for execution.
    ///
    /// # Returns
    /// `Ok(Some(task))` if a task is available, `Ok(None)` if the queue is empty.
    ///
    /// # Errors
    /// Returns `SchedulerError` if there's an internal error accessing the queue
    /// or if the scheduler is in an invalid state.
    fn next_task(&self) -> SchedulerResult<Option<ScheduledTask>>;

    /// Attempts to steal a task from another scheduler (work-stealing).
    ///
    /// # Arguments
    /// * `victim` - The scheduler to attempt stealing from
    ///
    /// # Returns
    /// `Ok(Some(task))` if a task was successfully stolen, `Ok(None)` if no tasks available.
    ///
    /// # Errors
    /// Returns `SchedulerError` if the steal operation fails due to:
    /// - Lock contention or synchronization issues
    /// - Invalid victim scheduler state
    /// - Internal queue corruption
    fn try_steal<S>(&self, victim: &S) -> SchedulerResult<Option<ScheduledTask>>
    where
        S: Scheduler,
    {
        if victim.can_be_stolen_from() {
            victim.next_task()
        } else {
            Ok(None)
        }
    }

    /// Returns the current number of queued tasks.
    fn load(&self) -> usize;

    /// Returns a unique identifier for this scheduler instance.
    fn id(&self) -> SchedulerId;

    /// Returns whether this scheduler can have tasks stolen from it.
    ///
    /// # Returns
    /// `true` if the scheduler has stealable tasks, `false` otherwise.
    fn can_be_stolen_from(&self) -> bool {
        self.load() > 0
    }
}

/// A unique identifier for schedulers within the work-stealing system.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchedulerId(usize);

impl SchedulerId {
    /// Creates a new scheduler ID.
    ///
    /// # Arguments
    /// * `id` - The numeric identifier for this scheduler
    ///
    /// # Returns
    /// A new scheduler ID instance
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    ///
    /// # Returns
    /// The numeric identifier for this scheduler
    #[must_use]
    pub const fn get(&self) -> usize {
        self.0
    }
}

impl fmt::Display for SchedulerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scheduler({})", self.0)
    }
}
