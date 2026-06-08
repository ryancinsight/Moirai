use super::id_and_context::TaskContext;

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

#[cfg(feature = "std")]
impl<T> Task for Box<T>
where
    T: Task,
{
    type Output = T::Output;

    fn execute(self) -> Self::Output {
        (*self).execute()
    }

    fn context(&self) -> &TaskContext {
        (**self).context()
    }

    fn is_stealable(&self) -> bool {
        (**self).is_stealable()
    }

    fn estimated_cost(&self) -> u32 {
        (**self).estimated_cost()
    }
}
