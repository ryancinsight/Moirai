//! Task abstractions and utilities.

use crate::{TaskId, TaskContext, Priority, Task, AsyncTask};
use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    fmt,
};

/// A trait for objects that can spawn tasks.
pub trait TaskSpawner {
    /// Spawn a task for execution.
    fn spawn<T>(&self, task: T) -> crate::TaskHandle<T::Output>
    where
        T: Task;

    /// Spawn an async task for execution.
    fn spawn_async<F>(&self, future: F) -> crate::TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static;

    /// Spawn a blocking task that may block the current thread.
    fn spawn_blocking<F, R>(&self, func: F) -> crate::TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static;
}

/// A trait for objects that can schedule tasks.
pub trait TaskScheduler {
    /// Schedule a task for execution.
    fn schedule<T>(&self, task: T) -> crate::error::SchedulerResult<()>
    where
        T: Task;

    /// Try to steal work from another scheduler.
    fn try_steal(&self) -> crate::error::SchedulerResult<Option<Box<dyn Task<Output = ()>>>>;

    /// Get the current load of this scheduler.
    fn load(&self) -> usize;

    /// Check if the scheduler is idle.
    fn is_idle(&self) -> bool {
        self.load() == 0
    }
}

/// A task that yields control periodically to allow other tasks to run.
pub struct YieldingTask<T> {
    inner: T,
    yield_count: usize,
    yield_interval: usize,
    context: TaskContext,
}

impl<T> YieldingTask<T>
where
    T: Task,
{
    /// Create a new yielding task that yields every `yield_interval` iterations.
    pub fn new(task: T, yield_interval: usize) -> Self {
        let context = TaskContext::new(TaskId::new(0)); // Will be assigned by scheduler
        Self {
            inner: task,
            yield_count: 0,
            yield_interval,
            context,
        }
    }

    /// Set the task context.
    pub fn with_context(mut self, context: TaskContext) -> Self {
        self.context = context;
        self
    }
}

impl<T> Task for YieldingTask<T>
where
    T: Task,
{
    type Output = T::Output;

    fn execute(self) -> Self::Output {
        // For now, just execute the inner task
        // In a real implementation, this would yield periodically
        self.inner.execute()
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_stealable(&self) -> bool {
        self.inner.is_stealable()
    }

    fn estimated_cost(&self) -> u32 {
        self.inner.estimated_cost()
    }
}

/// A task that can be cancelled.
pub struct CancellableTask<T> {
    inner: Option<T>,
    cancelled: bool,
    context: TaskContext,
}

impl<T> CancellableTask<T>
where
    T: Task,
{
    /// Create a new cancellable task.
    pub fn new(task: T) -> Self {
        let context = TaskContext::new(TaskId::new(0)); // Will be assigned by scheduler
        Self {
            inner: Some(task),
            cancelled: false,
            context,
        }
    }

    /// Cancel this task.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.inner = None;
    }

    /// Check if this task is cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Set the task context.
    pub fn with_context(mut self, context: TaskContext) -> Self {
        self.context = context;
        self
    }
}

impl<T> Task for CancellableTask<T>
where
    T: Task,
{
    type Output = Result<T::Output, crate::error::TaskError>;

    fn execute(mut self) -> Self::Output {
        if self.cancelled {
            return Err(crate::error::TaskError::Cancelled);
        }

        match self.inner.take() {
            Some(task) => Ok(task.execute()),
            None => Err(crate::error::TaskError::Cancelled),
        }
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_stealable(&self) -> bool {
        !self.cancelled && self.inner.as_ref().map_or(false, |t| t.is_stealable())
    }

    fn estimated_cost(&self) -> u32 {
        if self.cancelled {
            0
        } else {
            self.inner.as_ref().map_or(0, |t| t.estimated_cost())
        }
    }
}

/// A task that times out after a specified duration.
pub struct TimedTask<T> {
    inner: T,
    timeout_micros: u64,
    context: TaskContext,
}

impl<T> TimedTask<T>
where
    T: Task,
{
    /// Create a new timed task with the specified timeout in microseconds.
    pub fn new(task: T, timeout_micros: u64) -> Self {
        let context = TaskContext::new(TaskId::new(0)); // Will be assigned by scheduler
        Self {
            inner: task,
            timeout_micros,
            context,
        }
    }

    /// Set the task context.
    pub fn with_context(mut self, context: TaskContext) -> Self {
        self.context = context;
        self
    }
}

impl<T> Task for TimedTask<T>
where
    T: Task,
{
    type Output = Result<T::Output, crate::error::TaskError>;

    fn execute(self) -> Self::Output {
        // For now, just execute the inner task
        // In a real implementation, this would enforce timeouts
        Ok(self.inner.execute())
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_stealable(&self) -> bool {
        self.inner.is_stealable()
    }

    fn estimated_cost(&self) -> u32 {
        self.inner.estimated_cost()
    }
}

/// A future that wraps a task for async execution.
pub struct TaskFuture<T> {
    task: Option<T>,
    context: TaskContext,
}

impl<T> TaskFuture<T>
where
    T: Task,
{
    /// Create a new task future.
    pub fn new(task: T) -> Self {
        let context = TaskContext::new(TaskId::new(0)); // Will be assigned by scheduler
        Self {
            task: Some(task),
            context,
        }
    }

    /// Set the task context.
    pub fn with_context(mut self, context: TaskContext) -> Self {
        self.context = context;
        self
    }
}

impl<T> Future for TaskFuture<T>
where
    T: Task,
{
    type Output = T::Output;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.task.take() {
            Some(task) => Poll::Ready(task.execute()),
            None => panic!("TaskFuture polled after completion"),
        }
    }
}

impl<T> AsyncTask for TaskFuture<T>
where
    T: Task,
{
    type Output = T::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Future::poll(self, cx)
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_ready(&self) -> bool {
        self.task.is_some()
    }
}

impl<T> fmt::Debug for TaskFuture<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskFuture")
            .field("task", &self.task)
            .field("context", &self.context)
            .finish()
    }
}

/// Extension trait for tasks to add common functionality.
pub trait TaskExt: Task + Sized {
    /// Make this task cancellable.
    fn cancellable(self) -> CancellableTask<Self> {
        CancellableTask::new(self)
    }

    /// Add a timeout to this task.
    fn timeout(self, timeout_micros: u64) -> TimedTask<Self> {
        TimedTask::new(self, timeout_micros)
    }

    /// Make this task yield periodically.
    fn yielding(self, yield_interval: usize) -> YieldingTask<Self> {
        YieldingTask::new(self, yield_interval)
    }

    /// Convert this task to a future.
    fn into_future(self) -> TaskFuture<Self> {
        TaskFuture::new(self)
    }
}

impl<T: Task> TaskExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TaskBuilder, TaskId};

    #[test]
    fn test_cancellable_task() {
        let task = TaskBuilder::new(|| 42, TaskId::new(1)).build();
        let mut cancellable = task.cancellable();
        
        assert!(!cancellable.is_cancelled());
        assert!(cancellable.is_stealable());
        
        cancellable.cancel();
        assert!(cancellable.is_cancelled());
        assert!(!cancellable.is_stealable());
        
        let result = cancellable.execute();
        assert!(matches!(result, Err(crate::error::TaskError::Cancelled)));
    }

    #[test]
    fn test_timed_task() {
        let task = TaskBuilder::new(|| 42, TaskId::new(1)).build();
        let timed = task.timeout(1000);
        
        let result = timed.execute();
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_task_future() {
        let task = TaskBuilder::new(|| 42, TaskId::new(1)).build();
        let future = task.into_future();
        
        assert!(future.is_ready());
    }
}