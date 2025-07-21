//! Task abstractions and utilities for the Moirai runtime.

use crate::{TaskId, TaskContext, Task, Box};
use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

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
    pub fn new(task: T, context: TaskContext) -> Self {
        Self {
            task: Some(task),
            context,
        }
    }

    /// Get the task context.
    pub fn context(&self) -> &TaskContext {
        &self.context
    }
}

impl<T> Future for TaskFuture<T>
where
    T: Task + Unpin,
{
    type Output = T::Output;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Get a mutable reference to the task option
        let task_opt = &mut self.get_mut().task;
        
        match task_opt.take() {
            Some(task) => Poll::Ready(task.execute()),
            None => Poll::Pending, // Task already executed
        }
    }
}

/// Extension trait for tasks to provide additional functionality.
pub trait TaskExt: Task {
    /// Convert this task into a future.
    fn into_future(self) -> TaskFuture<Self>
    where
        Self: Sized + Unpin,
    {
        let context = self.context().clone();
        TaskFuture::new(self, context)
    }

    /// Create a boxed version of this task.
    fn boxed(self) -> Box<dyn Task<Output = Self::Output>>
    where
        Self: Sized + 'static,
        Self::Output: 'static,
    {
        Box::new(self)
    }

    /// Chain this task with another task.
    fn then<F, U>(self, func: F) -> ChainedTask<Self, F>
    where
        Self: Sized,
        F: FnOnce(Self::Output) -> U + Send + 'static,
        U: Send + 'static,
    {
        ChainedTask::new(self, func)
    }

    /// Map the output of this task.
    fn map<F, U>(self, func: F) -> MappedTask<Self, F>
    where
        Self: Sized,
        F: FnOnce(Self::Output) -> U + Send + 'static,
        U: Send + 'static,
    {
        MappedTask::new(self, func)
    }

    /// Add error handling to this task.
    fn catch<F>(self, handler: F) -> CatchTask<Self, F>
    where
        Self: Sized,
        F: FnOnce() -> Self::Output + Send + 'static,
    {
        CatchTask::new(self, handler)
    }
}

// Implement TaskExt for all types that implement Task
impl<T: Task> TaskExt for T {}



/// A task that chains two operations together.
pub struct ChainedTask<T, F> {
    task: Option<T>,
    func: Option<F>,
    context: TaskContext,
}

impl<T, F> ChainedTask<T, F> {
    /// Create a new chained task.
    pub fn new(task: T, func: F) -> Self
    where
        T: Task,
    {
        let context = task.context().clone();
        Self {
            task: Some(task),
            func: Some(func),
            context,
        }
    }
}

impl<T, F, U> Task for ChainedTask<T, F>
where
    T: Task,
    F: FnOnce(T::Output) -> U + Send + 'static,
    U: Send + 'static,
{
    type Output = U;

    fn execute(mut self) -> Self::Output {
        let task = self.task.take().expect("Task already executed");
        let func = self.func.take().expect("Function already used");
        let result = task.execute();
        func(result)
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_stealable(&self) -> bool {
        self.task.as_ref().map_or(false, |t| t.is_stealable())
    }

    fn estimated_cost(&self) -> u32 {
        self.task.as_ref().map_or(1, |t| t.estimated_cost())
    }
}

/// A task that maps the output of another task.
pub struct MappedTask<T, F> {
    task: Option<T>,
    func: Option<F>,
    context: TaskContext,
}

impl<T, F> MappedTask<T, F> {
    /// Create a new mapped task.
    pub fn new(task: T, func: F) -> Self
    where
        T: Task,
    {
        let context = task.context().clone();
        Self {
            task: Some(task),
            func: Some(func),
            context,
        }
    }
}

impl<T, F, U> Task for MappedTask<T, F>
where
    T: Task,
    F: FnOnce(T::Output) -> U + Send + 'static,
    U: Send + 'static,
{
    type Output = U;

    fn execute(mut self) -> Self::Output {
        let task = self.task.take().expect("Task already executed");
        let func = self.func.take().expect("Function already used");
        let result = task.execute();
        func(result)
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_stealable(&self) -> bool {
        self.task.as_ref().map_or(false, |t| t.is_stealable())
    }

    fn estimated_cost(&self) -> u32 {
        self.task.as_ref().map_or(1, |t| t.estimated_cost())
    }
}

/// A task that provides error handling.
pub struct CatchTask<T, F> {
    task: Option<T>,
    handler: Option<F>,
    context: TaskContext,
}

impl<T, F> CatchTask<T, F> {
    /// Create a new catch task.
    pub fn new(task: T, handler: F) -> Self
    where
        T: Task,
    {
        let context = task.context().clone();
        Self {
            task: Some(task),
            handler: Some(handler),
            context,
        }
    }
}

impl<T, F> Task for CatchTask<T, F>
where
    T: Task,
    F: FnOnce() -> T::Output + Send + 'static,
{
    type Output = T::Output;

    fn execute(mut self) -> Self::Output {
        let task = self.task.take().expect("Task already executed");
        let handler = self.handler.take().expect("Handler already used");
        
        // For no_std compatibility, we'll just execute the task normally
        // In a std environment, this could catch panics
        #[cfg(feature = "std")]
        {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task.execute())) {
                Ok(result) => result,
                Err(_) => handler(),
            }
        }
        
        #[cfg(not(feature = "std"))]
        {
            // In no_std, we can't catch panics, so just execute normally
            task.execute()
        }
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_stealable(&self) -> bool {
        self.task.as_ref().map_or(false, |t| t.is_stealable())
    }

    fn estimated_cost(&self) -> u32 {
        self.task.as_ref().map_or(1, |t| t.estimated_cost())
    }
}

/// A task that can be spawned multiple times with different inputs.
pub struct ParameterizedTask<F, P> {
    func: F,
    params: P,
    context: TaskContext,
}

impl<F, P> ParameterizedTask<F, P> {
    /// Create a new parameterized task.
    pub fn new(func: F, params: P, context: TaskContext) -> Self {
        Self {
            func,
            params,
            context,
        }
    }
}

impl<F, P, R> Task for ParameterizedTask<F, P>
where
    F: FnOnce(P) -> R + Send + 'static,
    P: Send + 'static,
    R: Send + 'static,
{
    type Output = R;

    fn execute(self) -> Self::Output {
        (self.func)(self.params)
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }
}

/// A collection of tasks that can be executed together.
pub struct TaskGroup {
    tasks: alloc::vec::Vec<Box<dyn FnOnce() + Send + 'static>>,
    context: TaskContext,
}

impl TaskGroup {
    /// Create a new task group.
    pub fn new(id: TaskId) -> Self {
        Self {
            tasks: alloc::vec::Vec::new(),
            context: TaskContext::new(id),
        }
    }

    /// Add a task to the group.
    pub fn add_task<T>(&mut self, task: T)
    where
        T: Task<Output = ()> + 'static,
    {
        self.tasks.push(Box::new(move || {
            task.execute();
        }));
    }

    /// Get the number of tasks in the group.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if the group is empty.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

impl Task for TaskGroup {
    type Output = ();

    fn execute(self) -> Self::Output {
        // Execute each task function
        for task_fn in self.tasks.into_iter() {
            task_fn();
        }
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn estimated_cost(&self) -> u32 {
        // Estimate based on number of tasks
        self.tasks.len() as u32
    }
}

/// A task that spawns other tasks.
pub struct SpawnerTask<F> {
    spawner: Option<F>,
    context: TaskContext,
}

impl<F> SpawnerTask<F> {
    /// Create a new spawner task.
    pub fn new(spawner: F, context: TaskContext) -> Self {
        Self {
            spawner: Some(spawner),
            context,
        }
    }
}

impl<F> Task for SpawnerTask<F>
where
    F: FnOnce() + Send + 'static,
{
    type Output = ();

    fn execute(mut self) -> Self::Output {
        if let Some(spawner) = self.spawner.take() {
            spawner();
        }
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Priority, TaskBuilder};

    #[test]
    fn test_task_future() {
        let id = TaskId::new(1);
        let task = TaskBuilder::new(|| 42, id).build();
        let future = TaskFuture::new(task, TaskContext::new(id));
        
        assert_eq!(future.context().id, id);
    }

    #[test]
    fn test_chained_task() {
        let id = TaskId::new(1);
        let task = TaskBuilder::new(|| 10, id).build();
        let chained = task.then(|x| x * 2);
        
        assert_eq!(chained.execute(), 20);
    }

    #[test]
    fn test_mapped_task() {
        let id = TaskId::new(1);
        let task = TaskBuilder::new(|| 5, id).build();
        let mapped = task.map(|x| x.to_string());
        
        assert_eq!(mapped.execute(), "5");
    }

    #[test]
    fn test_task_group() {
        let mut group = TaskGroup::new(TaskId::new(1));
        
        let task1 = TaskBuilder::new(|| (), TaskId::new(2)).build();
        let task2 = TaskBuilder::new(|| (), TaskId::new(3)).build();
        
        group.add_task(task1);
        group.add_task(task2);
        
        assert_eq!(group.len(), 2);
        assert!(!group.is_empty());
        
        group.execute(); // Should not panic
    }

    #[test]
    fn test_parameterized_task() {
        let id = TaskId::new(1);
        let task = ParameterizedTask::new(|x: i32| x * 3, 7, TaskContext::new(id));
        
        assert_eq!(task.execute(), 21);
    }

    #[test]
    fn test_spawner_task() {
        let id = TaskId::new(1);
        let spawner = SpawnerTask::new(|| {
            // This would spawn other tasks in a real implementation
        }, TaskContext::new(id));
        
        spawner.execute(); // Should not panic
    }
}