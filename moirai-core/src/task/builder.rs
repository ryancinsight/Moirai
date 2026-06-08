use core::marker::PhantomData;

use super::id_and_context::{TaskContext, TaskId};
use super::traits::Task;

// ── BaseTask ──────────────────────────────────────────────────────────────────

/// Base implementation for common task patterns to reduce redundancy.
pub struct BaseTask<F, R> {
    pub(super) func: F,
    pub(super) context: TaskContext,
    pub(super) _phantom: PhantomData<R>,
}

impl<F, R> BaseTask<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    /// Create a new base task with the given function and context.
    pub fn new(func: F, context: TaskContext) -> Self {
        Self {
            func,
            context,
            _phantom: PhantomData,
        }
    }
}

// ── Closure ───────────────────────────────────────────────────────────────────

/// A simple closure-based task implementation.
pub struct Closure<F, R> {
    base: BaseTask<F, R>,
}

impl<F, R> Closure<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    /// Create a new closure task.
    pub fn new(func: F, context: TaskContext) -> Self {
        Self {
            base: BaseTask::new(func, context),
        }
    }

    /// Chain another operation after this task.
    pub fn then<G, S>(self, continuation: G) -> Chained<Self, G>
    where
        G: FnOnce(R) -> S + Send + 'static,
        S: Send + 'static,
    {
        Chained::new(self, continuation)
    }

    /// Map the output of this task.
    pub fn map<G, S>(self, mapper: G) -> Mapped<Self, G>
    where
        G: FnOnce(R) -> S + Send + 'static,
        S: Send + 'static,
    {
        Mapped::new(self, mapper)
    }
}

impl<F, R> Task for Closure<F, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    type Output = R;

    fn execute(self) -> Self::Output {
        (self.base.func)()
    }

    fn context(&self) -> &TaskContext {
        &self.base.context
    }
}

// ── Chained ───────────────────────────────────────────────────────────────────

/// A task that chains two operations together.
pub struct Chained<T, F> {
    task: T,
    continuation: F,
    context: TaskContext,
}

impl<T, F> Chained<T, F> {
    /// Create a new chained task.
    pub fn new(task: T, continuation: F) -> Self
    where
        T: Task,
    {
        let _context = task.context().clone();
        Self {
            task,
            continuation,
            context: _context,
        }
    }
}

impl<T, F, U> Task for Chained<T, F>
where
    T: Task,
    F: FnOnce(T::Output) -> U + Send + 'static,
    U: Send + 'static,
{
    type Output = U;

    fn execute(self) -> Self::Output {
        let result = self.task.execute();
        (self.continuation)(result)
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_stealable(&self) -> bool {
        self.task.is_stealable()
    }

    fn estimated_cost(&self) -> u32 {
        self.task.estimated_cost() + 1
    }
}

// ── Mapped ────────────────────────────────────────────────────────────────────

/// A task that maps the output of another task.
pub struct Mapped<T, F> {
    task: T,
    mapper: F,
    context: TaskContext,
}

impl<T, F> Mapped<T, F> {
    /// Create a new mapped task.
    pub fn new(task: T, mapper: F) -> Self
    where
        T: Task,
    {
        let _context = task.context().clone();
        Self {
            task,
            mapper,
            context: _context,
        }
    }
}

impl<T, F, U> Task for Mapped<T, F>
where
    T: Task,
    F: FnOnce(T::Output) -> U + Send + 'static,
    U: Send + 'static,
{
    type Output = U;

    fn execute(self) -> Self::Output {
        let result = self.task.execute();
        (self.mapper)(result)
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    fn is_stealable(&self) -> bool {
        self.task.is_stealable()
    }

    fn estimated_cost(&self) -> u32 {
        self.task.estimated_cost()
    }
}

// ── TaskBuilder ───────────────────────────────────────────────────────────────

/// Builder for creating and configuring tasks.
#[allow(clippy::module_name_repetitions)]
pub struct TaskBuilder {
    context: TaskContext,
}

impl TaskBuilder {
    /// Creates a new task builder with default settings.
    ///
    /// # Returns
    /// A new builder instance ready for configuration
    #[must_use]
    pub fn new() -> Self {
        // Generate a dummy ID for now - this should be replaced by the executor
        Self {
            context: TaskContext::new(TaskId::new(0)),
        }
    }

    /// Sets the priority level for the task.
    ///
    /// # Arguments
    /// * `priority` - The scheduling priority for this task
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn priority(mut self, priority: crate::Priority) -> Self {
        self.context.priority = priority;
        self
    }

    /// Sets a descriptive name for the task.
    ///
    /// # Arguments
    /// * `name` - A static string name for debugging and monitoring
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn name(mut self, name: &'static str) -> Self {
        self.context.name = Some(name);
        self
    }

    /// Sets the task ID and returns the modified task builder.
    ///
    /// # Arguments
    /// * `id` - The unique identifier for this task
    ///
    /// # Returns
    /// The task builder with the specified ID set
    #[must_use]
    pub fn with_id(mut self, id: TaskId) -> Self {
        self.context.id = id;
        self
    }

    /// Build the task with the provided function.
    pub fn build<F, R>(self, func: F) -> Closure<F, R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        Closure::new(func, self.context)
    }
}

impl Default for TaskBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Parameterized ─────────────────────────────────────────────────────────────

/// A task that accepts parameters for customized execution.
///
/// This provides a way to create reusable task templates that can
/// be parameterized at execution time.
pub struct Parameterized<F, P> {
    /// The parameterized function to execute
    function: Option<F>,
    /// The parameters to pass to the function
    parameters: Option<P>,
    /// Task execution context and metadata
    context: TaskContext,
}

impl<F, P> Parameterized<F, P> {
    /// Create a new parameterized task.
    pub fn new(func: F, params: P, context: TaskContext) -> Self {
        Self {
            function: Some(func),
            parameters: Some(params),
            context,
        }
    }
}

impl<F, P, R> Task for Parameterized<F, P>
where
    F: FnOnce(P) -> R + Send + 'static,
    P: Send + 'static,
    R: Send + 'static,
{
    type Output = R;

    fn execute(mut self) -> Self::Output {
        let func = self.function.take().expect("Task already executed");
        let params = self.parameters.take().expect("Parameters already used");
        func(params)
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }
}

// ── Group ─────────────────────────────────────────────────────────────────────

/// A collection of related tasks that can be executed as a group.
///
/// This provides batch execution capabilities and allows for
/// coordinated task management and monitoring.
pub struct Group {
    /// The unique identifier for this task group
    /// Allows the task group ID field to be unused for now
    #[allow(dead_code)]
    id: TaskId,
    /// Collection of tasks in this group
    tasks: Vec<Box<dyn FnOnce() + Send + 'static>>,
    /// Task execution context and metadata
    context: TaskContext,
}

impl Group {
    /// Creates a new task group with the specified ID.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the task group
    ///
    /// # Returns
    /// A new empty task group
    #[must_use]
    pub fn new(id: TaskId) -> Self {
        Self {
            id,
            tasks: Vec::new(),
            context: TaskContext::new(id),
        }
    }

    /// Add a task to the group.
    pub fn add_task<F>(&mut self, task_fn: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.tasks.push(Box::new(move || {
            task_fn();
        }));
    }

    /// Returns the number of tasks in this group.
    ///
    /// # Returns
    /// The count of tasks currently in the group
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Checks if the task group is empty.
    ///
    /// # Returns
    /// `true` if the group contains no tasks, `false` otherwise
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

impl Task for Group {
    type Output = ();

    fn execute(self) -> Self::Output {
        // Execute each task function
        for task_fn in self.tasks {
            task_fn();
        }
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }

    #[allow(clippy::cast_possible_truncation)]
    fn estimated_cost(&self) -> u32 {
        self.tasks.len() as u32
    }
}

// ── Spawner ───────────────────────────────────────────────────────────────────

/// A task that can spawn other tasks during its execution.
///
/// This provides dynamic task creation capabilities, allowing tasks
/// to generate additional work based on runtime conditions.
pub struct Spawner<F> {
    /// The spawning function that creates new tasks
    spawner: Option<F>,
    /// Task execution context and metadata
    context: TaskContext,
}

impl<F> Spawner<F> {
    /// Create a new spawner task.
    pub fn new(spawner: F, context: TaskContext) -> Self {
        Self {
            spawner: Some(spawner),
            context,
        }
    }
}

impl<F> Task for Spawner<F>
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
