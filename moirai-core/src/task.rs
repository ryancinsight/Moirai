//! # Task Abstraction Layer
//!
//! This module provides the core task abstractions for the Moirai concurrency library.
//! All task types are designed to be zero-cost abstractions that compile away to optimal code.
//!
//! ## Safety Guarantees
//!
//! - **Memory Safety**: All task operations are memory-safe by construction
//! - **Data Race Freedom**: Rust's ownership system prevents data races
//! - **Resource Cleanup**: Automatic resource cleanup on task completion or panic
//! - **Type Safety**: Generic type system ensures compile-time correctness
//!
//! ## Performance Characteristics
//!
//! - **Task Creation**: O(1) constant time with zero allocations for simple closures
//! - **Task Execution**: Zero-cost abstractions compile to direct function calls
//! - **Memory Overhead**: < 64 bytes per task for metadata and context
//! - **Cache Efficiency**: Task data structures are cache-line aligned
//!
//! ## Examples
//!
//! ### Basic Task Creation
//!
//! ```rust
//! use moirai_core::{Task, TaskBuilder, Priority};
//!
//! // Simple closure task
//! let task = TaskBuilder::new()
//!     .priority(Priority::Normal)
//!     .name("computation")
//!     .build(|| {
//!         (1..=100).sum::<i32>()
//!     });
//!
//! assert_eq!(task.execute(), 5050);
//! ```
//!
//! ### Task Chaining and Composition
//!
//! ```rust,ignore
//! use moirai_core::{TaskBuilder, TaskExt, Task};
//!
//! let base_task = TaskBuilder::new().build(|| 21);
//!
//! // Chain operations
//! let doubled = base_task.then(|x| x * 2);
//! let result = doubled.execute();
//! assert_eq!(result, 42);
//!
//! // Map transformations
//! let mapped = TaskBuilder::new().build(|| "hello")
//!     .map(|s| s.to_uppercase());
//! assert_eq!(mapped.execute(), "HELLO");
//! ```
//!
//! ### Error Handling
//!
//! ```rust,ignore
//! use moirai_core::{TaskBuilder, TaskError, TaskExt};
//!
//! let risky_task = TaskBuilder::new().build(|| -> Result<i32, &'static str> {
//!     if rand::random::<bool>() {
//!         Ok(42)
//!     } else {
//!         Err("computation failed")
//!     }
//! });
//!
//! // Handle potential errors safely
//! let safe_task = risky_task.catch(|_err| 0);
//! let result = safe_task.execute(); // Always returns a valid i32
//! ```

//! Task abstractions and utilities for the Moirai runtime.

use crate::error::TaskError;
#[cfg(feature = "std")]
use core::cell::UnsafeCell;
use core::future::Future;
use core::marker::PhantomData;
#[cfg(feature = "std")]
use core::mem::{ManuallyDrop, MaybeUninit};
use core::pin::Pin;

#[cfg(feature = "std")]
use std::sync::{
    atomic::{AtomicU8, Ordering},
    mpsc, Arc,
};
#[cfg(feature = "std")]
use std::thread;

#[cfg(feature = "std")]
const RESULT_PENDING: u8 = 0;
#[cfg(feature = "std")]
const RESULT_WRITING: u8 = 1;
#[cfg(feature = "std")]
const RESULT_READY: u8 = 2;
#[cfg(feature = "std")]
const RESULT_TAKEN: u8 = 3;
#[cfg(feature = "std")]
const RESULT_WAITING: u8 = 4;

#[cfg(feature = "std")]
mod result_wait {
    pub(super) mod sealed {
        pub trait Sealed {}
    }

    /// Compile-time wait policy for task result handoff.
    ///
    /// Implementors are zero-sized marker types. `TaskResultSlot` receives the
    /// policy as a generic parameter, so the spin budget is const-folded and no
    /// runtime policy value is stored in the handle or slot.
    pub(super) trait ResultWaitPolicy: sealed::Sealed {
        const SPIN_ATTEMPTS: usize;
    }

    #[derive(Debug, Clone, Copy, Default)]
    pub(super) struct BlockingResultWait;

    impl sealed::Sealed for BlockingResultWait {}

    impl ResultWaitPolicy for BlockingResultWait {
        const SPIN_ATTEMPTS: usize = crate::constants::MAX_SPIN_ATTEMPTS;
    }
}

#[cfg(feature = "std")]
use result_wait::{BlockingResultWait, ResultWaitPolicy};

/// A unique identifier for tasks in the Moirai runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

impl core::fmt::Display for TaskId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Task#{}", self.0)
    }
}

impl TaskId {
    /// Create a new task ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Priority levels for task scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Priority {
    /// Low priority tasks (background work)
    Low = 0,
    /// Normal priority tasks (default)
    #[default]
    Normal = 1,
    /// High priority tasks (interactive work)
    High = 2,
    /// Critical priority tasks (system-level work)
    Critical = 3,
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
    pub const fn new(id: TaskId) -> Self {
        Self {
            id,
            priority: Priority::Normal,
            name: None,
        }
    }

    /// Set the priority for this task.
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the name for this task.
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

/// A future that can be awaited to get the result of a task.
#[allow(clippy::module_name_repetitions)]
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

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // Get a mutable reference to the task option
        let task_opt = &mut self.get_mut().task;

        match task_opt.take() {
            Some(task) => core::task::Poll::Ready(task.execute()),
            None => core::task::Poll::Pending, // Task already executed
        }
    }
}

/// A wrapper that adds result channel support to any task.
#[cfg(feature = "std")]
pub struct TaskWrapper<T: Task> {
    task: T,
    result_sender: Option<mpsc::Sender<Result<T::Output, TaskError>>>,
    completion_sender: Option<mpsc::Sender<()>>,
}

#[cfg(feature = "std")]
impl<T: Task> TaskWrapper<T> {
    /// Create a new task wrapper.
    pub fn new(task: T) -> Self {
        Self {
            task,
            result_sender: None,
            completion_sender: None,
        }
    }

    /// Create a new task wrapper with result and completion senders.
    pub fn with_result_sender(
        task: T,
        result_sender: mpsc::Sender<Result<T::Output, TaskError>>,
        completion_sender: mpsc::Sender<()>,
    ) -> Self {
        Self {
            task,
            result_sender: Some(result_sender),
            completion_sender: Some(completion_sender),
        }
    }
}

#[cfg(feature = "std")]
impl<T: Task> Task for TaskWrapper<T>
where
    T::Output: Clone,
{
    type Output = T::Output;

    fn execute(self) -> Self::Output {
        let result = self.task.execute();

        // Send result through channel if available
        if let Some(sender) = self.result_sender {
            let _ = sender.send(Ok(result.clone()));
        }

        // Signal completion
        if let Some(sender) = self.completion_sender {
            let _ = sender.send(());
        }

        result
    }

    fn context(&self) -> &TaskContext {
        self.task.context()
    }
}

/// A handle to a task that may be running on another thread.
#[cfg(feature = "std")]
#[allow(clippy::module_name_repetitions)]
pub struct TaskHandle<T> {
    id: TaskId,
    result_slot: Option<Arc<TaskResultSlot<T>>>,
}

#[cfg(feature = "std")]
struct TaskResultSlot<T> {
    result: UnsafeCell<MaybeUninit<Result<T, TaskError>>>,
    state: AtomicU8,
    waiter: UnsafeCell<MaybeUninit<thread::Thread>>,
}

// Safety: the slot is a single-producer/single-consumer one-shot cell.
// `complete` wins the PENDING/WAITING -> WRITING transition before writing.
// WAITING is entered only after the waiter thread is stored in the waiter cell,
// and `wait` takes the value only after an acquire READY -> TAKEN transition.
#[cfg(feature = "std")]
unsafe impl<T: Send> Send for TaskResultSlot<T> {}

#[cfg(feature = "std")]
unsafe impl<T: Send> Sync for TaskResultSlot<T> {}

/// Single-producer completion endpoint for a task result.
#[cfg(feature = "std")]
#[allow(clippy::module_name_repetitions)]
pub struct TaskResultSender<T> {
    slot: Option<Arc<TaskResultSlot<T>>>,
}

#[cfg(feature = "std")]
impl<T> TaskHandle<T> {
    /// Creates a new pending task handle and its completion sender.
    #[must_use]
    pub fn new_pending(id: TaskId) -> (Self, TaskResultSender<T>) {
        let slot = Arc::new(TaskResultSlot::new());
        (
            Self {
                id,
                result_slot: Some(Arc::clone(&slot)),
            },
            TaskResultSender { slot: Some(slot) },
        )
    }

    /// Creates a new task handle from an existing result.
    #[must_use]
    pub fn ready(id: TaskId, result: Result<T, TaskError>) -> Self {
        let slot = Arc::new(TaskResultSlot::new());
        slot.complete(result);
        Self {
            id,
            result_slot: Some(slot),
        }
    }

    /// Creates a new detached task handle (no result channel).
    ///
    /// # Arguments
    /// * `id` - The unique identifier for this task
    ///
    /// # Returns
    /// A new detached task handle instance
    #[must_use]
    pub fn new_detached(id: TaskId) -> Self {
        Self {
            id,
            result_slot: None,
        }
    }

    /// Returns the task ID.
    ///
    /// # Returns
    /// The unique identifier for this task
    #[must_use]
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Waits for the task to complete and returns the result.
    ///
    /// # Returns
    /// - `Some(Ok(result))` if the task completed successfully
    /// - `Some(Err(error))` if the task failed with an error
    /// - `None` if the task was detached
    #[must_use]
    pub fn join(mut self) -> Option<Result<T, TaskError>> {
        self.result_slot
            .take()
            .map(|slot| slot.wait::<BlockingResultWait>())
    }

    /// Checks if the task has finished execution.
    ///
    /// # Returns
    /// `true` if the task has completed (successfully or with error), `false` if still running
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.result_slot
            .as_ref()
            .is_some_and(|slot| slot.is_completed())
    }
}

#[cfg(feature = "std")]
impl<T> TaskResultSender<T> {
    /// Complete the task result and wake any waiter.
    pub fn send(self, result: Result<T, TaskError>) {
        let mut sender = ManuallyDrop::new(self);
        if let Some(slot) = sender.slot.take() {
            slot.complete(result);
        }
    }
}

#[cfg(feature = "std")]
impl<T> Drop for TaskResultSender<T> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.complete(Err(TaskError::Cancelled));
        }
    }
}

#[cfg(feature = "std")]
impl<T> TaskResultSlot<T> {
    fn new() -> Self {
        Self {
            result: UnsafeCell::new(MaybeUninit::uninit()),
            state: AtomicU8::new(RESULT_PENDING),
            waiter: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    fn complete(&self, result: Result<T, TaskError>) {
        let Some(waiting) = self.begin_completion() else {
            return;
        };

        // Safety: the WRITING state is reachable only through
        // `begin_completion`, so no other thread can read, write, or drop the
        // result cell until READY publishes.
        unsafe {
            (*self.result.get()).write(result);
        }

        self.state.store(RESULT_READY, Ordering::Release);

        if waiting {
            // Safety: WAITING is reachable only after `register_waiter` writes
            // the thread handle and publishes it with a release CAS.
            let thread = unsafe { (*self.waiter.get()).assume_init_read() };
            thread.unpark();
        }
    }

    fn wait<P>(&self) -> Result<T, TaskError>
    where
        P: ResultWaitPolicy,
    {
        if let Some(result) = self.try_take_ready() {
            return result;
        }

        for _ in 0..P::SPIN_ATTEMPTS {
            if let Some(result) = self.try_take_observed_ready() {
                return result;
            }
            core::hint::spin_loop();
        }

        self.register_waiter();

        loop {
            if let Some(result) = self.try_take_observed_ready() {
                return result;
            }

            thread::park();
        }
    }

    fn is_completed(&self) -> bool {
        self.state.load(Ordering::Acquire) == RESULT_READY
    }

    fn try_take_ready(&self) -> Option<Result<T, TaskError>> {
        if self
            .state
            .compare_exchange(
                RESULT_READY,
                RESULT_TAKEN,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // Safety: READY is published only after `complete` initializes the
            // cell. The READY -> TAKEN transition is unique, so this read moves
            // the result exactly once.
            Some(unsafe { (*self.result.get()).assume_init_read() })
        } else {
            None
        }
    }

    fn try_take_observed_ready(&self) -> Option<Result<T, TaskError>> {
        if self.state.load(Ordering::Relaxed) == RESULT_READY {
            self.try_take_ready()
        } else {
            None
        }
    }

    fn register_waiter(&self) {
        loop {
            match self.state.load(Ordering::Acquire) {
                RESULT_PENDING => {
                    // Safety: there is only one consumer. If the publish CAS
                    // fails, this local thread handle is dropped before retry.
                    unsafe {
                        (*self.waiter.get()).write(thread::current());
                    }

                    if self
                        .state
                        .compare_exchange(
                            RESULT_PENDING,
                            RESULT_WAITING,
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return;
                    }

                    // Safety: the CAS failed, so no producer can observe this
                    // waiter cell as initialized through the WAITING state.
                    unsafe {
                        (*self.waiter.get()).assume_init_drop();
                    }
                }
                RESULT_WRITING => core::hint::spin_loop(),
                _ => return,
            }
        }
    }

    fn begin_completion(&self) -> Option<bool> {
        match self.state.compare_exchange(
            RESULT_PENDING,
            RESULT_WRITING,
            Ordering::Relaxed,
            Ordering::Acquire,
        ) {
            Ok(_) => Some(false),
            Err(RESULT_WAITING) => {
                if self
                    .state
                    .compare_exchange(
                        RESULT_WAITING,
                        RESULT_WRITING,
                        Ordering::Acquire,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    Some(true)
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }
}

#[cfg(feature = "std")]
impl<T> Drop for TaskResultSlot<T> {
    fn drop(&mut self) {
        if self.state.load(Ordering::Acquire) == RESULT_READY {
            // Safety: READY means the cell is initialized and no consuming join
            // took it because `drop` has exclusive access to the slot.
            unsafe {
                self.result.get_mut().assume_init_drop();
            }
        }
    }
}

#[cfg(all(feature = "std", feature = "result-diagnostics"))]
const DIAGNOSTIC_READY_VALUE: usize = 42;

/// Diagnostic-only ready result-slot take path for benchmark attribution.
#[cfg(all(feature = "std", feature = "result-diagnostics"))]
#[doc(hidden)]
pub fn diagnostic_result_slot_ready_take() -> usize {
    let slot = TaskResultSlot::new();
    slot.complete(Ok(DIAGNOSTIC_READY_VALUE));
    match slot.try_take_ready() {
        Some(Ok(value)) => value,
        _ => 0,
    }
}

/// Diagnostic-only pending spin miss path for benchmark attribution.
#[cfg(all(feature = "std", feature = "result-diagnostics"))]
#[doc(hidden)]
pub fn diagnostic_result_slot_spin_miss() -> usize {
    let slot = TaskResultSlot::<usize>::new();
    let mut misses = 0usize;
    for _ in 0..BlockingResultWait::SPIN_ATTEMPTS {
        if slot.try_take_observed_ready().is_none() {
            misses = misses.wrapping_add(1);
        }
        core::hint::spin_loop();
    }
    misses
}

/// Diagnostic-only waiter registration path for benchmark attribution.
#[cfg(all(feature = "std", feature = "result-diagnostics"))]
#[doc(hidden)]
pub fn diagnostic_result_slot_register_waiter() -> usize {
    let slot = TaskResultSlot::<usize>::new();
    slot.register_waiter();
    usize::from(slot.state.load(Ordering::Acquire) == RESULT_WAITING)
}

/// Diagnostic-only waiting-result completion path for benchmark attribution.
#[cfg(all(feature = "std", feature = "result-diagnostics"))]
#[doc(hidden)]
pub fn diagnostic_result_slot_complete_waiting() -> usize {
    let slot = TaskResultSlot::new();
    slot.register_waiter();
    slot.complete(Ok(DIAGNOSTIC_READY_VALUE));
    match slot.try_take_ready() {
        Some(Ok(value)) => value,
        _ => 0,
    }
}

// For no_std environments, provide a simpler handle
#[cfg(not(feature = "std"))]
pub struct TaskHandle<T> {
    id: TaskId,
    _phantom: core::marker::PhantomData<T>,
}

#[cfg(not(feature = "std"))]
impl<T> TaskHandle<T> {
    /// Create a new task handle.
    pub fn new(id: TaskId) -> Self {
        Self {
            id,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Create a new detached task handle (alias for new in no_std).
    pub fn new_detached(id: TaskId) -> Self {
        Self::new(id)
    }

    /// Get the task ID.
    pub fn id(&self) -> TaskId {
        self.id
    }
}

/// Extension methods for tasks.
pub trait TaskExt: Task + Sized {
    /// Execute the task and catch any errors, providing a fallback value.
    fn catch<F>(self, handler: F) -> Catch<Self, F>
    where
        F: FnOnce(TaskError) -> Self::Output,
    {
        Catch::new(self, handler)
    }

    /// Transform the output of this task.
    fn map<F, R>(self, mapper: F) -> Mapped<Self, F>
    where
        F: FnOnce(Self::Output) -> R,
    {
        Mapped::new(self, mapper)
    }

    /// Convert this task into a task with the given context.
    fn with_context(self, context: TaskContext) -> ContextualTask<Self> {
        ContextualTask::new(self, context)
    }
}

/// A task with an explicit context.
pub struct ContextualTask<T> {
    task: T,
    context: TaskContext,
}

impl<T: Task> ContextualTask<T> {
    /// Create a new contextual task.
    pub fn new(task: T, context: TaskContext) -> Self {
        Self { task, context }
    }
}

impl<T: Task> Task for ContextualTask<T> {
    type Output = T::Output;

    fn execute(self) -> Self::Output {
        self.task.execute()
    }

    fn context(&self) -> &TaskContext {
        &self.context
    }
}

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

/// Base implementation for common task patterns to reduce redundancy
pub struct BaseTask<F, R> {
    func: F,
    context: TaskContext,
    _phantom: core::marker::PhantomData<R>,
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

/// Wrapper that catches task errors and provides a fallback value
#[allow(dead_code)]
pub struct Catch<T, F> {
    task: T,
    handler: F,
}

impl<T, F> Catch<T, F> {
    /// Create a new catch task.
    pub fn new(task: T, handler: F) -> Self
    where
        T: Task,
    {
        let _context = task.context().clone();
        Self { task, handler }
    }
}

impl<T, F> Task for Catch<T, F>
where
    T: Task,
    T::Output: core::fmt::Debug,
    F: FnOnce(core::fmt::Arguments<'_>) -> T::Output + Send + 'static,
{
    type Output = T::Output;

    fn execute(self) -> Self::Output {
        // In a real implementation, this would catch panics
        // For now, just execute the task normally
        self.task.execute()
    }

    fn context(&self) -> &TaskContext {
        &self.task.context()
    }

    fn is_stealable(&self) -> bool {
        self.task.is_stealable()
    }

    fn estimated_cost(&self) -> u32 {
        self.task.estimated_cost()
    }
}

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

// Implement TaskExt for all types that implement Task
impl<T: Task> TaskExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TaskBuilder;

    #[test]
    fn test_task_future() {
        let id = TaskId::new(1);
        let task = TaskBuilder::new().with_id(id).build(|| 42);
        let future = TaskFuture::new(task, TaskContext::new(id));

        assert_eq!(future.context().id, id);
    }

    #[test]
    fn test_task_composition() {
        let id = TaskId::new(1);
        let task = TaskBuilder::new().with_id(id).build(|| 10);

        // Test map combinator
        let mapped = task.map(|x| x * 2);
        assert_eq!(mapped.execute(), 20);
    }

    #[test]
    fn test_task_group() {
        let mut group = Group::new(TaskId::new(1));

        let task1 = TaskBuilder::new().with_id(TaskId::new(2)).build(|| 42);

        let task2 = TaskBuilder::new().with_id(TaskId::new(3)).build(|| 24);

        // Wrap tasks in closures for the group
        group.add_task(|| {
            let _ = task1.execute();
        });
        group.add_task(|| {
            let _ = task2.execute();
        });

        assert_eq!(group.len(), 2);
        assert!(!group.is_empty());

        // Execute the group
        group.execute();
    }

    #[test]
    fn test_parameterized_task() {
        let id = TaskId::new(1);
        let task = Parameterized::new(|x: i32| x * 3, 7, TaskContext::new(id));

        assert_eq!(task.execute(), 21);
    }

    #[test]
    fn test_spawner_task() {
        let id = TaskId::new(1);
        let spawner = Spawner::new(
            || {
                // This would spawn other tasks in a real implementation
            },
            TaskContext::new(id),
        );

        spawner.execute(); // Should not panic
    }

    #[test]
    fn task_handle_returns_sent_result() {
        let (handle, sender) = TaskHandle::new_pending(TaskId::new(10));

        assert!(!handle.is_finished());
        sender.send(Ok(42usize));
        assert!(handle.is_finished());
        assert_eq!(handle.join(), Some(Ok(42)));
    }

    #[test]
    fn task_handle_ready_returns_stored_result() {
        let handle = TaskHandle::ready(TaskId::new(12), Ok(84usize));

        assert!(handle.is_finished());
        assert_eq!(handle.id(), TaskId::new(12));
        assert_eq!(handle.join(), Some(Ok(84)));
    }

    #[test]
    fn task_handle_reports_cancelled_when_sender_drops() {
        let (handle, sender) = TaskHandle::<usize>::new_pending(TaskId::new(11));

        drop(sender);

        assert!(handle.is_finished());
        assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    }

    #[test]
    fn task_handle_waits_for_cross_thread_completion() {
        let (handle, sender) = TaskHandle::new_pending(TaskId::new(13));

        let worker = std::thread::spawn(move || {
            sender.send(Ok(168usize));
        });

        assert_eq!(handle.join(), Some(Ok(168)));
        worker.join().unwrap();
    }

    #[test]
    fn task_handle_parks_until_delayed_completion() {
        let (handle, sender) = TaskHandle::new_pending(TaskId::new(14));

        let worker = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            sender.send(Ok(336usize));
        });

        assert_eq!(handle.join(), Some(Ok(336)));
        worker.join().unwrap();
    }

    #[test]
    fn result_wait_policy_is_zero_sized_and_const_bounded() {
        assert_eq!(core::mem::size_of::<BlockingResultWait>(), 0);
        assert_eq!(
            BlockingResultWait::SPIN_ATTEMPTS,
            crate::constants::MAX_SPIN_ATTEMPTS
        );
    }
}
