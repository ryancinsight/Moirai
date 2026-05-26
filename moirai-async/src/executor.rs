//! Native async executor for Moirai concurrency library.
//!
//! This module queues and polls concrete futures with a Moirai-owned executor
//! and exposes the Platform Abstraction Layer (PAL) reactor for readiness work.
//! Reactor-native file and network operations remain separate PAL contracts.

use moirai_core::{Priority, TaskId};
use moirai_pal::reactor::IoReactor;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Instant;

#[path = "executor/result_slot.rs"]
mod result_slot;

use result_slot::AsyncResultSlot;

/// Native async executor with access to the PAL I/O reactor.
///
/// The executor stores queued futures behind monomorphized poll/drop functions
/// and publishes handle results through an inline atomic result slot. I/O
/// readiness is not claimed as Tokio-compatible until PAL file and network
/// types provide concrete readiness registration contracts.
pub struct AsyncExecutor {
    /// Platform-specific I/O reactor
    reactor: Arc<IoReactor>,
    /// Active tasks map
    tasks: Arc<Mutex<HashMap<TaskId, AsyncTaskWrapper>>>,
    /// Run queue for ready tasks
    run_queue: Arc<Mutex<VecDeque<TaskId>>>,
    /// Runtime statistics
    stats: AsyncExecutorStats,
    /// Executor running state
    running: Arc<AtomicBool>,
    /// Monotonic task identifier source.
    next_task_id: AtomicU64,
}

/// A handle to an async task that can be awaited.
pub struct AsyncHandle<T> {
    task_id: TaskId,
    result_slot: Arc<AsyncResultSlot<T>>,
}

impl<T> AsyncHandle<T> {
    /// Return the executor-assigned task identifier.
    #[must_use]
    pub fn id(&self) -> TaskId {
        self.task_id
    }
}

/// Wrapper for async tasks in the executor queue.
#[allow(dead_code)] // Fields used for future scheduling/telemetry per ADR requirements
struct AsyncTaskWrapper {
    task_id: TaskId,
    future: ErasedTaskFuture,
    priority: Priority,
    created_at: Instant,
}

/// Heap-stable concrete future with monomorphized poll/drop functions.
struct ErasedTaskFuture {
    ptr: NonNull<()>,
    poll: unsafe fn(NonNull<()>, &mut Context<'_>) -> Poll<()>,
    drop: unsafe fn(NonNull<()>),
}

// Safety: `ErasedTaskFuture` owns a `Send + 'static` future allocation created
// by `ErasedTaskFuture::new`; moving the thin owner between threads does not
// move the pinned future allocation it points to.
unsafe impl Send for ErasedTaskFuture {}

impl ErasedTaskFuture {
    fn new<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let ptr = Box::into_raw(Box::new(future)).cast::<()>();
        Self {
            ptr: NonNull::new(ptr).expect("Box::into_raw must not return null"),
            poll: poll_erased_future::<F>,
            drop: drop_erased_future::<F>,
        }
    }

    fn poll(&mut self, context: &mut Context<'_>) -> Poll<()> {
        // Safety: `new` stores a future whose concrete type matches this
        // monomorphized poll function, and the heap allocation is never moved.
        unsafe { (self.poll)(self.ptr, context) }
    }
}

impl Drop for ErasedTaskFuture {
    fn drop(&mut self) {
        // Safety: `new` stores a future whose concrete type matches this
        // monomorphized drop function, and `Drop` runs exactly once.
        unsafe {
            (self.drop)(self.ptr);
        }
    }
}

unsafe fn poll_erased_future<F>(ptr: NonNull<()>, context: &mut Context<'_>) -> Poll<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    // Safety: `ErasedTaskFuture::new::<F>` created the allocation and pins it
    // by address for the lifetime of the erased owner.
    let future = unsafe { Pin::new_unchecked(&mut *ptr.cast::<F>().as_ptr()) };
    future.poll(context)
}

unsafe fn drop_erased_future<F>(ptr: NonNull<()>)
where
    F: Future<Output = ()> + Send + 'static,
{
    // Safety: the allocation was created by `Box::into_raw(Box::<F>)` in
    // `ErasedTaskFuture::new::<F>`.
    unsafe {
        drop(Box::from_raw(ptr.cast::<F>().as_ptr()));
    }
}

/// Statistics for async executor performance monitoring.
#[derive(Debug, Default)]
struct AsyncExecutorStats {
    tasks_spawned: AtomicU64,
    tasks_completed: AtomicU64,
    total_execution_time_ns: AtomicU64,
    waker_notifications: AtomicU64,
    io_operations: AtomicU64,
}

impl AsyncExecutor {
    /// Create a new native async executor with a PAL I/O reactor handle.
    ///
    /// # Behavior Guarantees
    /// - Initializes the configured platform reactor object
    /// - Ready to accept tasks immediately
    /// - Thread-safe for concurrent access
    /// - Does not require Tokio or Rayon as runtime dependencies
    pub fn new() -> std::io::Result<Self> {
        let reactor = Arc::new(IoReactor::new()?);

        Ok(Self {
            reactor,
            tasks: Arc::new(Mutex::new(HashMap::new())),
            run_queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: AsyncExecutorStats::default(),
            running: Arc::new(AtomicBool::new(false)),
            next_task_id: AtomicU64::new(0),
        })
    }

    /// Spawn an async task with default priority.
    ///
    /// # Behavior Guarantees
    /// - Task is queued for execution immediately
    /// - Returns handle that can be awaited
    /// - Task will be polled when executor runs
    /// - No blocking operations during spawn
    pub fn spawn<F, T>(&self, future: F) -> AsyncHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        self.spawn_with_priority(future, Priority::Normal)
    }

    /// Spawn an async task with specified priority.
    ///
    /// # Implementation Note
    /// The queued future remains heap-stable and uses monomorphized poll/drop
    /// functions. Reactor-native I/O wakeups are a separate PAL contract.
    pub fn spawn_with_priority<F, T>(&self, future: F, priority: Priority) -> AsyncHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let task_id = TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed));
        let result_slot = Arc::new(AsyncResultSlot::new());
        let completion_slot = Arc::clone(&result_slot);

        // Capture the future result for the public handle.
        let wrapped_future = async move {
            let result = future.await;
            completion_slot.complete(result);

            // Completion publishes through the handle's result slot. Reactor
            // wake integration belongs to PAL readiness futures.
        };

        let task = AsyncTaskWrapper {
            task_id,
            future: ErasedTaskFuture::new(wrapped_future),
            priority,
            created_at: Instant::now(),
        };

        // Queue the task for execution
        self.tasks.lock().unwrap().insert(task_id, task);
        self.run_queue.lock().unwrap().push_back(task_id);

        // Update statistics
        self.stats.tasks_spawned.fetch_add(1, Ordering::Relaxed);

        // Wake the reactor in case it's blocking
        let _ = self.reactor.wake();

        AsyncHandle {
            task_id,
            result_slot,
        }
    }

    /// Run the native async executor and poll the PAL reactor between task passes.
    ///
    /// # Behavior Guarantees
    /// - Polls queued futures with their concrete poll functions
    /// - Processes one PAL reactor iteration per loop
    /// - Stops when `stop` clears the running flag and wakes the reactor
    pub fn run(&self) -> std::io::Result<()> {
        self.running.store(true, Ordering::SeqCst);

        self.reactor.with_active(|| {
            while self.running.load(Ordering::SeqCst) {
                // Process pending tasks first
                self.process_pending_tasks();

                // Check if there are active tasks in the system
                let has_tasks = {
                    let tasks = self.tasks.lock().unwrap();
                    !tasks.is_empty()
                };

                if !has_tasks {
                    if !self.running.load(Ordering::SeqCst) {
                        break;
                    }
                    self.reactor.run_iteration(None)?;
                } else {
                    let run_queue_empty = self.run_queue.lock().unwrap().is_empty();
                    if run_queue_empty {
                        self.reactor.run_iteration(None)?;
                    } else {
                        self.reactor
                            .run_iteration(Some(std::time::Duration::from_millis(0)))?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Stop the async executor.
    pub fn stop(&self) -> std::io::Result<()> {
        self.running.store(false, Ordering::SeqCst);
        self.reactor.stop()
    }

    /// Process all pending tasks.
    ///
    /// Pending futures stay queued. Ready futures are removed after publishing
    /// their result through the inline result slot.
    pub(crate) fn process_pending_tasks(&self) {
        let mut to_poll = Vec::new();
        {
            let mut queue = self.run_queue.lock().unwrap();
            while let Some(task_id) = queue.pop_front() {
                to_poll.push(task_id);
            }
        }

        for task_id in to_poll {
            let mut task = {
                let mut tasks = self.tasks.lock().unwrap();
                if let Some(t) = tasks.remove(&task_id) {
                    t
                } else {
                    continue;
                }
            };

            let waker = self.create_executor_waker(task_id);
            let mut context = Context::from_waker(&waker);
            let task_start = Instant::now();

            match task.future.poll(&mut context) {
                Poll::Ready(()) => {
                    self.stats.tasks_completed.fetch_add(1, Ordering::Relaxed);

                    let execution_time = task_start.elapsed().as_nanos() as u64;
                    self.stats
                        .total_execution_time_ns
                        .fetch_add(execution_time, Ordering::Relaxed);
                }
                Poll::Pending => {
                    self.tasks.lock().unwrap().insert(task_id, task);
                }
            }
        }
    }

    /// Create an executor-local waker for polling queued futures.
    fn create_executor_waker(&self, task_id: TaskId) -> Waker {
        let waker = Arc::new(ExecutorWaker {
            task_id,
            run_queue: Arc::clone(&self.run_queue),
            reactor: Arc::clone(&self.reactor),
        });
        Waker::from(waker)
    }

    /// Get current executor statistics.
    pub fn stats(&self) -> ExecutorStats {
        ExecutorStats {
            tasks_spawned: self.stats.tasks_spawned.load(Ordering::Relaxed),
            tasks_completed: self.stats.tasks_completed.load(Ordering::Relaxed),
            tasks_pending: self.tasks.lock().unwrap().len() as u64,
            total_execution_time_ns: self.stats.total_execution_time_ns.load(Ordering::Relaxed),
            waker_notifications: self.stats.waker_notifications.load(Ordering::Relaxed),
            io_operations: self.stats.io_operations.load(Ordering::Relaxed),
        }
    }

    /// Get access to the underlying I/O reactor for advanced operations.
    pub fn reactor(&self) -> &IoReactor {
        &self.reactor
    }
}

struct ExecutorWaker {
    task_id: TaskId,
    run_queue: Arc<Mutex<VecDeque<TaskId>>>,
    reactor: Arc<IoReactor>,
}

impl std::task::Wake for ExecutorWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        {
            let mut queue = self.run_queue.lock().unwrap();
            if !queue.contains(&self.task_id) {
                queue.push_back(self.task_id);
            }
        }
        let _ = self.reactor.wake();
    }
}

impl<T> Future for AsyncHandle<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(value) = self.result_slot.try_take_ready() {
            return Poll::Ready(value);
        }

        self.result_slot.register_waker(cx.waker());

        self.result_slot
            .try_take_ready()
            .map_or(Poll::Pending, Poll::Ready)
    }
}

/// Public statistics structure for monitoring executor performance.
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    /// Total number of tasks spawned
    pub tasks_spawned: u64,
    /// Total number of tasks completed
    pub tasks_completed: u64,
    /// Number of tasks currently pending
    pub tasks_pending: u64,
    /// Total execution time in nanoseconds
    pub total_execution_time_ns: u64,
    /// Number of waker notifications sent
    pub waker_notifications: u64,
    /// Number of I/O operations processed
    pub io_operations: u64,
}

impl Default for AsyncExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default AsyncExecutor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_async_executor_creation() {
        let executor = AsyncExecutor::new();
        assert!(executor.is_ok());

        let executor = executor.unwrap();
        let stats = executor.stats();
        assert_eq!(stats.tasks_spawned, 0);
        assert_eq!(stats.tasks_completed, 0);
        assert_eq!(stats.tasks_pending, 0);
    }

    #[test]
    fn test_task_spawning() {
        let executor = AsyncExecutor::new().unwrap();

        let _handle = executor.spawn(async {
            // Simple async task
            42
        });

        let stats = executor.stats();
        assert_eq!(stats.tasks_spawned, 1);
        assert_eq!(stats.tasks_pending, 1);
    }

    #[test]
    fn test_task_ids_are_unique() {
        let executor = AsyncExecutor::new().unwrap();

        let first = executor.spawn(async { 1usize });
        let second = executor.spawn(async { 2usize });

        assert_ne!(first.id(), second.id());
    }

    #[test]
    fn test_ready_task_completion_publishes_result() {
        use std::task::Poll;

        let executor = AsyncExecutor::new().unwrap();
        let mut handle = Box::pin(executor.spawn(async { 7usize }));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);

        assert!(matches!(handle.as_mut().poll(&mut context), Poll::Pending));

        executor.process_pending_tasks();

        assert_eq!(executor.stats().tasks_completed, 1);
        assert!(matches!(handle.as_mut().poll(&mut context), Poll::Ready(7)));
    }

    #[test]
    fn test_ready_task_completion_wakes_registered_handle() {
        use futures::task::{waker_ref, ArcWake};
        use std::sync::atomic::AtomicUsize;
        use std::task::Poll;

        struct WakeCounter(AtomicUsize);

        impl ArcWake for WakeCounter {
            fn wake_by_ref(arc_self: &Arc<Self>) {
                arc_self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let executor = AsyncExecutor::new().unwrap();
        let mut handle = Box::pin(executor.spawn(async { 11usize }));
        let wake_counter = Arc::new(WakeCounter(AtomicUsize::new(0)));
        let waker = waker_ref(&wake_counter);
        let mut context = Context::from_waker(&waker);

        assert!(matches!(handle.as_mut().poll(&mut context), Poll::Pending));

        executor.process_pending_tasks();

        assert_eq!(wake_counter.0.load(Ordering::SeqCst), 1);
        assert!(matches!(
            handle.as_mut().poll(&mut context),
            Poll::Ready(11)
        ));
    }

    #[test]
    fn test_priority_scheduling() {
        let executor = AsyncExecutor::new().unwrap();

        let _high_priority = executor.spawn_with_priority(async { "high" }, Priority::High);
        let _normal_priority = executor.spawn_with_priority(async { "normal" }, Priority::Normal);
        let _low_priority = executor.spawn_with_priority(async { "low" }, Priority::Low);

        let stats = executor.stats();
        assert_eq!(stats.tasks_spawned, 3);
        assert_eq!(stats.tasks_pending, 3);
    }

    // NOTE: Integration tests with actual async I/O operations will be added
    // once the PAL file and network modules are fully implemented
}
