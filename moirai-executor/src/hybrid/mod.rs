//! Main hybrid executor implementation.
//!
//! `HybridExecutor` exposes one public execution surface while delegating sync,
//! async, and blocking work to the same thread scheduler. The work-shape choice
//! is encoded by zero-sized marker types in `crate::schedule`.

use std::{
    cell::UnsafeCell,
    future::Future,
    mem::MaybeUninit,
    panic::{catch_unwind, AssertUnwindSafe},
    pin::Pin,
    ptr::{self, NonNull},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll, Wake, Waker},
};

use moirai_core::constants::DEFAULT_POLL_INTERVAL_MS;
use moirai_core::{
    error::{ExecutorError, ExecutorResult, TaskError},
    executor::{
        Executor, ExecutorConfig, ExecutorControl, ExecutorStats, TaskManager, TaskSpawner,
        TaskStats, TaskStatus,
    },
    task::{Task, TaskHandle, TaskId, TaskResultSender},
    Priority,
};

use crate::{
    metrics::ExecutorMetrics,
    registry::{RunningTaskToken, TaskLifecycleToken, TaskRegistry},
    schedule::{
        wake::block_on_current_thread, AsyncTask, BlockingTask, SchedulerScope, SyncTask,
        ThreadScheduler, WorkClass,
    },
};

const ASYNC_IDLE: u8 = 0;
const ASYNC_QUEUED: u8 = 1;
const ASYNC_POLLING: u8 = 2;
const ASYNC_NOTIFIED: u8 = 3;
const ASYNC_COMPLETED: u8 = 4;
const ASYNC_INLINE_REPOLL_LIMIT: usize = 1;

#[derive(Clone, Copy)]
struct MetricsRef {
    metrics: NonNull<ExecutorMetrics>,
}

// Safety: `MetricsRef` points at `HybridExecutor.metrics`. The executor owns
// the scheduler and drains scheduled jobs during shutdown/drop before dropping
// the metrics allocation, so scheduled synchronous/blocking jobs cannot observe
// a dangling metrics pointer.
unsafe impl Send for MetricsRef {}

impl MetricsRef {
    #[inline]
    fn new(metrics: &Arc<ExecutorMetrics>) -> Self {
        Self {
            metrics: NonNull::from(metrics.as_ref()),
        }
    }

    #[inline]
    fn get(self) -> &'static ExecutorMetrics {
        // Safety: see the `Send` impl invariant. The returned reference is used
        // only inside scheduled jobs that complete before executor destruction.
        unsafe { self.metrics.as_ref() }
    }
}

/// Main hybrid executor that coordinates sync, async, and blocking tasks.
pub struct HybridExecutor {
    config: ExecutorConfig,
    scheduler: ThreadScheduler,
    task_registry: Arc<Mutex<TaskRegistry>>,
    metrics: Arc<ExecutorMetrics>,
    shutdown_signal: Arc<AtomicBool>,
    next_task_id: AtomicU64,
}

impl HybridExecutor {
    /// Create a new hybrid executor with the given configuration.
    pub fn new(config: ExecutorConfig) -> ExecutorResult<Self> {
        let scheduler = ThreadScheduler::new(config.worker_threads, &config.thread_name_prefix)?;
        let metrics = Arc::new(ExecutorMetrics::new());
        metrics.update_worker_counts(0, scheduler.worker_count(), scheduler.worker_count());

        Ok(Self {
            config,
            scheduler,
            task_registry: Arc::new(Mutex::new(TaskRegistry::new())),
            metrics,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            next_task_id: AtomicU64::new(1),
        })
    }

    /// Get executor configuration.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Start the executor.
    pub fn start(&mut self) -> ExecutorResult<()> {
        Ok(())
    }

    /// Shutdown the executor gracefully.
    pub fn shutdown(&mut self) -> ExecutorResult<()> {
        self.shutdown_signal.store(true, Ordering::Release);
        self.scheduler.shutdown();
        Ok(())
    }

    /// Get executor metrics.
    pub fn metrics(&self) -> &ExecutorMetrics {
        self.refresh_scheduler_metrics();
        &self.metrics
    }

    /// Submit an untyped synchronous job.
    pub fn submit_task<F>(&self, task: F) -> ExecutorResult<TaskId>
    where
        F: FnOnce() + Send + 'static,
    {
        let task_id = self.allocate_task_id();
        let lifecycle = self.register_task(task_id)?;

        let metrics = MetricsRef::new(&self.metrics);
        self.scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |worker_id| {
                let running = lifecycle.start(worker_id);
                let succeeded = catch_unwind(AssertUnwindSafe(task)).is_ok();
                if succeeded {
                    metrics.get().record_task_completed(running.complete());
                } else {
                    running.complete();
                    metrics.get().record_task_failed();
                }
            })?;

        self.metrics.record_task_spawned();
        Ok(task_id)
    }

    /// Run a scoped fan-out directly on the unified scheduler.
    ///
    /// This path is for completion-only work that does not require per-task
    /// result handles or lifecycle metadata. It preserves borrowing semantics
    /// by waiting for all spawned jobs before returning.
    pub fn scope<'scope, C, F>(&'scope self, body: F) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(&SchedulerScope<'scope, C>) -> ExecutorResult<()>,
    {
        self.scheduler.scope::<C, _>(Priority::Normal, None, body)
    }

    /// Run indexed work in worker-sized chunks on the unified scheduler.
    ///
    /// This path avoids per-item task handles and lifecycle metadata when the
    /// caller only needs completion for a bounded index domain.
    pub fn for_each_indexed<'scope, C, F>(&'scope self, count: usize, task: F) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: Fn(usize) + Send + Sync + 'scope,
    {
        self.scheduler
            .for_each_indexed::<C, _>(Priority::Normal, None, count, task)
    }

    /// Run indexed map/reduce in worker-sized chunks on the unified scheduler.
    pub fn map_reduce_indexed<'scope, C, T, Map, Reduce>(
        &'scope self,
        count: usize,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> ExecutorResult<T>
    where
        C: WorkClass,
        T: Send + Clone + 'scope,
        Map: Fn(usize) -> T + Send + Sync + 'scope,
        Reduce: Fn(T, T) -> T + Send + Sync + 'scope,
    {
        self.scheduler.map_reduce_indexed::<C, _, _, _>(
            Priority::Normal,
            None,
            count,
            identity,
            map,
            reduce,
        )
    }

    /// Get the number of active workers.
    pub fn active_workers(&self) -> usize {
        self.scheduler.active_workers()
    }

    /// Get the total number of workers.
    pub fn total_workers(&self) -> usize {
        self.scheduler.worker_count()
    }

    /// Get pending task count across all workers.
    pub fn pending_tasks(&self) -> usize {
        self.scheduler.pending_tasks()
    }

    /// Returns true when queued or active scheduler work exists.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Wait until queued and active scheduler work completes without shutting down workers.
    pub fn join(&self) -> ExecutorResult<()> {
        self.scheduler.join()?;
        self.refresh_scheduler_metrics();
        Ok(())
    }

    fn allocate_task_id(&self) -> TaskId {
        TaskId::new(self.next_task_id.fetch_add(1, Ordering::Relaxed))
    }

    fn register_task(&self, task_id: TaskId) -> ExecutorResult<TaskLifecycleToken> {
        let mut registry = self.task_registry.lock().map_err(|_| {
            ExecutorError::ResourceExhausted("task registry lock poisoned".to_string())
        })?;
        Ok(registry.register_task_with_id(task_id.0))
    }

    fn refresh_scheduler_metrics(&self) {
        let snapshot = self.scheduler.metrics();
        self.metrics.update_worker_counts(
            snapshot.active_workers,
            snapshot
                .worker_count
                .saturating_sub(snapshot.active_workers),
            snapshot.worker_count,
        );
        self.metrics.update_queue_metrics(snapshot.pending_tasks);
    }
}

impl TaskSpawner for HybridExecutor {
    fn spawn<T>(&self, task: T) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
        T::Output: Send + 'static,
    {
        let priority = task.context().priority;
        let task_id = self.allocate_task_id();
        let lifecycle = self.register_task(task_id)?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<SyncTask, _>(priority, None, move |worker_id| {
                let running = lifecycle.start(worker_id);
                let result = catch_unwind(AssertUnwindSafe(|| task.execute()));
                let execution_time = running.complete();
                send_task_result(result, result_sender, metrics.get(), execution_time);
            })?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }

    fn spawn_async<F>(&self, future: F) -> ExecutorResult<TaskHandle<F::Output>>
    where
        F: core::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let task_id = self.allocate_task_id();
        let lifecycle = self.register_task(task_id)?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let state = AsyncFutureState::new(
            self.scheduler.clone(),
            future,
            lifecycle,
            result_sender,
            Arc::clone(&self.metrics),
        );
        Arc::clone(&state).schedule()?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }

    fn spawn_blocking<F, R>(&self, func: F) -> ExecutorResult<TaskHandle<R>>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let task_id = self.allocate_task_id();
        let lifecycle = self.register_task(task_id)?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |worker_id| {
                let running = lifecycle.start(worker_id);
                let result = catch_unwind(AssertUnwindSafe(func));
                let execution_time = running.complete();
                send_task_result(result, result_sender, metrics.get(), execution_time);
            })?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }

    fn spawn_with_priority<T>(
        &self,
        task: T,
        priority: Priority,
        locality_hint: Option<usize>,
    ) -> ExecutorResult<TaskHandle<T::Output>>
    where
        T: Task + Send + 'static,
        T::Output: Send + 'static,
    {
        let task_id = self.allocate_task_id();
        let lifecycle = self.register_task(task_id)?;

        let (handle, result_sender) = TaskHandle::new_pending(task_id);
        let metrics = MetricsRef::new(&self.metrics);

        self.scheduler
            .schedule::<SyncTask, _>(priority, locality_hint, move |worker_id| {
                let running = lifecycle.start(worker_id);
                let result = catch_unwind(AssertUnwindSafe(|| task.execute()));
                let execution_time = running.complete();
                send_task_result(result, result_sender, metrics.get(), execution_time);
            })?;

        self.metrics.record_task_spawned();
        Ok(handle)
    }
}

struct AsyncFutureState<F>
where
    F: Future,
{
    scheduler: ThreadScheduler,
    future: UnsafeCell<MaybeUninit<F>>,
    lifecycle: UnsafeCell<AsyncLifecycle>,
    result_sender: UnsafeCell<Option<TaskResultSender<F::Output>>>,
    metrics: Arc<ExecutorMetrics>,
    state: AtomicU8,
    future_present: UnsafeCell<bool>,
}

enum AsyncLifecycle {
    Registered(TaskLifecycleToken),
    Running(RunningTaskToken),
    Completed,
}

// Safety: `state` serializes all future polling. Wakers may schedule work
// concurrently, but they only mutate atomics and never touch the future cell.
// The future cell is dropped either by the unique polling thread after Ready or
// panic, or by `Drop` after the last `Arc` reference is gone.
unsafe impl<F> Send for AsyncFutureState<F>
where
    F: Future + Send,
    F::Output: Send,
{
}

// Safety: see the `Send` impl. Shared references are used only for atomic
// scheduling, metrics, and fields guarded by the single poll owner selected by
// the async state machine.
unsafe impl<F> Sync for AsyncFutureState<F>
where
    F: Future + Send,
    F::Output: Send,
{
}

impl<F> AsyncFutureState<F>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    fn new(
        scheduler: ThreadScheduler,
        future: F,
        lifecycle: TaskLifecycleToken,
        result_sender: TaskResultSender<F::Output>,
        metrics: Arc<ExecutorMetrics>,
    ) -> Arc<Self> {
        Arc::new(Self {
            scheduler,
            future: UnsafeCell::new(MaybeUninit::new(future)),
            lifecycle: UnsafeCell::new(AsyncLifecycle::Registered(lifecycle)),
            result_sender: UnsafeCell::new(Some(result_sender)),
            metrics,
            state: AtomicU8::new(ASYNC_IDLE),
            future_present: UnsafeCell::new(true),
        })
    }

    #[inline]
    fn schedule(self: Arc<Self>) -> ExecutorResult<()> {
        loop {
            match self.state.load(Ordering::Acquire) {
                ASYNC_IDLE => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_IDLE,
                            ASYNC_QUEUED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return self.enqueue();
                    }
                }
                ASYNC_POLLING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_POLLING,
                            ASYNC_NOTIFIED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Ok(());
                    }
                }
                ASYNC_QUEUED | ASYNC_NOTIFIED | ASYNC_COMPLETED => return Ok(()),
                _ => return Ok(()),
            }
        }
    }

    #[inline]
    fn schedule_by_ref(self: &Arc<Self>) -> ExecutorResult<()> {
        loop {
            match self.state.load(Ordering::Acquire) {
                ASYNC_POLLING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_POLLING,
                            ASYNC_NOTIFIED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Ok(());
                    }
                }
                ASYNC_IDLE => return Arc::clone(self).schedule(),
                ASYNC_QUEUED | ASYNC_NOTIFIED | ASYNC_COMPLETED => return Ok(()),
                _ => return Ok(()),
            }
        }
    }

    #[inline]
    fn enqueue(self: Arc<Self>) -> ExecutorResult<()> {
        let state = Arc::clone(&self);
        self.scheduler
            .schedule::<AsyncTask, _>(Priority::Normal, None, move |worker_id| {
                state.poll(worker_id);
            })
    }

    fn poll(self: &Arc<Self>, worker_id: usize) {
        if self
            .state
            .compare_exchange(
                ASYNC_QUEUED,
                ASYNC_POLLING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }

        self.mark_running(worker_id);
        let waker = Waker::from(Arc::clone(self));
        let mut context = Context::from_waker(&waker);
        let mut inline_repolls = 0usize;

        loop {
            let poll_result = {
                // Safety: `state` grants this worker the only polling
                // permission, and the `Arc` allocation keeps the address
                // stable while the future is pinned. Future storage remains
                // initialized until the poll owner reaches ready or panic.
                let future = unsafe { Pin::new_unchecked(&mut *(*self.future.get()).as_mut_ptr()) };
                catch_unwind(AssertUnwindSafe(|| future.poll(&mut context)))
            };

            match poll_result {
                Ok(Poll::Ready(output)) => {
                    self.drop_future();
                    self.state.store(ASYNC_COMPLETED, Ordering::Release);
                    let execution_time = self.complete_lifecycle();
                    if let Some(sender) = self.take_result_sender() {
                        sender.send(Ok(output));
                    }
                    self.metrics.record_task_completed(execution_time);
                    return;
                }
                Ok(Poll::Pending) => {
                    if self.finish_pending_poll(&mut inline_repolls) {
                        continue;
                    }
                    return;
                }
                Err(_) => {
                    self.drop_future();
                    self.state.store(ASYNC_COMPLETED, Ordering::Release);
                    self.complete_lifecycle();
                    if let Some(sender) = self.take_result_sender() {
                        sender.send(Err(TaskError::Panicked));
                    }
                    self.metrics.record_task_failed();
                    return;
                }
            }
        }
    }

    fn mark_running(&self, worker_id: usize) {
        // Safety: only the poll owner selected by the async state machine calls
        // this method, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        if matches!(*lifecycle, AsyncLifecycle::Registered(_)) {
            let registered = std::mem::replace(lifecycle, AsyncLifecycle::Completed);
            if let AsyncLifecycle::Registered(token) = registered {
                *lifecycle = AsyncLifecycle::Running(token.start(worker_id));
            }
        }
    }

    fn complete_lifecycle(&self) -> core::time::Duration {
        // Safety: only the poll owner selected by the async state machine calls
        // this method, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        let running = std::mem::replace(lifecycle, AsyncLifecycle::Completed);
        if let AsyncLifecycle::Running(token) = running {
            token.complete()
        } else {
            core::time::Duration::ZERO
        }
    }

    fn drop_future(&self) {
        // Safety: only the poll owner selected by the async state machine calls
        // this method while shared references exist. `Drop` reaches the same
        // flag only after the final `Arc` is gone and has exclusive access.
        // The poll hot path does not read this flag; `state` is the authoritative
        // polling permission and guarantees initialized future storage.
        let future_present = unsafe { &mut *self.future_present.get() };
        if *future_present {
            *future_present = false;
            // Safety: the caller owns poll/completion permission or `Drop` owns
            // the last state reference. The initialized future is dropped once.
            unsafe {
                ptr::drop_in_place((*self.future.get()).as_mut_ptr());
            }
        }
    }

    fn take_result_sender(&self) -> Option<TaskResultSender<F::Output>> {
        // Safety: result publication is reached only by the single poll owner
        // selected by the async state machine. `Drop` has exclusive access after
        // the last `Arc` is gone and does not read this cell.
        unsafe { (&mut *self.result_sender.get()).take() }
    }

    #[inline]
    fn finish_pending_poll(self: &Arc<Self>, inline_repolls: &mut usize) -> bool {
        match self.state.compare_exchange(
            ASYNC_POLLING,
            ASYNC_IDLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => false,
            Err(ASYNC_NOTIFIED) if *inline_repolls < ASYNC_INLINE_REPOLL_LIMIT => {
                if self
                    .state
                    .compare_exchange(
                        ASYNC_NOTIFIED,
                        ASYNC_POLLING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    *inline_repolls += 1;
                    true
                } else {
                    false
                }
            }
            Err(ASYNC_NOTIFIED) => {
                self.state.store(ASYNC_IDLE, Ordering::Release);
                let _ = Arc::clone(self).schedule();
                false
            }
            Err(_) => false,
        }
    }
}

impl<F> Drop for AsyncFutureState<F>
where
    F: Future,
{
    fn drop(&mut self) {
        if *self.future_present.get_mut() {
            // Safety: `Drop` has exclusive access to the state because the last
            // `Arc` reference is being destroyed.
            unsafe {
                ptr::drop_in_place((*self.future.get()).as_mut_ptr());
            }
        }
    }
}

impl<F> Wake for AsyncFutureState<F>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    fn wake(self: Arc<Self>) {
        let _ = self.schedule();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.schedule_by_ref();
    }
}

impl TaskManager for HybridExecutor {
    fn cancel_task(&self, id: TaskId) -> ExecutorResult<()> {
        let registry = self.task_registry.lock().map_err(|_| {
            ExecutorError::ResourceExhausted("task registry lock poisoned".to_string())
        })?;

        if registry.get_metadata(id.0).is_some() {
            Ok(())
        } else {
            Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation))
        }
    }

    fn task_status(&self, id: TaskId) -> Option<TaskStatus> {
        let registry = self.task_registry.lock().ok()?;
        registry.get_metadata(id.0).map(|metadata| {
            if metadata.completed_at.is_some() {
                TaskStatus::Completed
            } else if metadata.started_at.is_some() {
                TaskStatus::Running
            } else {
                TaskStatus::Queued
            }
        })
    }

    fn wait_for_task(
        &self,
        id: TaskId,
        timeout: Option<core::time::Duration>,
    ) -> impl core::future::Future<Output = ExecutorResult<()>> + Send {
        let registry = Arc::clone(&self.task_registry);
        async move {
            let start = std::time::Instant::now();

            loop {
                let registry = registry.lock().map_err(|_| {
                    ExecutorError::ResourceExhausted("task registry lock poisoned".to_string())
                })?;
                if registry.is_completed(id.0) {
                    return Ok(());
                }

                if registry.get_metadata(id.0).is_none() {
                    return Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation));
                }
                drop(registry);

                if let Some(timeout) = timeout {
                    if start.elapsed() >= timeout {
                        return Err(ExecutorError::ResourceExhausted(
                            "Task wait timeout".to_string(),
                        ));
                    }
                }

                std::thread::sleep(std::time::Duration::from_millis(DEFAULT_POLL_INTERVAL_MS));
            }
        }
    }

    fn task_stats(&self, id: TaskId) -> Option<TaskStats> {
        let registry = self.task_registry.lock().ok()?;
        registry.get_metadata(id.0).map(|metadata| TaskStats {
            id,
            priority: Priority::Normal,
            status: if metadata.completed_at.is_some() {
                TaskStatus::Completed
            } else if metadata.started_at.is_some() {
                TaskStatus::Running
            } else {
                TaskStatus::Queued
            },
            spawn_time: metadata.created_at,
            start_time: metadata.started_at,
            completion_time: metadata.completed_at,
            preemption_count: 0,
            cpu_time_ns: metadata
                .execution_duration()
                .map_or(0, |duration| duration.as_nanos() as u64),
            memory_used_bytes: 0,
        })
    }
}

impl ExecutorControl for HybridExecutor {
    fn block_on<F>(&self, future: F) -> F::Output
    where
        F: core::future::Future,
    {
        block_on_current_thread(future)
    }

    fn try_run(&self) -> bool {
        self.refresh_scheduler_metrics();
        self.scheduler.has_work()
    }

    fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Release);
        self.scheduler.shutdown();
    }

    fn shutdown_timeout(&self, _timeout: core::time::Duration) {
        self.shutdown();
    }

    fn is_shutting_down(&self) -> bool {
        self.shutdown_signal.load(Ordering::Acquire)
    }

    fn worker_count(&self) -> usize {
        self.scheduler.worker_count()
    }

    fn load(&self) -> usize {
        self.scheduler.pending_tasks()
    }
}

impl Executor for HybridExecutor {
    #[cfg(feature = "metrics")]
    fn stats(&self) -> ExecutorStats {
        self.refresh_scheduler_metrics();
        ExecutorStats {
            tasks_executed: self.metrics.tasks_completed.load(Ordering::Acquire),
            tasks_queued: self.scheduler.pending_tasks(),
            avg_execution_time_ns: self.metrics.average_task_duration().as_nanos() as u64,
        }
    }
}

impl Drop for HybridExecutor {
    fn drop(&mut self) {
        let _ = HybridExecutor::shutdown(self);
    }
}

#[inline]
fn send_task_result<T>(
    result: Result<T, Box<dyn std::any::Any + Send>>,
    sender: TaskResultSender<T>,
    metrics: &ExecutorMetrics,
    execution_time: core::time::Duration,
) where
    T: Send + 'static,
{
    match result {
        Ok(value) => {
            sender.send(Ok(value));
            metrics.record_task_completed(execution_time);
        }
        Err(_) => {
            sender.send(Err(TaskError::Panicked));
            metrics.record_task_failed();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HybridExecutor;
    use moirai_core::{
        executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus},
        task::TaskBuilder,
        Priority,
    };

    #[test]
    fn spawn_blocking_returns_value_and_updates_status() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 2,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handle = executor.spawn_blocking(|| 21 * 2).unwrap();
        let id = handle.id();
        let result = handle.join().unwrap().unwrap();

        assert_eq!(result, 42);
        assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
        executor.shutdown();
    }

    #[test]
    fn spawn_blocking_reports_panicked_result() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handle = executor
            .spawn_blocking(|| -> usize { panic!("blocking task panic") })
            .unwrap();

        assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Panicked)));
        executor.shutdown();
    }

    #[test]
    fn spawn_async_uses_unified_scheduler() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 2,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handle = executor.spawn_async(async { 7usize }).unwrap();
        let result = handle.join().unwrap().unwrap();

        assert_eq!(result, 7);
        assert_eq!(executor.worker_count(), 2);
        executor.shutdown();
    }

    #[test]
    fn spawn_async_requeues_after_wake_without_blocking_worker() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc, Mutex,
        };
        use std::task::Waker;
        use std::time::{Duration, Instant};

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let ready = Arc::new(AtomicBool::new(false));
        let waker_slot = Arc::new(Mutex::new(None::<Waker>));
        let ready_for_future = Arc::clone(&ready);
        let waker_for_future = Arc::clone(&waker_slot);
        let handle = executor
            .spawn_async(async {
                std::future::poll_fn(move |cx| {
                    if ready_for_future.load(Ordering::Acquire) {
                        std::task::Poll::Ready(21usize)
                    } else {
                        *waker_for_future.lock().unwrap() = Some(cx.waker().clone());
                        std::task::Poll::Pending
                    }
                })
                .await
            })
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        let waker = loop {
            if let Some(waker) = waker_slot.lock().unwrap().take() {
                break waker;
            }

            assert!(
                Instant::now() < deadline,
                "async future must publish a waker before timeout"
            );
            std::thread::sleep(Duration::from_millis(1));
        };

        let (ran_sender, ran_receiver) = mpsc::channel();
        let independent = executor
            .spawn_blocking(move || {
                ran_sender.send(()).unwrap();
                13usize
            })
            .unwrap();

        ran_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("pending async future must not block the only worker");

        ready.store(true, Ordering::Release);
        waker.wake();

        assert_eq!(independent.join().unwrap().unwrap(), 13);
        assert_eq!(handle.join().unwrap().unwrap(), 21);
        executor.shutdown();
    }

    #[test]
    fn spawn_async_completes_single_self_wake() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let poll_count = Arc::new(AtomicUsize::new(0));
        let poll_count_for_future = Arc::clone(&poll_count);
        let handle = executor
            .spawn_async(async move {
                std::future::poll_fn(move |context| {
                    match poll_count_for_future.fetch_add(1, Ordering::AcqRel) {
                        0 => {
                            context.waker().wake_by_ref();
                            std::task::Poll::Pending
                        }
                        previous => std::task::Poll::Ready(previous + 1),
                    }
                })
                .await
            })
            .unwrap();

        assert_eq!(handle.join().unwrap().unwrap(), 2);
        assert_eq!(poll_count.load(Ordering::Acquire), 2);
        executor.shutdown();
    }

    #[test]
    fn priority_spawn_preserves_task_result() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let task = TaskBuilder::new()
            .priority(Priority::Critical)
            .build(|| 11usize);
        let handle = executor
            .spawn_with_priority(task, Priority::Critical, Some(0))
            .unwrap();

        assert_eq!(handle.join().unwrap().unwrap(), 11);
        executor.shutdown();
    }

    #[test]
    fn join_waits_for_public_result_tasks_without_shutdown() {
        let executor = HybridExecutor::new(ExecutorConfig {
            worker_threads: 2,
            ..ExecutorConfig::default()
        })
        .unwrap();

        let handles = (0..8)
            .map(|value| executor.spawn_blocking(move || value + 1).unwrap())
            .collect::<Vec<_>>();

        assert!(executor.has_work());
        executor.join().unwrap();
        assert!(!executor.has_work());

        let results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(results, (1..=8).collect::<Vec<_>>());
        executor.shutdown();
    }
}
