//! Unified thread scheduler runtime.

use std::{
    cell::RefCell,
    marker::PhantomData,
    mem,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Condvar, Mutex, MutexGuard, OnceLock,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    Priority,
};

use moirai_utils::cache::CacheAligned;
use moirai_pal::reactor::IoReactor;

use super::{
    class::WorkClass,
    job::ScheduledJob,
    queue::WorkerQueues,
    reduce::{inline_reduction_limit, ReduceSlots},
};

const WORKER_IDLE_SPIN_ATTEMPTS: usize = 256;
const JOIN_FAST_SPIN_ATTEMPTS: usize = WORKER_IDLE_SPIN_ATTEMPTS;

/// Point-in-time scheduler metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduleMetrics {
    /// Number of scheduler workers.
    pub worker_count: usize,
    /// Number of queued jobs not yet executing.
    pub pending_tasks: usize,
    /// Number of jobs currently executing.
    pub active_workers: usize,
    /// Number of jobs completed without panic at the scheduler boundary.
    pub completed_tasks: u64,
    /// Number of jobs that panicked at the scheduler boundary.
    pub failed_tasks: u64,
}

/// Single scheduler used by all executor task classes.
pub struct ThreadScheduler<
    const QUEUE_CAPACITY: usize = 256,
    const SPIN_LIMIT: usize = 256,
> {
    inner: Arc<SchedulerInner<QUEUE_CAPACITY>>,
}

mod contended_wake {
    pub trait Sealed {}
}

trait ContendedWakePolicy: contended_wake::Sealed + Send + Sync + 'static {
    const WAKE_LIMIT: usize;
}

#[derive(Debug, Clone, Copy, Default)]
struct BoundedContendedWake;

impl contended_wake::Sealed for BoundedContendedWake {}

impl ContendedWakePolicy for BoundedContendedWake {
    const WAKE_LIMIT: usize = 2;
}

#[cfg(feature = "scheduler-diagnostics")]
mod diagnostic_wake {
    pub trait Sealed {}
}

#[cfg(feature = "scheduler-diagnostics")]
pub trait DiagnosticWakeDecision: diagnostic_wake::Sealed + Send + Sync + 'static {
    fn previous_pending(worker_count: usize) -> usize;
}

#[cfg(feature = "scheduler-diagnostics")]
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyWakeDecision;

#[cfg(feature = "scheduler-diagnostics")]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContendedWakeDecision;

#[cfg(feature = "scheduler-diagnostics")]
#[derive(Debug, Clone, Copy, Default)]
pub struct SaturatedWakeDecision;

#[cfg(feature = "scheduler-diagnostics")]
impl diagnostic_wake::Sealed for EmptyWakeDecision {}

#[cfg(feature = "scheduler-diagnostics")]
impl diagnostic_wake::Sealed for ContendedWakeDecision {}

#[cfg(feature = "scheduler-diagnostics")]
impl diagnostic_wake::Sealed for SaturatedWakeDecision {}

#[cfg(feature = "scheduler-diagnostics")]
impl DiagnosticWakeDecision for EmptyWakeDecision {
    #[inline]
    fn previous_pending(_: usize) -> usize {
        0
    }
}

#[cfg(feature = "scheduler-diagnostics")]
impl DiagnosticWakeDecision for ContendedWakeDecision {
    #[inline]
    fn previous_pending(_: usize) -> usize {
        1
    }
}

#[cfg(feature = "scheduler-diagnostics")]
impl DiagnosticWakeDecision for SaturatedWakeDecision {
    #[inline]
    fn previous_pending(worker_count: usize) -> usize {
        worker_count
    }
}

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> Clone
    for ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Borrowing scope for scheduler jobs that must complete before the scope exits.
pub struct SchedulerScope<
    'scope,
    C: WorkClass,
    const QUEUE_CAPACITY: usize = 256,
    const SPIN_LIMIT: usize = 256,
> {
    scheduler: &'scope ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>,
    state: NonNull<SchedulerScopeState>,
    priority: Priority,
    locality_hint: Option<usize>,
    jobs: RefCell<Vec<ScheduledJob>>,
    _state: PhantomData<&'scope SchedulerScopeState>,
    _class: PhantomData<C>,
}

struct SchedulerInner<const QUEUE_CAPACITY: usize> {
    workers: Box<[Arc<WorkerState<QUEUE_CAPACITY>>]>,
    handles: Mutex<Vec<JoinHandle<()>>>,
    next_worker: CacheAligned<AtomicUsize>,
    pending_tasks: CacheAligned<AtomicUsize>,
    active_workers: CacheAligned<AtomicUsize>,
    completed_tasks: CacheAligned<AtomicU64>,
    failed_tasks: CacheAligned<AtomicU64>,
    shutdown: CacheAligned<AtomicBool>,
    join_waiters: CacheAligned<AtomicUsize>,
    wait_lock: Mutex<()>,
    wait_signal: Condvar,
    idle_workers: CacheAligned<AtomicU64>,
}

struct LifoSlot {
    state: std::sync::atomic::AtomicU8,
    job: std::cell::UnsafeCell<std::mem::MaybeUninit<ScheduledJob>>,
}

unsafe impl Sync for LifoSlot {}

impl LifoSlot {
    fn new() -> Self {
        Self {
            state: std::sync::atomic::AtomicU8::new(0),
            job: std::cell::UnsafeCell::new(std::mem::MaybeUninit::uninit()),
        }
    }

    fn push(&self, job: ScheduledJob) -> Option<ScheduledJob> {
        let current = self.state.load(Ordering::Relaxed);
        if current == 0 {
            if self.state.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                unsafe {
                    *self.job.get() = std::mem::MaybeUninit::new(job);
                }
                self.state.store(2, Ordering::Release);
                return None;
            }
        } else if current == 2 {
            if self.state.compare_exchange(2, 1, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                let old_job = unsafe {
                    std::ptr::read((*self.job.get()).as_ptr())
                };
                unsafe {
                    *self.job.get() = std::mem::MaybeUninit::new(job);
                }
                self.state.store(2, Ordering::Release);
                return Some(old_job);
            }
        }
        Some(job)
    }

    fn pop(&self) -> Option<ScheduledJob> {
        if self.state.load(Ordering::Relaxed) == 2 {
            if self.state.compare_exchange(2, 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                let job = unsafe {
                    std::ptr::read((*self.job.get()).as_ptr())
                };
                self.state.store(0, Ordering::Release);
                Some(job)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn steal(&self) -> Option<ScheduledJob> {
        if self.state.load(Ordering::Relaxed) == 2 {
            if self.state.compare_exchange(2, 3, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                let job = unsafe {
                    std::ptr::read((*self.job.get()).as_ptr())
                };
                self.state.store(0, Ordering::Release);
                Some(job)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl Drop for LifoSlot {
    fn drop(&mut self) {
        if *self.state.get_mut() == 2 {
            unsafe {
                std::ptr::drop_in_place((*self.job.get()).as_mut_ptr());
            }
        }
    }
}

thread_local! {
    static CURRENT_WORKER_ID: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[repr(align(64))]
struct WorkerState<const QUEUE_CAPACITY: usize> {
    id: usize,
    queues: WorkerQueues<QUEUE_CAPACITY>,
    lifo_slot: LifoSlot,
    thread: OnceLock<thread::Thread>,
}

struct SchedulerScopeState {
    pending_tasks: AtomicUsize,
    failed_tasks: AtomicBool,
    wait_lock: Mutex<()>,
    wait_signal: Condvar,
}

struct ScopedTaskCompletion<'scope> {
    state: NonNull<SchedulerScopeState>,
    _state: PhantomData<&'scope SchedulerScopeState>,
}

struct SharedScopedTaskCompletion {
    state: Arc<SchedulerScopeState>,
}

// Safety: the pointer targets the stack-owned scope state in
// `ThreadScheduler::scope`. That function waits for every scoped job before the
// state is dropped, and `SchedulerScopeState` uses atomics plus a mutex/condvar
// for cross-thread synchronization.
unsafe impl Send for ScopedTaskCompletion<'_> {}

impl ThreadScheduler<256, 256> {
    /// Start a scheduler with one worker set for all work classes.
    pub fn new(worker_count: usize, thread_name_prefix: &str) -> ExecutorResult<Self> {
        Self::new_with_config(worker_count, thread_name_prefix)
    }
}

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT> {
    /// Start a scheduler with custom configurations.
    pub fn new_with_config(worker_count: usize, thread_name_prefix: &str) -> ExecutorResult<Self> {
        let worker_count = worker_count.max(1);
        let workers = (0..worker_count)
            .map(|id| Arc::new(WorkerState::new(id)))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let inner = Arc::new(SchedulerInner {
            workers,
            handles: Mutex::new(Vec::with_capacity(worker_count)),
            next_worker: CacheAligned::new(AtomicUsize::new(0)),
            pending_tasks: CacheAligned::new(AtomicUsize::new(0)),
            active_workers: CacheAligned::new(AtomicUsize::new(0)),
            completed_tasks: CacheAligned::new(AtomicU64::new(0)),
            failed_tasks: CacheAligned::new(AtomicU64::new(0)),
            shutdown: CacheAligned::new(AtomicBool::new(false)),
            join_waiters: CacheAligned::new(AtomicUsize::new(0)),
            wait_lock: Mutex::new(()),
            wait_signal: Condvar::new(),
            idle_workers: CacheAligned::new(AtomicU64::new(0)),
        });

        for worker_id in 0..worker_count {
            let worker_inner = Arc::clone(&inner);
            let thread_name = format!("{thread_name_prefix}-{worker_id}");
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || worker_loop::<QUEUE_CAPACITY, SPIN_LIMIT>(worker_inner, worker_id))
                .map_err(|_| ExecutorError::ThreadPoolCreationFailed)?;

            lock_mutex(&inner.handles).push(handle);
        }

        Ok(Self { inner })
    }

    /// Schedule a job for a compile-time work class.
    pub fn schedule<C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'static,
    {
        let job = ScheduledJob::new(task);
        self.schedule_job::<C>(priority, locality_hint, job)
    }

    /// Run a borrowing job scope on the scheduler and wait for all spawned jobs.
    ///
    /// This is the scheduler-equivalent of a scoped fan-out. It avoids per-task
    /// result storage when the caller only needs completion, while preserving the
    /// invariant that borrowed data cannot outlive the scope.
    pub fn scope<'scope, C, F>(
        &'scope self,
        priority: Priority,
        locality_hint: Option<usize>,
        body: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(&SchedulerScope<'scope, C, QUEUE_CAPACITY, SPIN_LIMIT>) -> ExecutorResult<()>,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShuttingDown);
        }

        let state = SchedulerScopeState::new();
        let scope = SchedulerScope {
            scheduler: self,
            state: NonNull::from(&state),
            priority,
            locality_hint,
            jobs: RefCell::new(Vec::new()),
            _state: PhantomData,
            _class: PhantomData,
        };

        let body_result = catch_unwind(AssertUnwindSafe(|| body(&scope)));
        let flush_result = scope.flush();
        state.wait();

        match body_result {
            Ok(Ok(())) if state.failed_tasks.load(Ordering::Acquire) => Err(
                ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked),
            ),
            Ok(Ok(())) => flush_result,
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Run an indexed scoped fan-out with worker-sized scheduler chunks.
    ///
    /// This path is for data-parallel work where the caller needs completion,
    /// not one task handle per logical item. It schedules at most one erased
    /// scheduler job per worker, while the item closure remains statically
    /// typed and shared by reference across chunks.
    pub fn for_each_indexed<C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        count: usize,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: Fn(usize) + Send + Sync,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShuttingDown);
        }

        if count == 0 {
            return Ok(());
        }

        let chunk_count = indexed_chunk_count(count, self.worker_count());
        let chunk_size = count.div_ceil(chunk_count);
        let caller_end = chunk_size.min(count);
        if chunk_count == 1 {
            return catch_unwind(AssertUnwindSafe(|| {
                for index in 0..caller_end {
                    task(index);
                }
            }))
            .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked));
        }

        let state = Arc::new(SchedulerScopeState::new());
        let task = &task;
        let mut schedule_result = Ok(());

        for chunk_index in 1..chunk_count {
            let start = chunk_index * chunk_size;
            let end = start.saturating_add(chunk_size).min(count);
            if start >= end {
                break;
            }

            state.register_task();
            let completion = SharedScopedTaskCompletion {
                state: Arc::clone(&state),
            };
            let scoped_job = move |_| {
                let completion = completion;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    for index in start..end {
                        task(index);
                    }
                }));

                if result.is_err() {
                    completion.mark_failed();
                }
            };

            if let Err(error) =
                self.schedule_scoped_job::<C, _>(priority, locality_hint, scoped_job)
            {
                state.complete_task();
                schedule_result = Err(error);
                break;
            }
        }

        let caller_result = if schedule_result.is_ok() {
            catch_unwind(AssertUnwindSafe(|| {
                for index in 0..caller_end {
                    task(index);
                }
            }))
            .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked))
        } else {
            Ok(())
        };

        state.wait();

        if state.failed_tasks.load(Ordering::Acquire) || caller_result.is_err() {
            Err(ExecutorError::SpawnFailed(
                moirai_core::error::TaskError::Panicked,
            ))
        } else {
            schedule_result
        }
    }

    /// Run an indexed map/reduce with one result slot per physical chunk.
    ///
    /// `identity` must be the neutral element for `reduce`. The scheduler
    /// computes local chunk reductions before combining them on the caller's
    /// thread, avoiding per-item atomic aggregation.
    pub fn map_reduce_indexed<C, T, Map, Reduce>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        count: usize,
        identity: T,
        map: Map,
        reduce: Reduce,
    ) -> ExecutorResult<T>
    where
        C: WorkClass,
        T: Send + Clone,
        Map: Fn(usize) -> T + Send + Sync,
        Reduce: Fn(T, T) -> T + Send + Sync,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShuttingDown);
        }

        if count == 0 {
            return Ok(identity);
        }

        let worker_count = self.worker_count().max(1);
        if count <= inline_reduction_limit::<T>(worker_count) {
            return inline_map_reduce(count, identity, map, reduce);
        }

        let chunk_count = indexed_reduce_chunk_count::<T>(count, worker_count);
        let chunk_size = count.div_ceil(chunk_count);
        let caller_end = chunk_size.min(count);
        if chunk_count == 1 {
            return inline_map_reduce(count, identity, map, reduce);
        }

        let state = Arc::new(SchedulerScopeState::new());
        let slots = Arc::new(ReduceSlots::new(chunk_count - 1));
        let map = &map;
        let reduce = &reduce;
        let mut schedule_result = Ok(());

        for chunk_index in 1..chunk_count {
            let start = chunk_index * chunk_size;
            let end = start.saturating_add(chunk_size).min(count);
            if start >= end {
                break;
            }

            state.register_task();
            let completion = SharedScopedTaskCompletion {
                state: Arc::clone(&state),
            };
            let slots = Arc::clone(&slots);
            let identity = identity.clone();
            let scoped_job = move |_| {
                let completion = completion;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let accumulator = map_reduce_range(start, end, identity, map, reduce);
                    slots.write(chunk_index - 1, accumulator);
                }));

                if result.is_err() {
                    completion.mark_failed();
                }
            };

            if let Err(error) =
                self.schedule_scoped_job::<C, _>(priority, locality_hint, scoped_job)
            {
                state.complete_task();
                schedule_result = Err(error);
                break;
            }
        }

        let caller_result = if schedule_result.is_ok() {
            catch_unwind(AssertUnwindSafe(|| {
                map_reduce_range(0, caller_end, identity.clone(), map, reduce)
            }))
            .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked))
        } else {
            Ok(identity.clone())
        };

        state.wait();

        if state.failed_tasks.load(Ordering::Acquire) {
            Err(ExecutorError::SpawnFailed(
                moirai_core::error::TaskError::Panicked,
            ))
        } else {
            schedule_result?;
            Ok(slots.reduce(caller_result?, reduce))
        }
    }

    fn schedule_job<C>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        job: ScheduledJob,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(ExecutorError::ShuttingDown);
        }

        let pending_before_submit = self.inner.pending_tasks.load(Ordering::Acquire);
        let active_before_submit = self.inner.active_workers.load(Ordering::Acquire);
        let worker_index = self.select_worker_for_state::<C>(
            priority,
            locality_hint,
            pending_before_submit,
            active_before_submit,
        );
        let previous_pending = self.inner.pending_tasks.fetch_add(1, Ordering::Release);

        let is_local = CURRENT_WORKER_ID.with(|cell| cell.get() == Some(worker_index));
        if is_local {
            if let Some(old_job) = self.inner.workers[worker_index].lifo_slot.push(job) {
                self.inner.workers[worker_index].queues.push(priority, old_job);
            }
        } else {
            self.inner.workers[worker_index].queues.push(priority, job);
        }

        // Try to wake up an idle worker via the lock-free wake lottery
        let mut woken = false;
        let mut idle = self.inner.idle_workers.load(Ordering::SeqCst);
        while idle != 0 {
            let worker_to_wake = idle.trailing_zeros() as usize;
            if worker_to_wake >= self.inner.workers.len() {
                break;
            }
            let mask = 1 << worker_to_wake;
            match self.inner.idle_workers.compare_exchange_weak(
                idle,
                idle & !mask,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    wake_worker(&self.inner.workers[worker_to_wake]);
                    woken = true;
                    break;
                }
                Err(actual) => {
                    idle = actual;
                }
            }
        }

        if !woken {
            if previous_pending == 0 {
                wake_worker(&self.inner.workers[worker_index]);
            } else {
                let worker_count = self.inner.workers.len();
                if previous_pending < worker_count {
                    let _ = wake_contended_workers::<BoundedContendedWake>(
                        &*self.inner,
                        worker_index,
                        previous_pending,
                    );
                }
            }
        }
        Ok(())
    }

    fn schedule_scoped_job<'scope, C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        scoped_job: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'scope,
    {
        // Safety: callers wait for their scope state counter to reach zero
        // before returning. Every scoped scheduler job owns a completion token
        // whose Drop decrements that counter on normal return, panic, or queued
        // drop. Scheduler shutdown drains queued work before workers exit.
        let job = unsafe { ScheduledJob::new_scoped(scoped_job) };
        self.schedule_job::<C>(priority, locality_hint, job)
    }

    /// Approximate number of queued jobs.
    pub fn pending_tasks(&self) -> usize {
        self.inner.pending_tasks.load(Ordering::Acquire)
    }

    /// Number of workers currently executing jobs.
    pub fn active_workers(&self) -> usize {
        self.inner.active_workers.load(Ordering::Acquire)
    }

    /// Returns true when queued or active work exists.
    pub fn has_work(&self) -> bool {
        !is_quiescent(&self.inner)
    }

    /// Wait until all queued and active work has completed without stopping workers.
    pub fn join(&self) -> ExecutorResult<()> {
        for _ in 0..JOIN_FAST_SPIN_ATTEMPTS {
            if is_quiescent(&self.inner) {
                return Ok(());
            }
            core::hint::spin_loop();
        }

        let mut guard = lock_mutex(&self.inner.wait_lock);
        self.inner.join_waiters.fetch_add(1, Ordering::AcqRel);
        while !is_quiescent(&self.inner) {
            guard = self
                .inner
                .wait_signal
                .wait(guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        self.inner.join_waiters.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }

    /// Number of worker threads.
    pub fn worker_count(&self) -> usize {
        self.inner.workers.len()
    }

    /// Capture scheduler metrics.
    pub fn metrics(&self) -> ScheduleMetrics {
        ScheduleMetrics {
            worker_count: self.worker_count(),
            pending_tasks: self.pending_tasks(),
            active_workers: self.active_workers(),
            completed_tasks: self.inner.completed_tasks.load(Ordering::Acquire),
            failed_tasks: self.inner.failed_tasks.load(Ordering::Acquire),
        }
    }

    /// Stop workers after queued work drains.
    pub fn shutdown(&self) {
        if !self.inner.shutdown.swap(true, Ordering::AcqRel) {
            wake_all_workers(&self.inner);
        }

        let mut handles = lock_mutex(&self.inner.handles);
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }

    #[cfg(test)]
    fn select_worker<C>(&self, priority: Priority, locality_hint: Option<usize>) -> usize
    where
        C: WorkClass,
    {
        self.select_worker_for_state::<C>(
            priority,
            locality_hint,
            self.inner.pending_tasks.load(Ordering::Acquire),
            self.inner.active_workers.load(Ordering::Acquire),
        )
    }

    fn select_worker_for_state<C>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        pending_tasks: usize,
        active_workers: usize,
    ) -> usize
    where
        C: WorkClass,
    {
        let worker_count = self.inner.workers.len();
        if let Some(hint) = locality_hint {
            return hint % worker_count;
        }

        if pending_tasks == 0 && active_workers <= 1 {
            return C::SERIAL_AFFINITY_OFFSET.wrapping_add(priority_weight(priority))
                % worker_count;
        }

        let ticket = self.inner.next_worker.fetch_add(1, Ordering::Relaxed);
        ticket
            .wrapping_add(C::AFFINITY_OFFSET)
            .wrapping_add(priority_weight(priority))
            % worker_count
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_select_worker_for_state<C>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        pending_tasks: usize,
        active_workers: usize,
    ) -> usize
    where
        C: WorkClass,
    {
        self.select_worker_for_state::<C>(priority, locality_hint, pending_tasks, active_workers)
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_pending_counter_pair(&self) -> usize {
        let previous = self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.pending_tasks.fetch_sub(1, Ordering::Release);
        previous
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_worker_unpark(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        wake_worker(&self.inner.workers[index]);
        index
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_priority_queue_push_pop(priority: Priority) -> usize {
        let queues = WorkerQueues::<QUEUE_CAPACITY>::new();
        queues.push(priority, ScheduledJob::new(|_| {}));
        queues
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_submission_queue_publication<C>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
    ) -> usize
    where
        C: WorkClass,
    {
        let pending_tasks = AtomicUsize::new(0);
        let active_workers = AtomicUsize::new(0);
        let pending_before_submit = pending_tasks.load(Ordering::Acquire);
        let active_before_submit = active_workers.load(Ordering::Acquire);
        let worker_index = self.select_worker_for_state::<C>(
            priority,
            locality_hint,
            pending_before_submit,
            active_before_submit,
        );
        let previous_pending = pending_tasks.fetch_add(1, Ordering::Release);
        let queues = WorkerQueues::<QUEUE_CAPACITY>::new();
        queues.push(priority, ScheduledJob::new(|_| {}));
        let completed = queues
            .pop_local()
            .map(|job| usize::from(job.execute(worker_index)))
            .unwrap_or(0);
        pending_tasks.fetch_sub(1, Ordering::Release);

        worker_index + previous_pending + completed
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_worker_execute_ready_job(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        execute_job(&self.inner, index, ScheduledJob::new(|_| {}));
        index
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_worker_local_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.workers[index]
            .queues
            .push(Priority::Normal, ScheduledJob::new(|_| {}));

        next_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_max_inline_job_construct_drop() -> usize {
        let words = [1usize; 14];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        drop(std::hint::black_box(job));
        words.len()
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_max_inline_job_construct_execute() -> usize {
        let words = [1usize; 14];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        usize::from(std::hint::black_box(job).execute(0))
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_oversized_job_construct_drop() -> usize {
        let words = [1usize; 32];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        drop(std::hint::black_box(job));
        words.len()
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_oversized_job_construct_execute() -> usize {
        let words = [1usize; 32];
        let job = ScheduledJob::new(move |_| {
            std::hint::black_box(words.iter().copied().sum::<usize>());
        });
        usize::from(std::hint::black_box(job).execute(0))
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_max_inline_queue_push_pop_execute() -> usize {
        let words = [1usize; 14];
        let queues = WorkerQueues::<QUEUE_CAPACITY>::new();
        queues.push(
            Priority::Normal,
            ScheduledJob::new(move |_| {
                std::hint::black_box(words.iter().copied().sum::<usize>());
            }),
        );

        queues
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_oversized_queue_push_pop_execute() -> usize {
        let words = [1usize; 32];
        let queues = WorkerQueues::<QUEUE_CAPACITY>::new();
        queues.push(
            Priority::Normal,
            ScheduledJob::new(move |_| {
                std::hint::black_box(words.iter().copied().sum::<usize>());
            }),
        );

        queues
            .pop_local()
            .map(|job| usize::from(job.execute(0)))
            .unwrap_or(0)
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_worker_local_max_inline_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        let words = [1usize; 14];
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.workers[index].queues.push(
            Priority::Normal,
            ScheduledJob::new(move |_| {
                std::hint::black_box(words.iter().copied().sum::<usize>());
            }),
        );

        next_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_worker_local_oversized_dequeue_execute(&self, worker_index: usize) -> usize {
        let index = worker_index % self.inner.workers.len();
        let words = [1usize; 32];
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        self.inner.workers[index].queues.push(
            Priority::Normal,
            ScheduledJob::new(move |_| {
                std::hint::black_box(words.iter().copied().sum::<usize>());
            }),
        );

        next_job(&self.inner, index)
            .map(|job| {
                execute_job(&self.inner, index, job);
                index + 1
            })
            .unwrap_or(0)
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_join_fast_spin_quiescent(&self) -> usize {
        for attempt in 0..JOIN_FAST_SPIN_ATTEMPTS {
            if is_quiescent(&self.inner) {
                return attempt + 1;
            }
            core::hint::spin_loop();
        }
        0
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_join_fast_spin_pending(&self) -> usize {
        self.inner.pending_tasks.fetch_add(1, Ordering::Release);
        let mut misses = 0usize;
        for _ in 0..JOIN_FAST_SPIN_ATTEMPTS {
            if !is_quiescent(&self.inner) {
                misses = misses.wrapping_add(1);
            }
            core::hint::spin_loop();
        }
        self.inner.pending_tasks.fetch_sub(1, Ordering::Release);
        misses
    }

    #[cfg(feature = "scheduler-diagnostics")]
    pub fn diagnostic_wake_decision<P>(&self, worker_index: usize) -> usize
    where
        P: DiagnosticWakeDecision,
    {
        let worker_count = self.inner.workers.len();
        let index = worker_index % worker_count;
        diagnostic_publish_work_available(&self.inner, index, P::previous_pending(worker_count))
    }
}

impl<
    'scope,
    C,
    const QUEUE_CAPACITY: usize,
    const SPIN_LIMIT: usize,
> SchedulerScope<'scope, C, QUEUE_CAPACITY, SPIN_LIMIT>
where
    C: WorkClass,
{
    /// Spawn a job into this scope.
    ///
    /// The job may borrow values that outlive the scope call. Scoped jobs are
    /// coalesced into worker-sized scheduler batches and complete before
    /// `ThreadScheduler::scope` returns. Jobs are not guaranteed to start while
    /// the scope body is still registering work.
    pub fn spawn<F>(&self, task: F) -> ExecutorResult<()>
    where
        F: FnOnce(usize) + Send + 'scope,
    {
        self.state().register_task();
        let completion = ScopedTaskCompletion {
            state: self.state,
            _state: PhantomData,
        };
        let scoped_task = move |worker_id| {
            let completion = completion;
            let result = catch_unwind(AssertUnwindSafe(|| task(worker_id)));
            if result.is_err() {
                completion.mark_failed();
            }
        };

        // Safety: `ThreadScheduler::scope` waits for every scheduled scoped
        // job and drops unscheduled buffered jobs before borrowed scope data
        // can expire.
        let job = unsafe { ScheduledJob::new_scoped(scoped_task) };
        self.jobs.borrow_mut().push(job);
        Ok(())
    }

    fn flush(&self) -> ExecutorResult<()> {
        let jobs = mem::take(&mut *self.jobs.borrow_mut());
        if jobs.is_empty() {
            return Ok(());
        }

        if jobs.len() == 1 {
            let job = jobs
                .into_iter()
                .next()
                .expect("single scoped job must exist");
            return self.schedule_single(job);
        }

        let worker_count = self.scheduler.worker_count();
        let chunk_count = jobs.len().min(worker_count.max(1));
        let chunk_size = jobs.len().div_ceil(chunk_count);
        let mut pending_jobs = jobs.into_iter();

        for _ in 0..chunk_count {
            let mut chunk = Vec::with_capacity(chunk_size);
            for _ in 0..chunk_size {
                if let Some(job) = pending_jobs.next() {
                    chunk.push(job);
                }
            }

            if chunk.is_empty() {
                break;
            }

            self.schedule_chunk(chunk)?;
        }

        Ok(())
    }

    fn schedule_single(&self, job: ScheduledJob) -> ExecutorResult<()> {
        self.scheduler
            .schedule_job::<C>(self.priority, self.locality_hint, job)?;
        Ok(())
    }

    fn schedule_chunk(&self, jobs: Vec<ScheduledJob>) -> ExecutorResult<()> {
        let scoped_job = move |worker_id| {
            for job in jobs {
                let _ = job.execute(worker_id);
            }
        };

        self.scheduler.schedule_scoped_job::<C, _>(
            self.priority,
            self.locality_hint,
            scoped_job,
        )?;
        Ok(())
    }

    fn state(&self) -> &SchedulerScopeState {
        // Safety: `ThreadScheduler::scope` creates this pointer from a local
        // state value and waits for every scheduled scoped job before returning.
        unsafe { self.state.as_ref() }
    }
}

impl SchedulerScopeState {
    fn new() -> Self {
        Self {
            pending_tasks: AtomicUsize::new(0),
            failed_tasks: AtomicBool::new(false),
            wait_lock: Mutex::new(()),
            wait_signal: Condvar::new(),
        }
    }

    fn register_task(&self) {
        self.pending_tasks.fetch_add(1, Ordering::AcqRel);
    }

    fn complete_task(&self) {
        if self.pending_tasks.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _guard = lock_mutex(&self.wait_lock);
            self.wait_signal.notify_all();
        }
    }

    fn wait(&self) {
        let mut guard = lock_mutex(&self.wait_lock);
        while self.pending_tasks.load(Ordering::Acquire) != 0 {
            guard = self
                .wait_signal
                .wait(guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn mark_failed(&self) {
        self.failed_tasks.store(true, Ordering::Release);
    }
}

impl ScopedTaskCompletion<'_> {
    fn mark_failed(&self) {
        self.state().mark_failed();
    }

    fn state(&self) -> &SchedulerScopeState {
        // Safety: this completion token is created only by `SchedulerScope`,
        // whose caller waits for all scoped jobs before the state is dropped.
        unsafe { self.state.as_ref() }
    }
}

impl Drop for ScopedTaskCompletion<'_> {
    fn drop(&mut self) {
        self.state().complete_task();
    }
}

impl SharedScopedTaskCompletion {
    fn mark_failed(&self) {
        self.state.mark_failed();
    }
}

impl Drop for SharedScopedTaskCompletion {
    fn drop(&mut self) {
        self.state.complete_task();
    }
}

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> Drop
    for ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            self.shutdown();
        }
    }
}

impl<const QUEUE_CAPACITY: usize> WorkerState<QUEUE_CAPACITY> {
    fn new(id: usize) -> Self {
        Self {
            id,
            queues: WorkerQueues::new(),
            lifo_slot: LifoSlot::new(),
            thread: OnceLock::new(),
        }
    }
}

fn worker_loop<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>(
    inner: Arc<SchedulerInner<QUEUE_CAPACITY>>,
    worker_id: usize,
) {
    CURRENT_WORKER_ID.with(|cell| cell.set(Some(worker_id)));
    let _ = inner.workers[worker_id].thread.set(thread::current());

    loop {
        if let Some(job) = next_job(&inner, worker_id) {
            execute_job(&inner, worker_id, job);
            continue;
        }

        if should_stop(&inner) {
            break;
        }

        if spin_for_work::<QUEUE_CAPACITY, SPIN_LIMIT>(&inner, worker_id) {
            continue;
        }

        wait_for_work(&inner, worker_id);
    }
}

fn next_job<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
    worker_id: usize,
) -> Option<ScheduledJob> {
    let local = &inner.workers[worker_id];
    local
        .lifo_slot
        .pop()
        .or_else(|| local.queues.pop_local())
        .or_else(|| steal_job(inner, worker_id))
}

fn steal_job<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
    worker_id: usize,
) -> Option<ScheduledJob> {
    let worker_count = inner.workers.len();
    let local = &inner.workers[worker_id];
    for offset in 1..worker_count {
        let victim_index = (worker_id + offset) % worker_count;
        let victim = &inner.workers[victim_index];
        if let Some(job) = local.queues.steal_batch(&victim.queues) {
            return Some(job);
        }
        if let Some(job) = victim.lifo_slot.steal() {
            return Some(job);
        }
    }

    None
}

fn execute_job<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
    worker_id: usize,
    job: ScheduledJob,
) {
    inner.active_workers.fetch_add(1, Ordering::Release);
    inner.pending_tasks.fetch_sub(1, Ordering::Release);

    if job.execute(inner.workers[worker_id].id) {
        inner.completed_tasks.fetch_add(1, Ordering::Relaxed);
    } else {
        inner.failed_tasks.fetch_add(1, Ordering::Relaxed);
    }

    if inner.active_workers.fetch_sub(1, Ordering::AcqRel) == 1 {
        notify_quiescent(inner);
    }
}

fn should_stop<const QUEUE_CAPACITY: usize>(inner: &SchedulerInner<QUEUE_CAPACITY>) -> bool {
    inner.shutdown.load(Ordering::Acquire) && inner.pending_tasks.load(Ordering::Acquire) == 0
}

fn spin_for_work<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
    worker_id: usize,
) -> bool {
    for attempt in 0..SPIN_LIMIT {
        core::hint::spin_loop();
        let local = &inner.workers[worker_id];
        if !local.queues.is_empty()
            || local.lifo_slot.state.load(Ordering::Relaxed) == 2
            || should_stop(inner)
        {
            return true;
        }

        // Periodically check if other workers have stealable tasks to avoid parking
        if attempt % 32 == 0 && (has_stealable_work(inner, worker_id) || should_stop(inner)) {
            return true;
        }
    }

    false
}

fn has_stealable_work<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
    worker_id: usize,
) -> bool {
    let worker_count = inner.workers.len();
    for offset in 1..worker_count {
        let victim_index = (worker_id + offset) % worker_count;
        let victim = &inner.workers[victim_index];
        if !victim.queues.is_empty() || victim.lifo_slot.state.load(Ordering::Relaxed) == 2 {
            return true;
        }
    }
    false
}

fn wait_for_work<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
    worker_id: usize,
) {
    if worker_id < 64 {
        let mask = 1 << worker_id;
        inner.idle_workers.fetch_or(mask, Ordering::SeqCst);
        while inner.pending_tasks.load(Ordering::SeqCst) == 0
            && !inner.shutdown.load(Ordering::SeqCst)
        {
            if let Some(reactor) = IoReactor::get_active() {
                let _ = reactor.run_iteration(Some(Duration::from_millis(1)));
                if inner.pending_tasks.load(Ordering::SeqCst) > 0 || inner.shutdown.load(Ordering::SeqCst) {
                    break;
                }
            } else {
                thread::park();
            }
        }
        inner.idle_workers.fetch_and(!mask, Ordering::SeqCst);
    } else {
        while inner.pending_tasks.load(Ordering::Acquire) == 0
            && !inner.shutdown.load(Ordering::Acquire)
        {
            if let Some(reactor) = IoReactor::get_active() {
                let _ = reactor.run_iteration(Some(Duration::from_millis(1)));
                if inner.pending_tasks.load(Ordering::Acquire) > 0 || inner.shutdown.load(Ordering::Acquire) {
                    break;
                }
            } else {
                thread::park();
            }
        }
    }
}

fn wake_worker<const QUEUE_CAPACITY: usize>(worker: &WorkerState<QUEUE_CAPACITY>) {
    if let Some(thread) = worker.thread.get() {
        thread.unpark();
    }
}

fn wake_all_workers<const QUEUE_CAPACITY: usize>(inner: &SchedulerInner<QUEUE_CAPACITY>) {
    for worker in inner.workers.iter() {
        wake_worker(worker);
    }
}

trait ContendedWakable {
    #[allow(dead_code)]
    fn worker_count(&self) -> usize;
    #[allow(dead_code)]
    fn wake_worker(&self, worker_index: usize);
    fn wake_contended<P>(&self, worker_index: usize, previous_pending: usize) -> usize
    where
        P: ContendedWakePolicy;
}

impl<const QUEUE_CAPACITY: usize> ContendedWakable for SchedulerInner<QUEUE_CAPACITY> {
    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    fn wake_worker(&self, worker_index: usize) {
        wake_worker(&self.workers[worker_index]);
    }

    fn wake_contended<P>(&self, worker_index: usize, previous_pending: usize) -> usize
    where
        P: ContendedWakePolicy,
    {
        let worker_count = self.workers.len();
        wake_worker(&self.workers[worker_index]);

        if P::WAKE_LIMIT < 2 || worker_count < 2 {
            return 1;
        }

        let peer_index = worker_index.wrapping_add(previous_pending) % worker_count;
        wake_worker(&self.workers[peer_index]);
        2
    }
}

#[cold]
#[inline(never)]
fn wake_contended_workers<P>(
    inner: &impl ContendedWakable,
    worker_index: usize,
    previous_pending: usize,
) -> usize
where
    P: ContendedWakePolicy,
{
    inner.wake_contended::<P>(worker_index, previous_pending)
}

#[cfg(feature = "scheduler-diagnostics")]
#[inline]
fn diagnostic_publish_work_available(
    inner: &impl ContendedWakable,
    worker_index: usize,
    previous_pending: usize,
) -> usize {
    let worker_count = inner.worker_count();
    if previous_pending == 0 {
        inner.wake_worker(worker_index);
        1
    } else if previous_pending < worker_count {
        wake_contended_workers::<BoundedContendedWake>(inner, worker_index, previous_pending)
    } else {
        0
    }
}

fn notify_quiescent<const QUEUE_CAPACITY: usize>(inner: &SchedulerInner<QUEUE_CAPACITY>) {
    if inner.join_waiters.load(Ordering::Acquire) != 0 && is_quiescent(inner) {
        let _guard = lock_mutex(&inner.wait_lock);
        if is_quiescent(inner) {
            inner.wait_signal.notify_all();
        }
    }
}

fn is_quiescent<const QUEUE_CAPACITY: usize>(inner: &SchedulerInner<QUEUE_CAPACITY>) -> bool {
    inner.pending_tasks.load(Ordering::Acquire) == 0
        && inner.active_workers.load(Ordering::Acquire) == 0
}

fn priority_weight(priority: Priority) -> usize {
    match priority {
        Priority::Low => 0,
        Priority::Normal => 1,
        Priority::High => 2,
        Priority::Critical => 3,
    }
}

fn inline_map_reduce<T, Map, Reduce>(
    count: usize,
    identity: T,
    map: Map,
    reduce: Reduce,
) -> ExecutorResult<T>
where
    Map: Fn(usize) -> T,
    Reduce: Fn(T, T) -> T,
{
    catch_unwind(AssertUnwindSafe(|| {
        let mut accumulator = identity;
        for index in 0..count {
            accumulator = reduce(accumulator, map(index));
        }
        accumulator
    }))
    .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked))
}

fn map_reduce_range<T, Map, Reduce>(
    start: usize,
    end: usize,
    identity: T,
    map: &Map,
    reduce: &Reduce,
) -> T
where
    Map: Fn(usize) -> T,
    Reduce: Fn(T, T) -> T,
{
    let mut accumulator = identity;
    for index in start..end {
        accumulator = reduce(accumulator, map(index));
    }
    accumulator
}

fn indexed_chunk_count(count: usize, worker_count: usize) -> usize {
    count.min(worker_count.max(1).saturating_add(1))
}

fn indexed_reduce_chunk_count<T>(count: usize, worker_count: usize) -> usize {
    let worker_count = worker_count.max(1);
    let max_chunks = count.min(worker_count.saturating_add(1));
    let scheduled_chunk_floor = inline_reduction_limit::<T>(worker_count)
        .saturating_mul(2)
        .max(1);

    max_chunks.min(count.div_ceil(scheduled_chunk_floor).max(1))
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    };

    use super::{indexed_reduce_chunk_count, ThreadScheduler};
    use crate::schedule::{AsyncTask, BlockingTask, SyncTask};
    use moirai_core::{
        error::{ExecutorError, TaskError},
        Priority,
    };

    #[test]
    fn scheduler_runs_all_work_classes_on_one_worker_set() {
        let scheduler = ThreadScheduler::new(2, "test-scheduler").unwrap();
        let completed = Arc::new(AtomicUsize::new(0));
        let (sender, receiver) = mpsc::channel();

        {
            let completed = Arc::clone(&completed);
            let sender = sender.clone();
            scheduler
                .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                    completed.fetch_add(1, Ordering::AcqRel);
                    sender.send(()).unwrap();
                })
                .unwrap();
        }

        {
            let completed = Arc::clone(&completed);
            let sender = sender.clone();
            scheduler
                .schedule::<AsyncTask, _>(Priority::Normal, None, move |_| {
                    completed.fetch_add(1, Ordering::AcqRel);
                    sender.send(()).unwrap();
                })
                .unwrap();
        }

        {
            let completed = Arc::clone(&completed);
            scheduler
                .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                    completed.fetch_add(1, Ordering::AcqRel);
                    sender.send(()).unwrap();
                })
                .unwrap();
        }

        for _ in 0..3 {
            receiver.recv().unwrap();
        }

        scheduler.shutdown();
        let metrics = scheduler.metrics();

        assert_eq!(completed.load(Ordering::Acquire), 3);
        assert_eq!(metrics.worker_count, 2);
        assert_eq!(metrics.pending_tasks, 0);
        assert_eq!(metrics.completed_tasks, 3);
        assert_eq!(metrics.failed_tasks, 0);
    }

    #[test]
    fn quiescent_single_task_selection_reuses_work_class_worker() {
        let scheduler = ThreadScheduler::new(4, "test-quiescent-route").unwrap();
        let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
        let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

        scheduler.shutdown();

        assert_eq!(first, second);
        assert_eq!(first, 3);
    }

    #[test]
    fn serial_handoff_selection_reuses_work_class_worker() {
        let scheduler = ThreadScheduler::new(4, "test-serial-handoff-route").unwrap();
        scheduler.inner.active_workers.store(1, Ordering::Release);

        let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
        let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

        scheduler.inner.active_workers.store(0, Ordering::Release);
        scheduler.shutdown();

        assert_eq!(first, second);
        assert_eq!(first, 3);
    }

    #[test]
    fn queued_parallel_selection_rotates_workers() {
        let scheduler = ThreadScheduler::new(4, "test-parallel-route").unwrap();
        scheduler.inner.pending_tasks.store(1, Ordering::Release);

        let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
        let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

        scheduler.inner.pending_tasks.store(0, Ordering::Release);
        scheduler.shutdown();

        assert_ne!(first, second);
    }

    #[test]
    fn scheduler_scope_runs_borrowing_jobs_before_return() {
        let scheduler = ThreadScheduler::new(2, "test-scope").unwrap();
        let sum = AtomicUsize::new(0);

        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
                for value in 1..=16 {
                    let sum = &sum;
                    scope.spawn(move |_| {
                        sum.fetch_add(value, Ordering::Relaxed);
                    })?;
                }
                Ok(())
            })
            .unwrap();

        scheduler.shutdown();
        assert_eq!(sum.load(Ordering::Relaxed), 136);
    }

    #[test]
    fn scheduler_scope_reports_panicked_job() {
        let scheduler = ThreadScheduler::new(1, "test-scope-panic").unwrap();
        let completed = AtomicUsize::new(0);

        let result = scheduler.scope::<SyncTask, _>(Priority::Normal, None, |scope| {
            scope.spawn(|_| panic!("scoped job panic"))?;
            let completed = &completed;
            scope.spawn(move |_| {
                completed.fetch_add(1, Ordering::Relaxed);
            })?;
            Ok(())
        });

        scheduler.shutdown();
        assert_eq!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));
        assert_eq!(completed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn scheduler_join_waits_for_queued_and_active_work() {
        let scheduler = ThreadScheduler::new(2, "test-join").unwrap();
        let completed = Arc::new(AtomicUsize::new(0));

        for _ in 0..8 {
            let completed = Arc::clone(&completed);
            scheduler
                .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                    completed.fetch_add(1, Ordering::AcqRel);
                })
                .unwrap();
        }

        assert!(scheduler.has_work());
        scheduler.join().unwrap();
        let metrics = scheduler.metrics();

        scheduler.shutdown();
        assert_eq!(completed.load(Ordering::Acquire), 8);
        assert_eq!(metrics.pending_tasks, 0);
        assert_eq!(metrics.active_workers, 0);
        assert_eq!(metrics.completed_tasks, 8);
        assert!(!scheduler.has_work());
    }

    #[test]
    fn scheduler_join_waits_for_work_submitted_while_active() {
        let scheduler = ThreadScheduler::new(2, "test-join-transitive").unwrap();
        let completed = Arc::new(AtomicUsize::new(0));
        let (started_sender, started_receiver) = mpsc::channel();
        let (scheduled_sender, scheduled_receiver) = mpsc::channel();

        {
            let completed = Arc::clone(&completed);
            scheduler
                .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                    started_sender.send(()).unwrap();
                    scheduled_receiver.recv().unwrap();
                    completed.fetch_add(1, Ordering::AcqRel);
                })
                .unwrap();
        }

        started_receiver.recv().unwrap();
        std::thread::scope(|scope| {
            let completed = Arc::clone(&completed);
            let scheduler_ref = &scheduler;
            scope.spawn(move || {
                scheduler_ref
                    .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                        completed.fetch_add(1, Ordering::AcqRel);
                    })
                    .unwrap();
                scheduled_sender.send(()).unwrap();
            });

            scheduler.join().unwrap();
        });

        let metrics = scheduler.metrics();
        scheduler.shutdown();

        assert_eq!(completed.load(Ordering::Acquire), 2);
        assert_eq!(metrics.pending_tasks, 0);
        assert_eq!(metrics.active_workers, 0);
        assert_eq!(metrics.completed_tasks, 2);
    }

    #[test]
    fn indexed_fan_out_runs_all_items() {
        let scheduler = ThreadScheduler::new(2, "test-indexed").unwrap();
        let sum = AtomicUsize::new(0);

        scheduler
            .for_each_indexed::<BlockingTask, _>(Priority::Normal, None, 32, |index| {
                sum.fetch_add(index + 1, Ordering::Relaxed);
            })
            .unwrap();

        scheduler.shutdown();
        assert_eq!(sum.load(Ordering::Relaxed), 528);
    }

    #[test]
    fn indexed_map_reduce_returns_reduced_value() {
        let scheduler = ThreadScheduler::new(2, "test-indexed-reduce").unwrap();

        let sum = scheduler
            .map_reduce_indexed::<BlockingTask, _, _, _>(
                Priority::Normal,
                None,
                32,
                0usize,
                |index| index + 1,
                usize::wrapping_add,
            )
            .unwrap();

        scheduler.shutdown();
        assert_eq!(sum, 528);
    }

    #[test]
    fn indexed_map_reduce_small_count_runs_inline() {
        let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-inline").unwrap();

        let sum = scheduler
            .map_reduce_indexed::<BlockingTask, _, _, _>(
                Priority::Normal,
                None,
                32,
                0usize,
                |index| index + 1,
                usize::wrapping_add,
            )
            .unwrap();
        let metrics = scheduler.metrics();

        scheduler.shutdown();
        assert_eq!(sum, 528);
        assert_eq!(metrics.completed_tasks, 0);
    }

    #[test]
    fn indexed_map_reduce_inline_reports_panicked_mapper() {
        let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-inline-panic").unwrap();

        let result = scheduler.map_reduce_indexed::<BlockingTask, _, _, _>(
            Priority::Normal,
            None,
            4,
            0usize,
            |index| {
                if index == 2 {
                    panic!("inline map panic");
                }
                index + 1
            },
            usize::wrapping_add,
        );
        let metrics = scheduler.metrics();

        scheduler.shutdown();
        assert_eq!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));
        assert_eq!(metrics.completed_tasks, 0);
    }

    #[test]
    fn indexed_map_reduce_above_inline_limit_uses_scheduler_chunks() {
        let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-parallel").unwrap();

        let sum = scheduler
            .map_reduce_indexed::<BlockingTask, _, _, _>(
                Priority::Normal,
                None,
                64,
                0usize,
                |index| index + 1,
                usize::wrapping_add,
            )
            .unwrap();
        let metrics = scheduler.metrics();

        scheduler.shutdown();
        assert_eq!(sum, 2080);
        assert_eq!(
            metrics.completed_tasks,
            indexed_reduce_chunk_count::<usize>(64, 2).saturating_sub(1) as u64
        );
    }

    #[test]
    fn indexed_reduce_chunk_count_amortizes_scheduled_work() {
        assert_eq!(indexed_reduce_chunk_count::<usize>(64, 4), 1);
        assert_eq!(indexed_reduce_chunk_count::<usize>(256, 4), 2);
        assert_eq!(indexed_reduce_chunk_count::<usize>(1024, 4), 5);
    }

    #[test]
    fn scheduler_scope_completes_registered_jobs_before_body_error_returns() {
        let scheduler = ThreadScheduler::new(2, "test-scope-body-error").unwrap();
        let completed = AtomicUsize::new(0);

        let result = scheduler.scope::<SyncTask, _>(Priority::Normal, None, |scope| {
            for _ in 0..8 {
                let completed = &completed;
                scope.spawn(move |_| {
                    completed.fetch_add(1, Ordering::Relaxed);
                })?;
            }

            Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation))
        });

        scheduler.shutdown();
        assert_eq!(
            result,
            Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation))
        );
        assert_eq!(completed.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn scheduler_scope_completes_registered_jobs_before_resuming_body_panic() {
        let scheduler = ThreadScheduler::new(2, "test-scope-body-panic").unwrap();
        let completed = AtomicUsize::new(0);

        let result = catch_unwind(AssertUnwindSafe(|| {
            scheduler
                .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
                    for _ in 0..8 {
                        let completed = &completed;
                        scope.spawn(move |_| {
                            completed.fetch_add(1, Ordering::Relaxed);
                        })?;
                    }

                    panic!("scope body panic");
                })
                .unwrap();
        }));

        scheduler.shutdown();
        assert!(result.is_err());
        assert_eq!(completed.load(Ordering::Relaxed), 8);
    }
}
