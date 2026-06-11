//! Internal types for the unified thread scheduler runtime.

use std::{
    cell::RefCell,
    marker::PhantomData,
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
    thread::{self, JoinHandle},
};

use moirai_core::Priority;

use moirai_utils::cache::CacheAligned;

use super::super::{class::WorkClass, job::ScheduledJob, queue::WorkerQueues};

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
pub struct ThreadScheduler<const QUEUE_CAPACITY: usize = 256, const SPIN_LIMIT: usize = 256> {
    pub(super) inner: Arc<SchedulerInner<QUEUE_CAPACITY>>,
}

pub(super) mod contended_wake {
    pub trait Sealed {}
}

pub(super) trait ContendedWakePolicy:
    contended_wake::Sealed + Send + Sync + 'static
{
    const WAKE_LIMIT: usize;
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BoundedContendedWake;

impl contended_wake::Sealed for BoundedContendedWake {}

impl ContendedWakePolicy for BoundedContendedWake {
    const WAKE_LIMIT: usize = 2;
}

#[cfg(feature = "scheduler-diagnostics")]
pub(super) mod diagnostic_wake {
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
    pub(super) scheduler: &'scope ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>,
    pub(super) state: NonNull<SchedulerScopeState>,
    pub(super) priority: Priority,
    pub(super) locality_hint: Option<usize>,
    pub(super) jobs: RefCell<Vec<ScheduledJob>>,
    pub(super) _state: PhantomData<&'scope SchedulerScopeState>,
    pub(super) _class: PhantomData<C>,
}

pub(super) struct SchedulerInner<const QUEUE_CAPACITY: usize> {
    pub(super) workers: Box<[Arc<WorkerState<QUEUE_CAPACITY>>]>,
    pub(super) handles: Mutex<Vec<JoinHandle<()>>>,
    pub(super) next_worker: CacheAligned<AtomicUsize>,
    pub(super) pending_tasks: CacheAligned<AtomicUsize>,
    pub(super) active_workers: CacheAligned<AtomicUsize>,
    pub(super) completed_tasks: CacheAligned<AtomicU64>,
    pub(super) failed_tasks: CacheAligned<AtomicU64>,
    pub(super) shutdown: CacheAligned<AtomicBool>,
    pub(super) join_waiters: CacheAligned<AtomicUsize>,
    pub(super) wait_lock: Mutex<()>,
    pub(super) wait_signal: Condvar,
    pub(super) idle_workers: CacheAligned<AtomicU64>,
}

pub(super) struct LifoSlot {
    pub(super) state: std::sync::atomic::AtomicU8,
    pub(super) job: std::cell::UnsafeCell<std::mem::MaybeUninit<ScheduledJob>>,
}

unsafe impl Sync for LifoSlot {}

impl LifoSlot {
    pub(super) fn new() -> Self {
        Self {
            state: std::sync::atomic::AtomicU8::new(0),
            job: std::cell::UnsafeCell::new(std::mem::MaybeUninit::uninit()),
        }
    }

    pub(super) fn push(&self, job: ScheduledJob) -> Option<ScheduledJob> {
        let current = self.state.load(Ordering::Relaxed);
        if current == 0 {
            if self
                .state
                .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                unsafe {
                    *self.job.get() = std::mem::MaybeUninit::new(job);
                }
                self.state.store(2, Ordering::Release);
                return None;
            }
        } else if current == 2
            && self
                .state
                .compare_exchange(2, 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            let old_job = unsafe { std::ptr::read((*self.job.get()).as_ptr()) };
            unsafe {
                *self.job.get() = std::mem::MaybeUninit::new(job);
            }
            self.state.store(2, Ordering::Release);
            return Some(old_job);
        }
        Some(job)
    }

    pub(super) fn pop(&self) -> Option<ScheduledJob> {
        if self.state.load(Ordering::Relaxed) == 2 {
            if self
                .state
                .compare_exchange(2, 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                let job = unsafe { std::ptr::read((*self.job.get()).as_ptr()) };
                self.state.store(0, Ordering::Release);
                Some(job)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub(super) fn steal(&self) -> Option<ScheduledJob> {
        if self.state.load(Ordering::Relaxed) == 2 {
            if self
                .state
                .compare_exchange(2, 3, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                let job = unsafe { std::ptr::read((*self.job.get()).as_ptr()) };
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

#[cfg(nightly_tls_active)]
#[thread_local]
pub(super) static mut CURRENT_WORKER_ID_NIGHTLY: Option<usize> = None;

#[cfg(not(nightly_tls_active))]
thread_local! {
    pub(super) static CURRENT_WORKER_ID: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[inline(always)]
pub(super) fn get_current_worker_id() -> Option<usize> {
    #[cfg(nightly_tls_active)]
    unsafe {
        CURRENT_WORKER_ID_NIGHTLY
    }
    #[cfg(not(nightly_tls_active))]
    CURRENT_WORKER_ID.with(|cell| cell.get())
}

#[inline(always)]
pub(super) fn set_current_worker_id(id: Option<usize>) {
    #[cfg(nightly_tls_active)]
    unsafe {
        CURRENT_WORKER_ID_NIGHTLY = id;
    }
    #[cfg(not(nightly_tls_active))]
    CURRENT_WORKER_ID.with(|cell| cell.set(id));
}

#[repr(align(64))]
pub(super) struct WorkerState<const QUEUE_CAPACITY: usize> {
    pub(super) id: usize,
    pub(super) queues: WorkerQueues<QUEUE_CAPACITY>,
    pub(super) lifo_slot: LifoSlot,
    pub(super) thread: OnceLock<thread::Thread>,
}

impl<const QUEUE_CAPACITY: usize> WorkerState<QUEUE_CAPACITY> {
    pub(super) fn new(id: usize) -> Self {
        Self {
            id,
            queues: WorkerQueues::new(),
            lifo_slot: LifoSlot::new(),
            thread: OnceLock::new(),
        }
    }
}

pub(super) struct SchedulerScopeState {
    pub(super) pending_tasks: AtomicUsize,
    pub(super) failed_tasks: AtomicBool,
    pub(super) wait_lock: Mutex<()>,
    pub(super) wait_signal: Condvar,
}

impl SchedulerScopeState {
    pub(super) fn new() -> Self {
        Self {
            pending_tasks: AtomicUsize::new(0),
            failed_tasks: AtomicBool::new(false),
            wait_lock: Mutex::new(()),
            wait_signal: Condvar::new(),
        }
    }

    pub(super) fn register_task(&self) {
        self.pending_tasks.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn complete_task(&self) {
        if self.pending_tasks.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _guard = super::worker::lock_mutex(&self.wait_lock);
            self.wait_signal.notify_all();
        }
    }

    pub(super) fn wait(&self) {
        let mut guard = super::worker::lock_mutex(&self.wait_lock);
        while self.pending_tasks.load(Ordering::Acquire) != 0 {
            guard = self
                .wait_signal
                .wait(guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    pub(super) fn mark_failed(&self) {
        self.failed_tasks.store(true, Ordering::Release);
    }
}

pub(super) struct ScopedTaskCompletion<'scope> {
    pub(super) state: NonNull<SchedulerScopeState>,
    pub(super) _state: PhantomData<&'scope SchedulerScopeState>,
}

pub(super) struct SharedScopedTaskCompletion {
    pub(super) state: Arc<SchedulerScopeState>,
}

// Safety: the pointer targets the stack-owned scope state in
// `ThreadScheduler::scope`. That function waits for every scoped job before the
// state is dropped, and `SchedulerScopeState` uses atomics plus a mutex/condvar
// for cross-thread synchronization.
unsafe impl Send for ScopedTaskCompletion<'_> {}

impl ScopedTaskCompletion<'_> {
    pub(super) fn mark_failed(&self) {
        self.state().mark_failed();
    }

    pub(super) fn state(&self) -> &SchedulerScopeState {
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
    pub(super) fn mark_failed(&self) {
        self.state.mark_failed();
    }
}

impl Drop for SharedScopedTaskCompletion {
    fn drop(&mut self) {
        self.state.complete_task();
    }
}
