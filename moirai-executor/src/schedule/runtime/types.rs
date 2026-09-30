//! Internal types for the unified thread scheduler runtime.

use std::{
    any::Any,
    cell::RefCell,
    marker::PhantomData,
    ptr::NonNull,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};

use moirai_core::{Priority, error::PlacementFailure};

use moirai_utils::cache::{CacheAligned, CachePad};

use super::super::{class::WorkClass, job::ScheduledJob, queue::WorkerQueues};
use super::blocking::BlockingLane;
use super::scope_state::SchedulerScopeState;

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
///
/// `BLOCKING_QUEUE_CAPACITY` is the hard per-queue admission bound of the lazy
/// blocking lane. Resizable compute-local queues use the independent runtime
/// initial-capacity policy from [`ExecutorConfig`](moirai_core::ExecutorConfig).
///
/// `SPIN_LIMIT` is how many times an idle worker re-checks for work (with
/// `spin_loop`) before parking. The default 8192 (~60 µs on this class of x86)
/// catches work arriving in a short burst while still parking quickly to avoid
/// idle-CPU waste; a parked worker is then woken in ~8 µs. Sustained throughput
/// is independent of this value (the spin never engages while work is
/// available), so latency-critical deployments can raise it for sub-µs wake
/// latency at the cost of more pre-park busy-spin.
///
/// Clones share one worker pool. Dropping the final external handle drains the
/// pool and releases its workers; an explicit [`shutdown`](Self::shutdown) may
/// still stop the shared pool earlier.
pub struct ThreadScheduler<
    const BLOCKING_QUEUE_CAPACITY: usize = 256,
    const SPIN_LIMIT: usize = 8192,
> {
    pub(super) inner: Arc<SchedulerInner<BLOCKING_QUEUE_CAPACITY>>,
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

/// Sealed wake-decision policy used by scheduler diagnostics.
#[cfg(feature = "scheduler-diagnostics")]
pub trait DiagnosticWakeDecision: diagnostic_wake::Sealed + Send + Sync + 'static {
    /// Return the synthetic pending count used by the diagnostic probe.
    fn previous_pending(worker_count: usize) -> usize;
}

/// Diagnostic policy representing an empty queue.
#[cfg(feature = "scheduler-diagnostics")]
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyWakeDecision;

/// Diagnostic policy representing a contended queue.
#[cfg(feature = "scheduler-diagnostics")]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContendedWakeDecision;

/// Diagnostic policy representing a saturated queue.
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

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> Clone
    for ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn clone(&self) -> Self {
        let inner = Arc::clone(&self.inner);
        // Relaxed: the cloned `Arc` already protects the shared state; this
        // counter only elects which external handle performs final shutdown.
        let previous = inner.external_handles.fetch_add(1, Ordering::Relaxed);
        debug_assert!(previous > 0, "scheduler handle count must stay positive");
        Self { inner }
    }
}

/// Borrowing scope for scheduler jobs that must complete before the scope exits.
pub struct SchedulerScope<
    'scope,
    C: WorkClass,
    const BLOCKING_QUEUE_CAPACITY: usize = 256,
    const SPIN_LIMIT: usize = 8192,
> {
    pub(super) scheduler: &'scope ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>,
    pub(super) state: NonNull<SchedulerScopeState>,
    pub(super) priority: Priority,
    pub(super) locality_hint: Option<usize>,
    pub(super) jobs: RefCell<Vec<ScheduledJob>>,
    /// Pins `'scope` invariant. A covariant scope could be shrunk inside the body,
    /// letting `spawn` accept a borrow of a body-local that a buffered job
    /// would use after the body returned and dropped it.
    pub(super) _scope: PhantomData<fn(&'scope ()) -> &'scope ()>,
    pub(super) _class: PhantomData<C>,
}

pub(super) struct SchedulerInner<const BLOCKING_QUEUE_CAPACITY: usize> {
    pub(super) workers: Box<[Arc<WorkerState>]>,
    pub(super) handles: Mutex<Vec<JoinHandle<()>>>,
    /// Counts only public scheduler handles, excluding worker-owned `Arc`s.
    ///
    /// Cloning and dropping handles are cold lifecycle operations. The final
    /// decrement initiates shutdown while its `ThreadScheduler` still owns an
    /// `Arc`, so worker joins complete before shared state can be released.
    pub(super) external_handles: AtomicUsize,
    pub(super) pending_tasks: CacheAligned<AtomicUsize>,
    pub(super) active_workers: CacheAligned<AtomicUsize>,
    pub(super) blocking_pending_tasks: CacheAligned<AtomicUsize>,
    pub(super) blocking_active_workers: CacheAligned<AtomicUsize>,
    pub(super) completed_tasks: CacheAligned<AtomicU64>,
    pub(super) failed_tasks: CacheAligned<AtomicU64>,
    pub(super) admission_caller_runs: CacheAligned<AtomicU64>,
    pub(super) shutdown: CacheAligned<AtomicBool>,
    /// Coordinates the single cold-path owner that joins scheduler workers.
    pub(super) shutdown_join_state: AtomicU8,
    pub(super) join_waiters: CacheAligned<AtomicUsize>,
    pub(super) wait_lock: Mutex<()>,
    pub(super) wait_signal: Condvar,
    pub(super) idle_workers: super::idle::IdleBitset,
    /// Per-worker NUMA node assignment for topology-aware victim selection.
    ///
    /// `worker_numa_nodes[i]` is the NUMA node of worker `i` under
    /// `WorkerPlacement::Pinned`, where the worker bound itself to a processor
    /// of that node before construction returned. It is `None` for unbound
    /// workers, and for every worker when fewer than two nodes are represented
    /// (single-node systems, VMs, containers).
    /// Stored separately from `WorkerState` to avoid cache-line pollution on
    /// the hot steal-path — this slice is read-only after construction.
    pub(super) worker_numa_nodes: Box<[Option<usize>]>,
    /// Lazily initialized so compute-only schedulers allocate no blocking lane.
    pub(super) blocking_lane: OnceLock<BlockingLane<BLOCKING_QUEUE_CAPACITY>>,
    /// Serializes only first-use initialization and shutdown, not submissions.
    pub(super) blocking_lane_init: Mutex<()>,
    pub(super) blocking_lane_prefix: Box<str>,
    /// Test-only rendezvous immediately after shutdown publication.
    #[cfg(test)]
    pub(super) shutdown_started_barrier: OnceLock<Arc<std::sync::Barrier>>,
    /// Cold-path ownership anchor installed once by the executor facade.
    ///
    /// Dynamic type erasure is confined to this construction/destruction
    /// boundary; scheduling and worker dispatch remain monomorphic.
    pub(super) lifetime_owner: OnceLock<Box<dyn Any + Send + Sync>>,
}

pub(super) struct LifoSlot {
    pub(super) state: std::sync::atomic::AtomicU8,
    pub(super) job: std::cell::UnsafeCell<std::mem::MaybeUninit<ScheduledJob>>,
}

// SAFETY: every access to `job` is gated by the `state` machine — a thread
// touches the cell only after winning the empty->1 or full->{1,3} CAS,
// which transfers exclusive ownership of the slot contents.
unsafe impl Sync for LifoSlot {}

impl LifoSlot {
    pub(super) fn new() -> Self {
        Self {
            state: std::sync::atomic::AtomicU8::new(0),
            job: std::cell::UnsafeCell::new(std::mem::MaybeUninit::uninit()),
        }
    }

    pub(super) fn try_push(&self, job: ScheduledJob) -> Option<ScheduledJob> {
        let current = self.state.load(Ordering::Relaxed);
        if current == 0
            && self
                .state
                .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            // SAFETY: the won CAS makes this thread exclusive owner of the
            // cell while state==1; it holds no live value until this store.
            unsafe {
                *self.job.get() = std::mem::MaybeUninit::new(job);
            }
            self.state.store(2, Ordering::Release);
            return None;
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
                // SAFETY: the won full->1 CAS transfers exclusive ownership;
                // reading moves the value out before state returns to 0.
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
                // SAFETY: the won full->3 CAS transfers ownership to the
                // stealing thread; the read moves the value out before the
                // state resets to 0.
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
            // SAFETY: exclusive `&mut self` in drop plus state==2 prove an
            // unconsumed value still sits in the cell; dropping it here
            // discharges the obligation the pop/steal paths would have.
            unsafe {
                std::ptr::drop_in_place((*self.job.get()).as_mut_ptr());
            }
        }
    }
}

melinoe::thread_cached! {
    /// Cached worker ID for the current scheduler thread.
    pub(super) mod current_worker_id: usize;
}

melinoe::thread_cached! {
    /// Indexed-region nesting depth for a participating non-worker caller.
    mod indexed_region_depth: usize;
}

#[inline(always)]
pub(super) fn get_current_worker_id() -> Option<usize> {
    current_worker_id::get()
}

#[inline(always)]
pub(super) fn set_current_worker_id(id: Option<usize>) {
    if let Some(val) = id {
        current_worker_id::set(val);
    } else {
        current_worker_id::clear();
    }
}

pub(super) fn is_in_indexed_region() -> bool {
    indexed_region_depth::get().is_some_and(|depth| depth > 0)
}

pub(super) struct IndexedRegionGuard {
    previous_depth: Option<usize>,
}

impl IndexedRegionGuard {
    pub(super) fn enter() -> Self {
        let previous_depth = indexed_region_depth::get();
        let depth = previous_depth
            .unwrap_or(0)
            .checked_add(1)
            .expect("invariant: indexed-region nesting depth fits usize");
        indexed_region_depth::set(depth);
        Self { previous_depth }
    }
}

impl Drop for IndexedRegionGuard {
    fn drop(&mut self) {
        if let Some(depth) = self.previous_depth {
            indexed_region_depth::set(depth);
        } else {
            indexed_region_depth::clear();
        }
    }
}

/// Per-worker mutable state, individually heap-allocated (`Arc<WorkerState>`).
///
/// `_sector` is a zero-sized marker that raises the allocation's alignment to
/// `moirai_utils::DESTRUCTIVE_INTERFERENCE_SIZE`, so two workers' states cannot
/// land in one false-sharing sector. It replaces a hardcoded
/// `#[repr(align(64))]`: 64 is too narrow on x86-64/aarch64, where the
/// adjacent-line prefetcher operates on 128-byte pairs, and the per-target
/// value belongs to `moirai-utils` rather than a literal here.
pub(super) struct WorkerState {
    _sector: CachePad,
    pub(super) queues: Arc<WorkerQueues>,
    pub(super) lifo_slot: LifoSlot,
    pub(super) thread: OnceLock<thread::Thread>,
    /// Outcome of the worker's processor binding, published once by the worker
    /// under `WorkerPlacement::Pinned` and never set for an unbound worker.
    pub(super) placement: OnceLock<Result<(), PlacementFailure>>,
}

impl WorkerState {
    pub(super) fn new(queues: Arc<WorkerQueues>) -> Self {
        Self {
            _sector: CachePad::new(()),
            queues,
            lifo_slot: LifoSlot::new(),
            thread: OnceLock::new(),
            placement: OnceLock::new(),
        }
    }
}
