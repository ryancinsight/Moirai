//! The scheduler seam — the dependency-inversion boundary between the executor
//! façade ([`HybridExecutor`](crate::hybrid::HybridExecutor)) and a concrete
//! work-stealing runtime.
//!
//! `HybridExecutor` depends on these role traits, not on the concrete
//! [`ThreadScheduler`], so an alternative runtime (for example a single-threaded
//! scheduler for `wasm32` targets) can be substituted as `HybridExecutor<S>`
//! without touching the façade. The contract is split per the
//! interface-segregation principle, so a substitute implements only the roles it
//! supports. Every method is statically dispatched (generic over the work class
//! and closure), so the seam compiles to the same code as a direct concrete
//! call — it is zero-cost.
//!
//! The richer borrowing `scope` API is intentionally *not* part of the seam: its
//! signature exposes a concrete [`SchedulerScope`](crate::schedule::SchedulerScope)
//! borrow handle, so it stays an inherent capability of `ThreadScheduler` rather
//! than forcing every substitute to reproduce that machinery.

use moirai_core::{error::ExecutorResult, Priority};

use crate::schedule::class::WorkClass;
use crate::schedule::runtime::{ScheduleMetrics, ThreadScheduler};

/// Submit type-erased work to a scheduler.
pub trait WorkSubmit: Send + Sync + 'static {
    /// Schedule `task` as a job of work class `C` at `priority`, optionally
    /// biased toward the worker named by `locality_hint`.
    ///
    /// # Errors
    /// Returns [`ExecutorError::ShuttingDown`](moirai_core::error::ExecutorError::ShuttingDown)
    /// if the scheduler is draining.
    fn schedule<C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'static;
}

/// Introspect and control a scheduler's lifecycle.
pub trait SchedulerControl {
    /// Number of queued-but-not-yet-running jobs.
    fn pending_tasks(&self) -> usize;
    /// Number of jobs currently executing.
    fn active_workers(&self) -> usize;
    /// Number of worker threads.
    fn worker_count(&self) -> usize;
    /// Whether any work is queued or running.
    fn has_work(&self) -> bool;
    /// Block until the scheduler is quiescent (no queued or running work).
    ///
    /// # Errors
    /// Returns an [`ExecutorError`](moirai_core::error::ExecutorError) if the
    /// scheduler cannot reach quiescence.
    fn join(&self) -> ExecutorResult<()>;
    /// Drain queued work and stop the worker sets.
    ///
    /// Exactly one non-worker caller joins the worker sets. Other external
    /// callers wait for that join. Scheduler workers close the blocking lane
    /// and return before the election; when no external caller remains, worker
    /// ownership releases the scheduler after every accepted task drains.
    fn shutdown(&self);
    /// Snapshot the scheduler metrics.
    fn metrics(&self) -> ScheduleMetrics;
}

/// Indexed data-parallel fan-out without per-item result storage.
///
/// Calls partition non-empty domains across the available worker-plus-caller
/// lanes. Operation-level execution policies own profitability thresholds
/// before invoking this scheduler seam.
pub trait DataParallel {
    /// Apply `task` to every index in `0..count`, completing before return.
    ///
    /// # Errors
    /// Returns an [`ExecutorError`](moirai_core::error::ExecutorError) if the
    /// scheduler is draining or a chunk panics.
    fn for_each_indexed<C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        count: usize,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: Fn(usize) + Send + Sync;

    /// Map every index in `0..count` and reduce the results with `identity` as
    /// the neutral element of `reduce`.
    ///
    /// # Errors
    /// Returns an [`ExecutorError`](moirai_core::error::ExecutorError) if the
    /// scheduler is draining or a chunk panics.
    fn map_reduce_indexed<C, T, Map, Reduce>(
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
        Reduce: Fn(T, T) -> T + Send + Sync;
}

/// The full runtime-scheduler contract that `HybridExecutor` depends on.
///
/// `Clone` is required because the async lane clones the scheduler handle into
/// each future's waker. This is a marker over the role traits, implemented for
/// every type that satisfies all of them.
pub trait WorkScheduler: WorkSubmit + SchedulerControl + DataParallel + Clone {}

impl<S> WorkScheduler for S where S: WorkSubmit + SchedulerControl + DataParallel + Clone {}

// ── ThreadScheduler: the canonical implementation ──────────────────────────
//
// Each role method forwards to the same-named inherent method via the
// `Type::method` form, which resolves to the inherent method (preferred over a
// trait method of the same name), so there is no recursion. The seam therefore
// adds no behavior — it only re-exposes the existing surface as a substitutable
// contract.

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> WorkSubmit
    for ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn schedule<C, F>(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'static,
    {
        ThreadScheduler::schedule::<C, F>(self, priority, locality_hint, task)
    }
}

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> SchedulerControl
    for ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn pending_tasks(&self) -> usize {
        ThreadScheduler::pending_tasks(self)
    }
    fn active_workers(&self) -> usize {
        ThreadScheduler::active_workers(self)
    }
    fn worker_count(&self) -> usize {
        ThreadScheduler::worker_count(self)
    }
    fn has_work(&self) -> bool {
        ThreadScheduler::has_work(self)
    }
    fn join(&self) -> ExecutorResult<()> {
        ThreadScheduler::join(self)
    }
    fn shutdown(&self) {
        ThreadScheduler::shutdown(self)
    }
    fn metrics(&self) -> ScheduleMetrics {
        ThreadScheduler::metrics(self)
    }
}

impl<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize> DataParallel
    for ThreadScheduler<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>
{
    fn for_each_indexed<C, F>(
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
        ThreadScheduler::for_each_indexed::<C, F>(self, priority, locality_hint, count, task)
    }

    fn map_reduce_indexed<C, T, Map, Reduce>(
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
        ThreadScheduler::map_reduce_indexed::<C, T, Map, Reduce>(
            self,
            priority,
            locality_hint,
            count,
            identity,
            map,
            reduce,
        )
    }
}
