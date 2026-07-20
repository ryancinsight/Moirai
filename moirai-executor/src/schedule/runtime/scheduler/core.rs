//! ThreadScheduler core implementation.

use std::{
    marker::PhantomData,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::NonNull,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
    thread,
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    Priority,
};

use moirai_utils::cache::CacheAligned;

use super::super::super::{class::WorkClass, job::ScheduledJob};
use super::super::types::{
    get_current_worker_id, BoundedContendedWake, SchedulerInner, SchedulerScope,
    SchedulerScopeState, ThreadScheduler,
};
use super::super::worker::{
    execute_job, is_quiescent, lock_mutex, next_shared_job, wake_all_workers,
    wake_contended_workers, wake_worker, JOIN_FAST_SPIN_ATTEMPTS,
};

/// Busy-spin iterations a worker-thread scope waiter performs after exhausting
/// runnable work before it parks on the scope condvar. The waiter only reaches
/// this path when its remaining scoped jobs are actively executing on other
/// workers (nothing left to steal), so a short spin absorbs the common
/// finish-imminently case without an OS park round-trip; the timed park below
/// then bounds idle-CPU while `complete_task` provides the real wakeup.
const SCOPE_HELP_SPIN_LIMIT: usize = 64;
/// A per-thread round-robin ticket for spreading queued submissions across
/// workers. Replaces a process-shared `AtomicUsize` that every producer thread
/// RMW'd on each submit: that counter's cache line bounced between all producing
/// cores under high submit rates. Each producer now rotates its own thread-local
/// sequence — contention-free, and still a uniform spread (the value is only a
/// load-balancing hint, not a synchronization point). A single producer still
/// rotates `0, 1, 2, …`, preserving the round-robin the runtime tests pin.
#[inline]
fn next_round_robin_ticket() -> usize {
    use std::cell::Cell;
    thread_local!(static TICKET: Cell<usize> = const { Cell::new(0) });
    TICKET.with(|cell| {
        let ticket = cell.get();
        cell.set(ticket.wrapping_add(1));
        ticket
    })
}

impl ThreadScheduler<256, 8192> {
    /// Start a scheduler with a compute worker set and a lazy blocking lane.
    pub fn new(worker_count: usize, thread_name_prefix: &str) -> ExecutorResult<Self> {
        Self::new_with_config(worker_count, thread_name_prefix)
    }
}

impl<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>
    ThreadScheduler<QUEUE_CAPACITY, SPIN_LIMIT>
{
    /// Start a scheduler with custom configurations.
    pub fn new_with_config(worker_count: usize, thread_name_prefix: &str) -> ExecutorResult<Self> {
        let worker_count = worker_count.max(1);
        let mut queue_owners = Vec::with_capacity(worker_count);
        let workers = (0..worker_count)
            .map(|_| {
                let (owner, queues) = super::super::super::queue::WorkerQueues::new();
                queue_owners.push(owner);
                Arc::new(super::super::types::WorkerState::new(queues))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        // Detect NUMA topology once at construction; derive a per-worker node
        // assignment so `steal_job` can prefer same-node victims without runtime
        // discovery overhead.  Falls back to `None` on single-node / VM systems.
        let topology = moirai_scheduler::numa::CpuTopology::detect();
        let worker_numa_nodes: Box<[Option<usize>]> = if let Some(ref topo) = topology {
            (0..worker_count)
                .map(|id| {
                    // Use CPU core ID equal to worker ID (modular wrap on many-core
                    // systems so indices stay in-bounds regardless of worker count).
                    let core_id = id % topo.logical_cores.max(1);
                    topo.core_to_numa_node(core_id)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        } else {
            vec![None; worker_count].into_boxed_slice()
        };

        let inner = Arc::new(SchedulerInner {
            workers,
            handles: std::sync::Mutex::new(Vec::with_capacity(worker_count)),
            pending_tasks: CacheAligned::new(AtomicUsize::new(0)),
            active_workers: CacheAligned::new(AtomicUsize::new(0)),
            blocking_pending_tasks: CacheAligned::new(AtomicUsize::new(0)),
            blocking_active_workers: CacheAligned::new(AtomicUsize::new(0)),
            completed_tasks: CacheAligned::new(std::sync::atomic::AtomicU64::new(0)),
            failed_tasks: CacheAligned::new(std::sync::atomic::AtomicU64::new(0)),
            shutdown: CacheAligned::new(std::sync::atomic::AtomicBool::new(false)),
            join_waiters: CacheAligned::new(AtomicUsize::new(0)),
            wait_lock: std::sync::Mutex::new(()),
            wait_signal: std::sync::Condvar::new(),
            idle_workers: super::super::idle::IdleBitset::new(worker_count),
            worker_numa_nodes,
            blocking_lane: OnceLock::new(),
            blocking_lane_init: Mutex::new(()),
            blocking_lane_prefix: thread_name_prefix.into(),
        });

        for (worker_id, owner) in queue_owners.into_iter().enumerate() {
            let worker_inner = Arc::clone(&inner);
            let thread_name = format!("{thread_name_prefix}-{worker_id}");
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    super::super::worker::worker_loop::<QUEUE_CAPACITY, SPIN_LIMIT>(
                        worker_inner,
                        worker_id,
                        owner,
                    )
                })
                .map_err(|_| ExecutorError::ThreadPoolCreationFailed)?;

            lock_mutex(&inner.handles).push(handle);
        }

        // Wait until all workers have registered their thread handles
        for worker in inner.workers.iter() {
            while worker.thread.get().is_none() {
                std::thread::yield_now();
            }
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
            jobs: std::cell::RefCell::new(Vec::new()),
            _state: PhantomData,
            _class: PhantomData,
        };

        let body_result = catch_unwind(AssertUnwindSafe(|| body(&scope)));
        let flush_result = scope.flush();
        self.drain_scope(&state);

        match body_result {
            Ok(Ok(())) if state.failed_tasks.load(Ordering::Acquire) => Err(
                ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked),
            ),
            Ok(Ok(())) => flush_result,
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Wait for every job registered on `state` to complete.
    ///
    /// If the caller is itself a scheduler worker, it participates in work
    /// stealing instead of parking: a worker that blocks inside `scope` while
    /// its nested scoped jobs sit unrun would otherwise remove itself from the
    /// pool and deadlock the fork-join (provably so on a single-worker pool, and
    /// a source of use-after-free on the scope's stack-owned state under
    /// concurrent nesting). Running its own queue via `next_job` keeps the pool
    /// making progress, so nesting is deadlock-free and the scope state stays
    /// live until every borrowing job has completed. `next_job(worker_id)` only
    /// pops this worker's own deque and steals into it, so the aliasing rules of
    /// the single-owner Chase–Lev deques are preserved.
    ///
    /// A non-worker caller parks (`SchedulerScopeState::wait`): the worker pool
    /// drains its scoped jobs, so it never starves anything by blocking.
    pub(super) fn drain_scope(&self, state: &SchedulerScopeState) {
        let Some(worker_id) = get_current_worker_id() else {
            state.wait();
            return;
        };

        let inner = &self.inner;
        let mut idle_spins = 0usize;
        loop {
            if state.pending_tasks.load(Ordering::Acquire) == 0 {
                state.wait();
                return;
            }

            if let Some(job) = next_shared_job(inner, worker_id) {
                execute_job(inner, worker_id, job);
                idle_spins = 0;
                continue;
            }

            // Scope still pending but nothing runnable: the remaining scoped jobs
            // are executing on other workers. Spin briefly, then park on the
            // scope condvar with a timeout so `complete_task` wakes us while we
            // still periodically re-probe for freshly stealable work.
            if idle_spins < SCOPE_HELP_SPIN_LIMIT {
                idle_spins += 1;
                core::hint::spin_loop();
                continue;
            }
            idle_spins = 0;

            let guard = lock_mutex(&state.wait_lock);
            if state.pending_tasks.load(Ordering::Acquire) != 0 {
                let _ = state
                    .wait_signal
                    .wait_timeout(guard, std::time::Duration::from_micros(50))
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }

    pub(crate) fn schedule_job<C>(
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

        if C::USES_BLOCKING_LANE {
            if let Some(lane) = self.inner.blocking_lane.get() {
                if self.inner.shutdown.load(Ordering::Acquire) {
                    return Err(ExecutorError::ShuttingDown);
                }
                return lane.submit(
                    priority,
                    locality_hint,
                    job,
                    &self.inner.blocking_pending_tasks,
                );
            }

            let _lane_init = lock_mutex(&self.inner.blocking_lane_init);
            if self.inner.shutdown.load(Ordering::Acquire) {
                return Err(ExecutorError::ShuttingDown);
            }
            if self.inner.blocking_lane.get().is_none() {
                let candidate = super::super::blocking::BlockingLane::new(self.worker_count());
                candidate.start(Arc::clone(&self.inner), &self.inner.blocking_lane_prefix)?;
                if self.inner.blocking_lane.set(candidate).is_err() {
                    return Err(ExecutorError::ThreadPoolCreationFailed);
                }
            }
            let lane = self
                .inner
                .blocking_lane
                .get()
                .expect("invariant: blocking lane initialized before submission");
            return lane.submit(
                priority,
                locality_hint,
                job,
                &self.inner.blocking_pending_tasks,
            );
        }

        let pending_before_submit = self.inner.pending_tasks.load(Ordering::Acquire);
        let active_before_submit = self.inner.active_workers.load(Ordering::Acquire);
        let worker_index = self.select_worker_for_state::<C>(
            priority,
            locality_hint,
            pending_before_submit,
            active_before_submit,
        );
        // SeqCst (not Release): this increment is one half of a store-buffer /
        // Dekker handshake with a parking worker, which does
        // `idle_workers.fetch_or(mask, SeqCst)` then `pending_tasks.load(SeqCst)`
        // before parking, while the producer below does this increment then
        // `idle_workers.load(SeqCst)`. Correctness requires all four accesses to
        // share one SeqCst total order; a Release RMW is NOT in that order, which
        // permits the worker to read `pending == 0` AND the producer to read the
        // worker's idle bit as clear — a lost wakeup that stalls the task until an
        // unrelated submission. The increment must stay BEFORE the push so the
        // `execute_job` decrement (worker.rs) can never underflow from 0.
        // (On x86 `lock xadd` is already full-barrier, so this is free.)
        let previous_pending = self.inner.pending_tasks.fetch_add(1, Ordering::SeqCst);

        let rejected = if get_current_worker_id() == Some(worker_index) {
            self.inner.workers[worker_index]
                .lifo_slot
                .try_push(job)
                .and_then(|job| {
                    self.inner.workers[worker_index]
                        .queues
                        .try_push_external(priority, job)
                })
        } else {
            self.inner.workers[worker_index]
                .queues
                .try_push_external(priority, job)
        };

        if let Some(job) = rejected {
            self.inner.pending_tasks.fetch_sub(1, Ordering::SeqCst);
            drop(job);
            return Err(ExecutorError::ResourceExhausted(format!(
                "worker {worker_index} scheduler admission queue is full"
            )));
        }

        // Try to wake up an idle worker via the lock-free wake lottery. The
        // bitset spans every worker (not just the first 64), so high-index
        // workers on large pools are reachable directly rather than stranded.
        let mut woken = false;
        if let Some(worker_to_wake) = self.inner.idle_workers.claim_one(self.inner.workers.len()) {
            wake_worker(&self.inner.workers[worker_to_wake]);
            woken = true;
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
                } else {
                    let wake_index = worker_index.wrapping_add(previous_pending) % worker_count;
                    wake_worker(&self.inner.workers[wake_index]);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn schedule_scoped_job<'scope, C, F>(
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
            + self.inner.blocking_pending_tasks.load(Ordering::Acquire)
    }

    /// Number of workers currently executing jobs.
    pub fn active_workers(&self) -> usize {
        self.inner.active_workers.load(Ordering::Acquire)
            + self.inner.blocking_active_workers.load(Ordering::Acquire)
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
        // SeqCst: the joiner half of the quiescence Dekker handshake — this
        // registration must be in the same SeqCst total order as a completing
        // worker's `active_workers` decrement and its `join_waiters` load in
        // `notify_quiescent`. With AcqRel the store-buffer outcome hangs `join()`
        // (see `execute_job` and `tests/loom_join_quiescence.rs`). The re-check
        // below runs under `wait_lock`, closing the window against a concurrent
        // `notify_quiescent` that also takes the lock before signalling.
        self.inner.join_waiters.fetch_add(1, Ordering::SeqCst);
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
    pub fn metrics(&self) -> super::super::types::ScheduleMetrics {
        super::super::types::ScheduleMetrics {
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

        let _lane_init = lock_mutex(&self.inner.blocking_lane_init);
        if let Some(lane) = self.inner.blocking_lane.get() {
            lane.shutdown();
        }

        let mut handles = lock_mutex(&self.inner.handles);
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }

    #[cfg(test)]
    pub(crate) fn select_worker<C>(&self, priority: Priority, locality_hint: Option<usize>) -> usize
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

    pub(crate) fn select_worker_for_state<C>(
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
            return C::SERIAL_AFFINITY_OFFSET.wrapping_add(priority.index()) % worker_count;
        }

        let ticket = next_round_robin_ticket();
        ticket
            .wrapping_add(C::AFFINITY_OFFSET)
            .wrapping_add(priority.index())
            % worker_count
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
