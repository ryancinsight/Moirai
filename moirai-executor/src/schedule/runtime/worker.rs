//! Worker loop and associated free functions for the thread scheduler runtime.

mod indexed;
mod wait;

use std::{
    sync::{Mutex, MutexGuard},
    thread::{self, JoinHandle},
};

use super::super::{job::ScheduledJob, queue::WorkerQueueOwner, COOPERATIVE_SPIN_ATTEMPTS};

use super::types::{set_current_worker_id, ContendedWakePolicy, SchedulerInner, WorkerState};
pub(super) use indexed::{
    indexed_chunk_bounds, indexed_chunk_count, inline_map_reduce, map_reduce_range,
};
use wait::{should_stop, spin_for_work, wait_for_work};

#[cfg(feature = "scheduler-diagnostics")]
use super::types::BoundedContendedWake;

pub(super) const WORKER_IDLE_SPIN_ATTEMPTS: usize = COOPERATIVE_SPIN_ATTEMPTS;
pub(super) const JOIN_FAST_SPIN_ATTEMPTS: usize = WORKER_IDLE_SPIN_ATTEMPTS;

/// Join every worker except the thread currently executing shutdown.
///
/// A runtime may lose its final external owner inside one of its own jobs. The
/// current worker cannot join itself; dropping that handle detaches it, while
/// its local scheduler `Arc` keeps runtime state alive until the job and worker
/// loop return.
pub(super) fn join_other_threads(handles: &mut Vec<JoinHandle<()>>) {
    let current = thread::current().id();
    while let Some(handle) = handles.pop() {
        if handle.thread().id() != current {
            let _ = handle.join();
        }
    }
}

pub(super) fn worker_loop<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>(
    inner: std::sync::Arc<SchedulerInner<BLOCKING_QUEUE_CAPACITY>>,
    worker_id: usize,
    mut owner: WorkerQueueOwner,
) {
    set_current_worker_id(Some(worker_id));
    let _ = inner.workers[worker_id].thread.set(thread::current());

    loop {
        if let Some(job) = next_job(&inner, worker_id, &mut owner) {
            execute_job(&inner, worker_id, job);
            continue;
        }

        if should_stop(&inner) {
            break;
        }

        if spin_for_work::<BLOCKING_QUEUE_CAPACITY, SPIN_LIMIT>(&inner, worker_id) {
            continue;
        }

        // Run defragmentation sweeps only right before blocking in wait_for_work
        // to avoid latency overheads during active work stealing and spinning.
        run_idle_memory_maintenance();

        wait_for_work(&inner, worker_id);
    }
}

#[cfg(feature = "mnemosyne")]
melinoe::thread_cached! {
    mod last_maintenance_time: std::time::Instant;
}

#[inline]
fn run_idle_memory_maintenance() {
    #[cfg(feature = "mnemosyne")]
    {
        use mnemosyne::{LocalAllocatorSelector, MemoryBackendWrapper, StandardPolicy};
        if <MemoryBackendWrapper as LocalAllocatorSelector<MemoryBackendWrapper>>::get_allocator_ptr_raw().is_null() {
            return;
        }

        let now = std::time::Instant::now();
        let should_run = if let Some(last) = last_maintenance_time::get() {
            if now.duration_since(last) >= std::time::Duration::from_millis(500) {
                last_maintenance_time::set(now);
                true
            } else {
                false
            }
        } else {
            last_maintenance_time::set(now);
            true
        };

        if should_run {
            let _ =
                <MemoryBackendWrapper as LocalAllocatorSelector<MemoryBackendWrapper>>::with_allocator(
                    // SAFETY: the selector hands this closure the allocator
                    // instance registered for this worker thread; the sweep
                    // requires exactly that exclusive per-thread allocator
                    // view for its duration.
                    |alloc| unsafe {
                        alloc.periodic_defragmentation_sweep::<StandardPolicy>();
                    },
                );
        }
    }
}

pub(super) fn next_job<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
    owner: &mut WorkerQueueOwner,
) -> Option<ScheduledJob> {
    let local = &inner.workers[worker_id];
    local
        .lifo_slot
        .pop()
        .or_else(|| owner.pop_local())
        .or_else(|| steal_job(inner, worker_id, owner))
}

/// Obtain runnable work using only shared top-side capabilities.
pub(super) fn next_shared_job<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
) -> Option<ScheduledJob> {
    let local = &inner.workers[worker_id];
    local
        .lifo_slot
        .pop()
        .or_else(|| local.queues.steal_one())
        .or_else(|| steal_shared_job(inner, worker_id))
}

fn steal_job<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
    owner: &mut WorkerQueueOwner,
) -> Option<ScheduledJob> {
    let worker_count = inner.workers.len();
    let my_node = inner.worker_numa_nodes.get(worker_id).copied().flatten();

    // Two-pass NUMA-aware victim selection:
    //
    // Pass 1 (same-NUMA-node): prefer victims on the same NUMA node as the
    // thief.  Same-node steals access memory already in the local NUMA bank,
    // avoiding cross-socket NUMA traffic on multi-socket systems.  Skipped
    // when topology is unavailable (my_node == None) or only one node exists.
    //
    // Pass 2 (all workers): fall back to the full-ring randomised scan so
    // coverage and worst-case load balance are preserved — same as the
    // previous implementation.
    //
    // Both passes use xorshift64 randomisation (Blumofe–Leiserson) to spread
    // the first steal attempt and prevent thundering-herd CAS contention.
    if let Some(node) = my_node {
        let start = next_steal_start();
        for offset in 0..worker_count {
            let victim_index = (start.wrapping_add(offset)) % worker_count;
            if victim_index == worker_id {
                continue;
            }
            // Only try same-node victims in pass 1.
            if inner.worker_numa_nodes.get(victim_index).copied().flatten() != Some(node) {
                continue;
            }
            let victim = &inner.workers[victim_index];
            if let Some(job) = owner.steal_batch(&victim.queues) {
                return Some(job);
            }
            if let Some(job) = victim.lifo_slot.steal() {
                return Some(job);
            }
        }
    }

    // Pass 2: full ring scan from a fresh random origin, skipping self.
    let start = next_steal_start();
    for offset in 0..worker_count {
        let victim_index = (start.wrapping_add(offset)) % worker_count;
        if victim_index == worker_id {
            continue;
        }
        let victim = &inner.workers[victim_index];
        if let Some(job) = owner.steal_batch(&victim.queues) {
            return Some(job);
        }
        if let Some(job) = victim.lifo_slot.steal() {
            return Some(job);
        }
    }

    None
}

fn steal_shared_job<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
) -> Option<ScheduledJob> {
    let worker_count = inner.workers.len();
    let start = next_steal_start();
    for offset in 0..worker_count {
        let victim_index = start.wrapping_add(offset) % worker_count;
        if victim_index == worker_id {
            continue;
        }
        let victim = &inner.workers[victim_index];
        if let Some(job) = victim.queues.steal_one() {
            return Some(job);
        }
        if let Some(job) = victim.lifo_slot.steal() {
            return Some(job);
        }
    }
    None
}

/// Thread-local xorshift64 producing a randomized starting victim index.
///
/// Seeded lazily from the per-thread RNG cell's own address (stable and unique
/// per worker thread, forced non-zero), so it needs no shared atomic on the hot
/// path -- the seed source is contention-free by construction.
fn next_steal_start() -> usize {
    use std::cell::Cell;
    std::thread_local! {
        // clippy 1.97.0 FP: already const. ATLAS-MNEMOSYNE-CI-1.
        #[allow(clippy::missing_const_for_thread_local)]
        static RNG: Cell<u64> = const { Cell::new(0) };
    }
    RNG.with(|cell| {
        let mut x = cell.get();
        if x == 0 {
            x = (cell as *const Cell<u64> as u64) | 1;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        cell.set(x);
        x as usize
    })
}

pub(super) fn execute_job<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
    job: ScheduledJob,
) {
    execute_job_with_counters(
        inner,
        worker_id,
        job,
        &inner.pending_tasks,
        &inner.active_workers,
    );
}

pub(super) fn execute_blocking_job<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
    job: ScheduledJob,
) {
    execute_job_with_counters(
        inner,
        worker_id,
        job,
        &inner.blocking_pending_tasks,
        &inner.blocking_active_workers,
    );
}

fn execute_job_with_counters<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
    job: ScheduledJob,
    pending_tasks: &std::sync::atomic::AtomicUsize,
    active_workers: &std::sync::atomic::AtomicUsize,
) {
    use std::sync::atomic::Ordering;
    active_workers.fetch_add(1, Ordering::Release);
    pending_tasks.fetch_sub(1, Ordering::Release);

    if job.execute(worker_id) {
        inner.completed_tasks.fetch_add(1, Ordering::Relaxed);
    } else {
        inner.failed_tasks.fetch_add(1, Ordering::Relaxed);
    }

    // SeqCst (not AcqRel): this decrement-to-zero publishes quiescence to a
    // parking `join()` waiter and is one half of a store-buffer (Dekker)
    // handshake — the worker stores `active -> 0` here then `notify_quiescent`
    // loads `join_waiters`, while `join` stores `join_waiters += 1` then
    // `is_quiescent` loads `active`. All four accesses must share one SeqCst
    // total order; with AcqRel the StoreLoad reordering admits the lost-wakeup
    // outcome (joiner reads stale `active != 0` and parks while the worker reads
    // stale `join_waiters == 0` and never signals — a hung `join()`), proven
    // reachable by `tests/loom_join_quiescence.rs`. On x86 `lock sub`/`lock xadd`
    // is already a full barrier, so this is free.
    if active_workers.fetch_sub(1, Ordering::SeqCst) == 1 {
        notify_quiescent(inner);
    }
}

pub(super) fn wake_worker(worker: &WorkerState) {
    if let Some(thread) = worker.thread.get() {
        thread.unpark();
    }
}

pub(super) fn wake_all_workers<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
) {
    for worker in inner.workers.iter() {
        wake_worker(worker);
    }
}

pub(super) trait ContendedWakable {
    /// Worker-pool size; consumed only by the diagnostics wake-decision path.
    #[cfg(feature = "scheduler-diagnostics")]
    fn worker_count(&self) -> usize;
    /// Direct single-worker wake; consumed only by the diagnostics wake-decision path.
    #[cfg(feature = "scheduler-diagnostics")]
    fn wake_worker(&self, worker_index: usize);
    fn wake_contended<P>(&self, worker_index: usize, previous_pending: usize) -> usize
    where
        P: ContendedWakePolicy;
}

impl<const BLOCKING_QUEUE_CAPACITY: usize> ContendedWakable
    for SchedulerInner<BLOCKING_QUEUE_CAPACITY>
{
    #[cfg(feature = "scheduler-diagnostics")]
    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    #[cfg(feature = "scheduler-diagnostics")]
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
pub(super) fn wake_contended_workers<P>(
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
pub(super) fn diagnostic_publish_work_available(
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

pub(super) fn notify_quiescent<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
) {
    use std::sync::atomic::Ordering;
    // SeqCst: the worker half of the quiescence Dekker handshake (see the
    // `active_workers` decrement in `execute_job`). This load must be in the same
    // SeqCst total order as `join`'s `join_waiters` increment, or a just-arrived
    // waiter is missed.
    if inner.join_waiters.load(Ordering::SeqCst) != 0 && is_quiescent(inner) {
        let _guard = lock_mutex(&inner.wait_lock);
        if is_quiescent(inner) {
            inner.wait_signal.notify_all();
        }
    }
}

pub(super) fn is_quiescent<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
) -> bool {
    use std::sync::atomic::Ordering;
    // SeqCst: the `active_workers` load is the joiner's half of the quiescence
    // Dekker handshake (see `execute_job`) and must sit in the shared SeqCst
    // total order; `pending_tasks` is loaded SeqCst too so the full quiescence
    // predicate is evaluated against one consistent order. SeqCst loads are a
    // plain load on x86 (`mov`), so this is cheap on the common target.
    inner.pending_tasks.load(Ordering::SeqCst) == 0
        && inner.active_workers.load(Ordering::SeqCst) == 0
        && inner.blocking_pending_tasks.load(Ordering::SeqCst) == 0
        && inner.blocking_active_workers.load(Ordering::SeqCst) == 0
}

pub(super) fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
