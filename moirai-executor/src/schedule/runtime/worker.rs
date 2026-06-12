//! Worker loop and associated free functions for the thread scheduler runtime.

use std::{
    sync::{Mutex, MutexGuard},
    thread,
    time::Duration,
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    Priority,
};

use moirai_pal::reactor::IoReactor;

use super::super::job::ScheduledJob;

use super::types::{set_current_worker_id, ContendedWakePolicy, SchedulerInner, WorkerState};

#[cfg(feature = "scheduler-diagnostics")]
use super::types::BoundedContendedWake;

pub(super) const WORKER_IDLE_SPIN_ATTEMPTS: usize = 256;
pub(super) const JOIN_FAST_SPIN_ATTEMPTS: usize = WORKER_IDLE_SPIN_ATTEMPTS;

pub(super) fn worker_loop<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>(
    inner: std::sync::Arc<SchedulerInner<QUEUE_CAPACITY>>,
    worker_id: usize,
) {
    set_current_worker_id(Some(worker_id));
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
                    |alloc| unsafe {
                        alloc.periodic_defragmentation_sweep::<StandardPolicy>();
                    },
                );
        }
    }
}

pub(super) fn next_job<const QUEUE_CAPACITY: usize>(
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

pub(super) fn execute_job<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
    worker_id: usize,
    job: ScheduledJob,
) {
    use std::sync::atomic::Ordering;
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
    use std::sync::atomic::Ordering;
    inner.shutdown.load(Ordering::Acquire) && inner.pending_tasks.load(Ordering::Acquire) == 0
}

fn spin_for_work<const QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
    worker_id: usize,
) -> bool {
    use std::sync::atomic::Ordering;
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
    use std::sync::atomic::Ordering;
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
    use std::sync::atomic::Ordering;
    if worker_id < 64 {
        let mask = 1 << worker_id;
        inner.idle_workers.fetch_or(mask, Ordering::SeqCst);
        while inner.pending_tasks.load(Ordering::SeqCst) == 0
            && !inner.shutdown.load(Ordering::SeqCst)
        {
            if let Some(reactor) = IoReactor::get_active() {
                let _ = reactor.run_iteration(Some(Duration::from_millis(1)));
                if inner.pending_tasks.load(Ordering::SeqCst) > 0
                    || inner.shutdown.load(Ordering::SeqCst)
                {
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
                if inner.pending_tasks.load(Ordering::Acquire) > 0
                    || inner.shutdown.load(Ordering::Acquire)
                {
                    break;
                }
            } else {
                thread::park();
            }
        }
    }
}

pub(super) fn wake_worker<const QUEUE_CAPACITY: usize>(worker: &WorkerState<QUEUE_CAPACITY>) {
    if let Some(thread) = worker.thread.get() {
        thread.unpark();
    }
}

pub(super) fn wake_all_workers<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
) {
    for worker in inner.workers.iter() {
        wake_worker(worker);
    }
}

pub(super) trait ContendedWakable {
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

pub(super) fn notify_quiescent<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
) {
    use std::sync::atomic::Ordering;
    if inner.join_waiters.load(Ordering::Acquire) != 0 && is_quiescent(inner) {
        let _guard = lock_mutex(&inner.wait_lock);
        if is_quiescent(inner) {
            inner.wait_signal.notify_all();
        }
    }
}

pub(super) fn is_quiescent<const QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<QUEUE_CAPACITY>,
) -> bool {
    use std::sync::atomic::Ordering;
    inner.pending_tasks.load(Ordering::Acquire) == 0
        && inner.active_workers.load(Ordering::Acquire) == 0
}

pub(super) fn priority_weight(priority: Priority) -> usize {
    match priority {
        Priority::Low => 0,
        Priority::Normal => 1,
        Priority::High => 2,
        Priority::Critical => 3,
    }
}

pub(super) fn inline_map_reduce<T, Map, Reduce>(
    count: usize,
    identity: T,
    map: Map,
    reduce: Reduce,
) -> ExecutorResult<T>
where
    Map: Fn(usize) -> T,
    Reduce: Fn(T, T) -> T,
{
    use std::panic::{catch_unwind, AssertUnwindSafe};
    catch_unwind(AssertUnwindSafe(|| {
        let mut accumulator = identity;
        for index in 0..count {
            accumulator = reduce(accumulator, map(index));
        }
        accumulator
    }))
    .map_err(|_| ExecutorError::SpawnFailed(moirai_core::error::TaskError::Panicked))
}

pub(super) fn map_reduce_range<T, Map, Reduce>(
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

pub(super) fn indexed_chunk_count(count: usize, worker_count: usize) -> usize {
    count.min(worker_count.max(1).saturating_add(1))
}

pub(super) fn indexed_reduce_chunk_count<T>(count: usize, worker_count: usize) -> usize {
    use super::super::reduce::inline_reduction_limit;
    let worker_count = worker_count.max(1);
    let max_chunks = count.min(worker_count.saturating_add(1));
    let scheduled_chunk_floor = inline_reduction_limit::<T>(worker_count)
        .saturating_mul(2)
        .max(1);

    max_chunks.min(count.div_ceil(scheduled_chunk_floor).max(1))
}

pub(super) fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
