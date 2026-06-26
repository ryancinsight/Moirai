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
    // Randomized victim order. Deterministic round-robin makes every idle
    // worker probe victims in the same `worker_id+1, +2, …` sequence, so after a
    // fork/join barrier (scope, map_reduce_indexed) the freshly-idle workers
    // pile onto the same victims' `top` CAS in lockstep. A thread-local
    // xorshift64 start spreads the first — and most contended — steal attempt
    // across victims (Blumofe–Leiserson randomized work-stealing). The full
    // ring is still scanned, so coverage and worst-case cost are unchanged.
    let start = next_steal_start();
    // Scan the full ring from a random origin, skipping self, so all
    // `worker_count - 1` victims are still covered regardless of `start`.
    for offset in 0..worker_count {
        let victim_index = (start.wrapping_add(offset)) % worker_count;
        if victim_index == worker_id {
            continue;
        }
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

/// Thread-local xorshift64 producing a randomized starting victim index.
///
/// Seeded lazily from the per-thread RNG cell's own address (stable and unique
/// per worker thread, forced non-zero), so it needs no shared atomic on the hot
/// path — the seed source is contention-free by construction.
fn next_steal_start() -> usize {
    use std::cell::Cell;
    thread_local!(static RNG: Cell<u64> = const { Cell::new(0) });
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
    // Register this worker as parked in the wake bitset, then re-check
    // `pending_tasks` under the same SeqCst order. The `set` and the load form
    // the worker half of the store-buffer handshake with `schedule_job`'s SeqCst
    // increment + bitset scan, which is what rules out a lost wakeup. Every
    // worker registers (no id < 64 special case), so large pools have no
    // unreachable workers.
    inner.idle_workers.set(worker_id);
    while inner.pending_tasks.load(Ordering::SeqCst) == 0 && !inner.shutdown.load(Ordering::SeqCst)
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
    inner.idle_workers.clear(worker_id);
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

/// Minimum number of index iterations per scheduled chunk.
///
/// Each chunk beyond the first requires one `thread::unpark()` + one SeqCst
/// fence on the submission path (~200–500 ns on x86/ARM). Below this element
/// count per chunk, dispatch overhead exceeds the benefit of parallelism.
/// Mirrors the guard `indexed_reduce_chunk_count` applies via
/// `inline_reduction_limit`, but expressed as a fixed floor independent of
/// element size (index-only ops have no type parameter to derive from).
pub(super) const MIN_ELEMENTS_PER_CHUNK: usize = 256;

pub(super) fn indexed_chunk_count(count: usize, worker_count: usize) -> usize {
    let max_by_workers = count.min(worker_count.max(1).saturating_add(1));
    // Cap chunk count so every scheduled chunk processes at least
    // MIN_ELEMENTS_PER_CHUNK iterations, keeping dispatch overhead sub-dominant.
    let max_by_size = count.div_ceil(MIN_ELEMENTS_PER_CHUNK).max(1);
    max_by_workers.min(max_by_size)
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

#[cfg(test)]
mod indexed_chunk_count_tests {
    use super::{indexed_chunk_count, MIN_ELEMENTS_PER_CHUNK};

    #[test]
    fn collapses_to_one_chunk_below_min_elements() {
        // 255 elements with 8 workers: size cap = ceil(255/256) = 1, so inline.
        assert_eq!(indexed_chunk_count(255, 8), 1);
        assert_eq!(indexed_chunk_count(1, 8), 1);
        assert_eq!(indexed_chunk_count(0, 8), 0);
    }

    #[test]
    fn exactly_min_elements_gives_one_chunk() {
        // Exactly MIN_ELEMENTS_PER_CHUNK is still one chunk (ceil(256/256) = 1).
        assert_eq!(indexed_chunk_count(MIN_ELEMENTS_PER_CHUNK, 8), 1);
    }

    #[test]
    fn scales_with_element_count_not_just_workers() {
        // 1024 elements, 8 workers: max_by_workers = 9, max_by_size = ceil(1024/256) = 4.
        assert_eq!(indexed_chunk_count(1024, 8), 4);
        // 2048 elements, 8 workers: max_by_workers = 9, max_by_size = 8.
        assert_eq!(indexed_chunk_count(2048, 8), 8);
        // 2304 elements, 8 workers: max_by_workers = 9, max_by_size = ceil(2304/256) = 9.
        assert_eq!(indexed_chunk_count(2304, 8), 9);
    }

    #[test]
    fn never_exceeds_worker_count_plus_one() {
        // Large counts are still bounded by worker_count + 1.
        assert_eq!(indexed_chunk_count(1_000_000, 8), 9);
    }

    #[test]
    fn single_worker_always_one_chunk() {
        // max_by_workers = min(n, 2); but for n=1 → 1.
        assert_eq!(indexed_chunk_count(1024, 1), 2);
        assert_eq!(indexed_chunk_count(128, 1), 1);
    }
}
