//! Worker spinning and parking protocol.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::super::{idle::IdleBitset, types::SchedulerInner};
use super::next_steal_start;

pub(super) fn should_stop<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
) -> bool {
    // This is the worker half of the shutdown/admission StoreLoad handshake.
    // A producer publishes pending before observing shutdown; the worker
    // publishes shutdown before observing pending. SeqCst makes the outcome
    // where both sides miss the other's publication unreachable.
    inner.shutdown.load(Ordering::SeqCst) && inner.pending_tasks.load(Ordering::SeqCst) == 0
}

pub(super) fn spin_for_work<const BLOCKING_QUEUE_CAPACITY: usize, const SPIN_LIMIT: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
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

        // Periodically check if other workers have stealable tasks to avoid parking.
        if attempt % 32 == 0 && (has_stealable_work(inner, worker_id) || should_stop(inner)) {
            return true;
        }
    }

    false
}

fn has_stealable_work<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
) -> bool {
    let worker_count = inner.workers.len();
    // Bounded randomized probe: instead of scanning all `worker_count - 1`
    // victims every 32 spins (O(N) per check, cache-line churn on large pools),
    // probe at most `STEAL_PROBE_LIMIT` victims starting from a random offset.
    // Missing stealable work is harmless: the worker continues spinning and
    // `steal_job` still scans the full ring before this wait path is reached.
    const STEAL_PROBE_LIMIT: usize = 8;
    let start = next_steal_start();
    let probe_count = worker_count.min(STEAL_PROBE_LIMIT);
    for offset in 0..probe_count {
        let victim_index = start.wrapping_add(offset) % worker_count;
        if victim_index == worker_id {
            continue;
        }
        let victim = &inner.workers[victim_index];
        if !victim.queues.is_empty() || victim.lifo_slot.state.load(Ordering::Relaxed) == 2 {
            return true;
        }
    }
    false
}

pub(super) fn wait_for_work<const BLOCKING_QUEUE_CAPACITY: usize>(
    inner: &SchedulerInner<BLOCKING_QUEUE_CAPACITY>,
    worker_id: usize,
) {
    wait_for_work_with_park(
        &inner.idle_workers,
        &inner.pending_tasks,
        &inner.shutdown,
        worker_id,
        std::thread::park,
    );
}

fn wait_for_work_with_park(
    idle_workers: &IdleBitset,
    pending_tasks: &AtomicUsize,
    shutdown: &AtomicBool,
    worker_id: usize,
    mut park: impl FnMut(),
) {
    loop {
        // Register before every park attempt, then re-check `pending_tasks` under
        // the same SeqCst order. A producer claims and clears this bit before
        // waking the worker; if another worker drains that task first, this loop
        // must publish the bit again before re-parking. The `set` and load form
        // the worker half of the store-buffer handshake with `schedule_job`'s
        // SeqCst increment + bitset scan, ruling out a lost wakeup.
        idle_workers.set(worker_id);
        if pending_tasks.load(Ordering::SeqCst) != 0 || shutdown.load(Ordering::SeqCst) {
            break;
        }

        // Park until `schedule_job` unparks us. Async I/O readiness is driven by
        // moirai_pal's dedicated global reactor thread, whose wakers reschedule
        // their tasks through `schedule_job`; a parked worker is therefore woken
        // the same way for async completion as for fresh sync work.
        park();
    }
    idle_workers.clear(worker_id);
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use super::{wait_for_work_with_park, IdleBitset};

    #[test]
    fn consumed_wake_republishes_worker_before_repark() {
        let idle_workers = IdleBitset::new(1);
        let pending_tasks = AtomicUsize::new(0);
        let shutdown = AtomicBool::new(false);
        let park_attempts = Cell::new(0);

        wait_for_work_with_park(&idle_workers, &pending_tasks, &shutdown, 0, || {
            let attempt = park_attempts.get();
            park_attempts.set(attempt + 1);
            assert_eq!(idle_workers.claim_one(1), Some(0));
            if attempt == 1 {
                shutdown.store(true, Ordering::SeqCst);
            }
        });

        assert_eq!(park_attempts.get(), 2);
        assert_eq!(idle_workers.claim_one(1), None);
    }
}
