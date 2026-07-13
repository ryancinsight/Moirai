//! Priority-aware worker queues.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use moirai_core::Priority;
use moirai_scheduler::{ChaseLevDeque, ChaseLevStealer, StealResult};
use moirai_utils::CacheAligned;

use super::job::ScheduledJob;

/// One queue per priority level; indices come from [`Priority::index`] (SSOT).
const PRIORITY_LEVELS: usize = Priority::Critical.index() + 1;
/// Initial slot count for each worker's multi-producer injector queue.
///
/// The injector holds `(Priority, ScheduledJob)` elements (~256 B each); the
/// previous `LockFreeQueue::new()` default of 65536 slots pre-allocated ~16 MiB
/// per worker at startup regardless of load (~256 MiB idle on 16 workers). A
/// 1024-slot injector bounds that to ~256 KiB per worker while the enqueue path
/// still backs off and eventually succeeds when momentarily full (unblocked-
/// sender contract), so throughput under burst load is unaffected.
///
/// This is a subtractive sizing fix; threading `ExecutorConfig::
/// max_global_queue_size` to this construction site is a separate follow-up.
const INJECTOR_CAPACITY: usize = 1024;
/// Pop scan order: highest [`Priority::index`] first.
const PRIORITY_POP_ORDER: [usize; PRIORITY_LEVELS] = [
    Priority::Critical.index(),
    Priority::High.index(),
    Priority::Normal.index(),
    Priority::Low.index(),
];

/// Per-worker task queues partitioned by priority using lock-free Chase-Lev deques.
///
/// Local pop operations and push operations are lock-free. Local operations (from the owner thread)
/// proceed directly on the private SPSC deques, while non-local schedules place tasks into the
/// lock-free multi-producer injector queue. Steal operations are entirely lock-free and proceed
/// without acquiring any locks.
///
/// Queue contents are synchronized by `state` (note: required contract comment).
/// Worker queues are also used to coordinate scheduler quiescence.
pub(crate) struct WorkerQueues<const CAPACITY: usize> {
    local_stealers: [ChaseLevStealer<ScheduledJob>; PRIORITY_LEVELS],
    injector: moirai_utils::queue::LockFreeQueue<(Priority, ScheduledJob)>,
    /// Advisory fast-path count used to skip checking when the queues are visibly
    /// empty. The owner writes it on every push/pop and thieves write it on every
    /// `steal_batch`, so it is cache-line isolated to keep those cross-thread RMWs
    /// from false-sharing with the multi-producer `injector` metadata above.
    len: CacheAligned<AtomicUsize>,
}

/// Unique bottom-side queue capabilities owned by one worker thread.
pub(crate) struct WorkerQueueOwner<const CAPACITY: usize> {
    local_queues: [ChaseLevDeque<ScheduledJob>; PRIORITY_LEVELS],
    shared: Arc<WorkerQueues<CAPACITY>>,
}

impl<const CAPACITY: usize> WorkerQueues<CAPACITY> {
    /// Create empty queues for one worker.
    pub(crate) fn new() -> (WorkerQueueOwner<CAPACITY>, Arc<Self>) {
        let local_queues = std::array::from_fn(|_| ChaseLevDeque::new(CAPACITY));
        let local_stealers = std::array::from_fn(|index| local_queues[index].stealer());
        let shared = Arc::new(Self {
            local_stealers,
            injector: moirai_utils::queue::LockFreeQueue::with_capacity(INJECTOR_CAPACITY),
            len: CacheAligned::new(AtomicUsize::new(0)),
        });
        (
            WorkerQueueOwner {
                local_queues,
                shared: Arc::clone(&shared),
            },
            shared,
        )
    }

    /// Push a job from an external thread (non-local push).
    pub(crate) fn try_push_external(
        &self,
        priority: Priority,
        job: ScheduledJob,
    ) -> Option<ScheduledJob> {
        match self.injector.try_enqueue((priority, job)) {
            Ok(()) => {
                self.len.fetch_add(1, Ordering::Relaxed);
                None
            }
            Err((_priority, job)) => Some(job),
        }
    }

    /// Steal one job without acquiring a bottom-side owner capability.
    pub(crate) fn steal_one(&self) -> Option<ScheduledJob> {
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }
        for &index in &PRIORITY_POP_ORDER {
            loop {
                match self.local_stealers[index].steal() {
                    StealResult::Success(job) => {
                        self.len.fetch_sub(1, Ordering::Relaxed);
                        return Some(job);
                    }
                    StealResult::Retry => continue,
                    StealResult::Empty => break,
                }
            }
        }
        if let Some((_priority, job)) = self.injector.try_dequeue() {
            self.len.fetch_sub(1, Ordering::Relaxed);
            return Some(job);
        }
        None
    }

    /// Returns true when the queue has no visible jobs.
    pub(crate) fn is_empty(&self) -> bool {
        self.len.load(Ordering::Relaxed) == 0
    }

    /// Approximate queued job count.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Initial slot count of the multi-producer injector queue.
    #[cfg(test)]
    pub(crate) fn injector_capacity(&self) -> usize {
        self.injector.capacity()
    }
}

impl<const CAPACITY: usize> WorkerQueueOwner<CAPACITY> {
    pub(crate) fn pop_local(&mut self) -> Option<ScheduledJob> {
        if self.shared.len.load(Ordering::Relaxed) == 0 {
            return None;
        }
        for &index in &PRIORITY_POP_ORDER {
            if let Some(job) = self.local_queues[index].pop() {
                self.shared.len.fetch_sub(1, Ordering::Relaxed);
                return Some(job);
            }
        }
        while let Some((priority, job)) = self.shared.injector.try_dequeue() {
            self.local_queues[priority.index()].push(job);
        }
        for &index in &PRIORITY_POP_ORDER {
            if let Some(job) = self.local_queues[index].pop() {
                self.shared.len.fetch_sub(1, Ordering::Relaxed);
                return Some(job);
            }
        }
        None
    }

    /// Steal multiple jobs from another worker, retaining all but one locally.
    pub(crate) fn steal_batch(&mut self, target: &WorkerQueues<CAPACITY>) -> Option<ScheduledJob> {
        if target.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        // 1. Try to steal from target's local queues
        for &index in &PRIORITY_POP_ORDER {
            loop {
                match target.local_stealers[index].steal_batch() {
                    StealResult::Success(mut batch) => {
                        let first_job = batch
                            .next()
                            .expect("invariant: successful batch contains one job");
                        let mut pushed_count = 0;
                        for job in batch {
                            self.local_queues[index].push(job);
                            pushed_count += 1;
                        }
                        if pushed_count > 0 {
                            self.shared.len.fetch_add(pushed_count, Ordering::Relaxed);
                        }
                        target.len.fetch_sub(pushed_count + 1, Ordering::Relaxed);
                        return Some(first_job);
                    }
                    StealResult::Retry => continue,
                    StealResult::Empty => break,
                }
            }
        }

        // 2. Try to steal from target's injector
        if let Some((_priority, first_job)) = target.injector.try_dequeue() {
            let mut pushed_count = 0;
            // Dequeue a batch (up to 15 more tasks to form a batch of 16)
            while pushed_count < 15 {
                if let Some((p, job)) = target.injector.try_dequeue() {
                    self.local_queues[p.index()].push(job);
                    pushed_count += 1;
                } else {
                    break;
                }
            }
            if pushed_count > 0 {
                self.shared.len.fetch_add(pushed_count, Ordering::Relaxed);
            }
            target.len.fetch_sub(pushed_count + 1, Ordering::Relaxed);
            return Some(first_job);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::WorkerQueues;
    use crate::schedule::job::ScheduledJob;
    use moirai_core::Priority;

    #[test]
    fn worker_queue_pops_highest_priority_first() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let (mut owner, queues) = WorkerQueues::<256>::new();

        for (priority, value) in [(Priority::Low, 1), (Priority::Critical, 2)] {
            let observed = Arc::clone(&observed);
            let () = queues
                .try_push_external(
                    priority,
                    ScheduledJob::new(move |_| {
                        observed.lock().unwrap().push(value);
                    }),
                )
                .map_or((), |_| panic!("test queue has capacity"));
        }

        owner.pop_local().unwrap().execute(0);
        owner.pop_local().unwrap().execute(0);

        assert_eq!(*observed.lock().unwrap(), vec![2, 1]);
        assert_eq!(queues.len(), 0);
    }

    #[test]
    fn injector_uses_sane_default_capacity_not_65536() {
        // The injector must not pre-allocate the LockFreeQueue 65536-slot
        // default (~16 MiB/worker for (Priority, ScheduledJob)); it is sized to
        // INJECTOR_CAPACITY instead.
        let (_owner, queues) = WorkerQueues::<256>::new();
        assert_eq!(queues.injector_capacity(), super::INJECTOR_CAPACITY);
        assert_eq!(queues.injector_capacity(), 1024);
        assert_ne!(queues.injector_capacity(), 65536);
    }

    #[test]
    fn injector_round_trips_through_external_push() {
        // The reduced-capacity injector still enqueues and drains: an external
        // push lands in the injector and pops out via pop_local's drain path.
        let observed = Arc::new(Mutex::new(Vec::new()));
        let (mut owner, queues) = WorkerQueues::<256>::new();

        for (priority, value) in [(Priority::Normal, 7), (Priority::Critical, 9)] {
            let observed = Arc::clone(&observed);
            let () = queues
                .try_push_external(
                    priority,
                    ScheduledJob::new(move |_| {
                        observed.lock().unwrap().push(value);
                    }),
                )
                .map_or((), |_| panic!("test queue has capacity"));
        }

        // Critical drains ahead of Normal once moved into the local queues.
        owner.pop_local().unwrap().execute(0);
        owner.pop_local().unwrap().execute(0);

        assert_eq!(*observed.lock().unwrap(), vec![9, 7]);
        assert_eq!(queues.len(), 0);
    }

    #[test]
    fn full_injector_returns_and_drops_rejected_job_once() {
        let (_owner, queues) = WorkerQueues::<256>::new();
        for _ in 0..super::INJECTOR_CAPACITY {
            let () = queues
                .try_push_external(Priority::Normal, ScheduledJob::new(|_| {}))
                .map_or((), |_| panic!("capacity-sized admission must succeed"));
        }

        let capture = Arc::new(());
        let rejected_capture = Arc::clone(&capture);
        let rejected = queues
            .try_push_external(
                Priority::Normal,
                ScheduledJob::new(move |_| drop(rejected_capture)),
            )
            .expect("one job beyond capacity must be rejected");

        assert_eq!(queues.len(), super::INJECTOR_CAPACITY);
        assert_eq!(Arc::strong_count(&capture), 2);
        drop(rejected);
        assert_eq!(Arc::strong_count(&capture), 1);
    }
}
