//! Priority-aware worker queues.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use moirai_core::Priority;
use moirai_scheduler::{ChaseLevDeque, ChaseLevStealer, DequeCapacity, StealResult};
use moirai_utils::CacheAligned;

use super::job::ScheduledJob;

/// One queue per priority level; indices come from [`Priority::index`] (SSOT).
const PRIORITY_LEVELS: usize = Priority::Critical.index() + 1;
/// Lost-race processor hints emitted before yielding to another runnable thread.
///
/// This matches the established upper handoff window used by Moirai's
/// contended spin lock while keeping steal retries allocation- and sleep-free.
const STEAL_SPINS_BEFORE_YIELD: usize = 1_000;
/// Pop scan order: highest [`Priority::index`] first.
const PRIORITY_POP_ORDER: [usize; PRIORITY_LEVELS] = [
    Priority::Critical.index(),
    Priority::High.index(),
    Priority::Normal.index(),
    Priority::Low.index(),
];

#[inline]
fn steal_after_contention<T>(steal: impl FnMut() -> StealResult<T>) -> Option<T> {
    steal_after_contention_with(steal, std::hint::spin_loop, std::thread::yield_now)
}

#[inline]
fn steal_after_contention_with<T>(
    mut steal: impl FnMut() -> StealResult<T>,
    mut spin: impl FnMut(),
    mut yield_now: impl FnMut(),
) -> Option<T> {
    let mut spins = 0usize;
    loop {
        match steal() {
            StealResult::Success(value) => return Some(value),
            StealResult::Empty => return None,
            StealResult::Retry if spins < STEAL_SPINS_BEFORE_YIELD => {
                spins += 1;
                spin();
            }
            StealResult::Retry => {
                spins = 0;
                yield_now();
            }
        }
    }
}

/// Per-worker task queues partitioned by priority using lock-free Chase-Lev deques.
///
/// Local pop operations and push operations are lock-free. Local operations (from the owner thread)
/// proceed directly on the private SPSC deques, while non-local schedules place tasks into the
/// lock-free multi-producer injector queue. Steal operations are entirely lock-free and proceed
/// without acquiring any locks.
///
/// Queue contents are synchronized by `state` (note: required contract comment).
/// Worker queues are also used to coordinate scheduler quiescence.
pub(crate) struct WorkerQueues {
    local_stealers: [ChaseLevStealer<ScheduledJob>; PRIORITY_LEVELS],
    injector: moirai_utils::queue::LockFreeQueue<(Priority, ScheduledJob)>,
    /// Advisory fast-path count used to skip checking when the queues are visibly
    /// empty. The owner writes it on every push/pop and thieves write it on every
    /// `steal_batch`, so it is cache-line isolated to keep those cross-thread RMWs
    /// from false-sharing with the multi-producer `injector` metadata above.
    len: CacheAligned<AtomicUsize>,
}

/// Unique bottom-side queue capabilities owned by one worker thread.
pub(crate) struct WorkerQueueOwner {
    local_queues: [ChaseLevDeque<ScheduledJob>; PRIORITY_LEVELS],
    shared: Arc<WorkerQueues>,
}

impl WorkerQueues {
    /// Create empty queues for one worker.
    pub(crate) fn new(
        injector_capacity: usize,
        local_queue_capacity: DequeCapacity<ScheduledJob>,
    ) -> (WorkerQueueOwner, Arc<Self>) {
        // Only the default-priority plane starts at the configured capacity.
        //
        // Every plane is the same size, so the retained storage is
        // `priority levels x capacity`, but a workload's pushes are not spread
        // over the planes: a submission carries one priority, and a consumer
        // that never sets one uses the default plane exclusively. Measured by
        // counting first pushes per plane, Apollo's chunked transforms touch
        // only the default plane, and across this workspace's own suite the
        // default plane is touched by 85 test processes against 4, 4 and 7 for
        // the other three.
        //
        // The three unused planes are not free: the payload is eager and
        // exactly `capacity x size_of::<ScheduledJob>()`, so at the 128-slot
        // default they retain 16,384 bytes each, 49,152 bytes per worker.
        // Starting them at the minimum keeps the busy plane's measured policy
        // (ADR 0035) untouched while the others pay 2,048 bytes and grow on the
        // owner's push if work does arrive -- the same resize the algorithm
        // already performs, and the same trade ADR 0035 accepted when it took
        // the default from 256 to 128.
        let default_plane = Priority::default().index();
        let local_queues = std::array::from_fn(|plane| {
            ChaseLevDeque::new(if plane == default_plane {
                local_queue_capacity
            } else {
                DequeCapacity::minimum()
            })
        });
        let local_stealers = std::array::from_fn(|index| local_queues[index].stealer());
        let shared = Arc::new(Self {
            local_stealers,
            injector: moirai_utils::queue::LockFreeQueue::with_capacity(injector_capacity),
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
            if let Some(job) = steal_after_contention(|| self.local_stealers[index].steal()) {
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(job);
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

    /// Allocated slot count of each local priority plane, read from the deques.
    ///
    /// Reads the live arrays through the stealers rather than a recorded
    /// intent, so a test of the per-plane policy fails when the construction
    /// changes.
    #[cfg(test)]
    pub(crate) fn local_queue_capacities(&self) -> [usize; PRIORITY_LEVELS] {
        std::array::from_fn(|plane| self.local_stealers[plane].capacity())
    }
}

impl WorkerQueueOwner {
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
    pub(crate) fn steal_batch(&mut self, target: &WorkerQueues) -> Option<ScheduledJob> {
        if target.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        // 1. Try to steal from target's local queues
        for &index in &PRIORITY_POP_ORDER {
            if let Some(mut batch) =
                steal_after_contention(|| target.local_stealers[index].steal_batch())
            {
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
    use core::mem::size_of;
    use std::{
        cell::Cell,
        sync::{Arc, Mutex},
    };

    use super::{steal_after_contention_with, WorkerQueues, STEAL_SPINS_BEFORE_YIELD};
    use crate::schedule::job::ScheduledJob;
    use moirai_core::Priority;
    use moirai_scheduler::{DequeCapacity, StealResult};

    const TEST_INJECTOR_CAPACITY: usize = 8;

    #[test]
    fn steal_contention_spins_yields_and_preserves_victim_priority() {
        let attempts = Cell::new(0usize);
        let spins = Cell::new(0usize);
        let yields = Cell::new(0usize);
        let retries = 2 * (STEAL_SPINS_BEFORE_YIELD + 1);

        let result = steal_after_contention_with(
            || {
                let attempt = attempts.get();
                attempts.set(attempt + 1);
                if attempt < retries {
                    StealResult::Retry
                } else {
                    StealResult::Success(7usize)
                }
            },
            || spins.set(spins.get() + 1),
            || yields.set(yields.get() + 1),
        );

        assert_eq!(result, Some(7));
        assert_eq!(attempts.get(), retries + 1);
        assert_eq!(spins.get(), 2 * STEAL_SPINS_BEFORE_YIELD);
        assert_eq!(yields.get(), 2);
    }

    fn local_capacity(requested: usize) -> DequeCapacity<ScheduledJob> {
        DequeCapacity::try_from(requested).expect("test capacity must be representable")
    }

    #[test]
    fn injector_payload_uses_seventeen_machine_words() {
        type InjectorPayload = (Priority, ScheduledJob);
        let expected_payload_bytes = 17 * size_of::<usize>();

        assert_eq!(size_of::<InjectorPayload>(), expected_payload_bytes);
        assert_eq!(size_of::<Option<InjectorPayload>>(), expected_payload_bytes);
    }

    #[test]
    fn worker_queue_pops_highest_priority_first() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let (mut owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(256));

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
    fn injector_uses_configured_capacity() {
        let (_owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(256));
        assert_eq!(queues.injector_capacity(), TEST_INJECTOR_CAPACITY);
    }

    #[test]
    fn local_queues_use_the_normalized_initial_capacity() {
        let (_owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(17));

        assert_eq!(
            queues.local_queue_capacities()[Priority::default().index()],
            32
        );
    }

    #[test]
    fn only_the_default_priority_plane_carries_the_configured_capacity() {
        // Retained local storage is `priority levels x capacity`, but a
        // consumer that never sets a priority uses one plane. The other three
        // start at the minimum and grow on push, so an unused plane costs
        // 2,048 bytes rather than 16,384 at the 128-slot default.
        let (_owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(128));
        let capacities = queues.local_queue_capacities();
        let minimum = DequeCapacity::<ScheduledJob>::minimum().get();

        for (plane, capacity) in capacities.iter().copied().enumerate() {
            if plane == Priority::default().index() {
                assert_eq!(capacity, 128, "default plane keeps the configured capacity");
            } else {
                assert_eq!(capacity, minimum, "plane {plane} starts at the minimum");
            }
        }
        assert!(
            minimum < 128,
            "the minimum must actually be smaller, or this policy saves nothing"
        );
    }

    #[test]
    fn a_non_default_plane_grows_past_its_minimum_initial_capacity() {
        // The saving is only sound because the deque grows on the owner's
        // push. Drive a minimum-capacity plane well past its initial slots
        // through the injector and require every job back.
        let minimum = DequeCapacity::<ScheduledJob>::minimum().get();
        let count = minimum * 4;
        let observed = Arc::new(Mutex::new(Vec::new()));
        let (mut owner, queues) = WorkerQueues::new(count * 2, local_capacity(128));

        for value in 0..count {
            let sink = Arc::clone(&observed);
            let () = queues
                .try_push_external(
                    Priority::Critical,
                    ScheduledJob::new(move |_| sink.lock().unwrap().push(value)),
                )
                .map_or((), |_| panic!("injector sized for the whole burst"));
        }

        let mut drained = 0;
        while let Some(job) = owner.pop_local() {
            job.execute(0);
            drained += 1;
        }

        assert_eq!(
            drained, count,
            "every job queued past the minimum initial capacity ran"
        );
        let mut values = observed.lock().unwrap().clone();
        values.sort_unstable();
        assert_eq!(values, (0..count).collect::<Vec<_>>());
    }

    #[test]
    fn injector_round_trips_through_external_push() {
        // The reduced-capacity injector still enqueues and drains: an external
        // push lands in the injector and pops out via pop_local's drain path.
        let observed = Arc::new(Mutex::new(Vec::new()));
        let (mut owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(256));

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
        let (_owner, queues) = WorkerQueues::new(TEST_INJECTOR_CAPACITY, local_capacity(256));
        for _ in 0..TEST_INJECTOR_CAPACITY {
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

        assert_eq!(queues.len(), TEST_INJECTOR_CAPACITY);
        assert_eq!(Arc::strong_count(&capture), 2);
        drop(rejected);
        assert_eq!(Arc::strong_count(&capture), 1);
    }
}
