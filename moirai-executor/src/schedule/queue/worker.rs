//! Per-worker queue state: the shared stealer side and its single owner.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use moirai_core::Priority;
use moirai_scheduler::{ChaseLevDeque, ChaseLevStealer, DequeCapacity};
use moirai_utils::CacheAligned;

use super::priority::{PRIORITY_LEVELS, PRIORITY_POP_ORDER};
use super::steal::steal_after_contention;
use crate::schedule::job::ScheduledJob;

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
                match target.injector.try_dequeue() {
                    Some((p, job)) => {
                        self.local_queues[p.index()].push(job);
                        pushed_count += 1;
                    }
                    _ => {
                        break;
                    }
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
