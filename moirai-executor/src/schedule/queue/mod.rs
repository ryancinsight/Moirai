//! Priority-aware worker queues.

use std::sync::atomic::{AtomicUsize, Ordering};

use moirai_core::Priority;
use moirai_scheduler::{ChaseLevDeque, QuiescentReclaim, StealResult};
use moirai_utils::CacheAligned;

use super::job::ScheduledJob;

const PRIORITY_LEVELS: usize = 4;
const CRITICAL_INDEX: usize = 3;
const HIGH_INDEX: usize = 2;
const NORMAL_INDEX: usize = 1;
const LOW_INDEX: usize = 0;
const PRIORITY_POP_ORDER: [usize; PRIORITY_LEVELS] =
    [CRITICAL_INDEX, HIGH_INDEX, NORMAL_INDEX, LOW_INDEX];

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
    local_queues: [ChaseLevDeque<ScheduledJob>; PRIORITY_LEVELS],
    injector: moirai_utils::queue::LockFreeQueue<(Priority, ScheduledJob)>,
    /// Advisory fast-path count used to skip checking when the queues are visibly
    /// empty. The owner writes it on every push/pop and thieves write it on every
    /// `steal_batch`, so it is cache-line isolated to keep those cross-thread RMWs
    /// from false-sharing with the multi-producer `injector` metadata above.
    len: CacheAligned<AtomicUsize>,
}

impl<const CAPACITY: usize> WorkerQueues<CAPACITY> {
    /// Create empty queues for one worker.
    pub(crate) fn new() -> Self {
        Self {
            local_queues: std::array::from_fn(|_| ChaseLevDeque::new(CAPACITY)),
            injector: moirai_utils::queue::LockFreeQueue::new(),
            len: CacheAligned::new(AtomicUsize::new(0)),
        }
    }

    /// Push a job from the owner thread (local push).
    pub(crate) fn push_local(&self, priority: Priority, job: ScheduledJob) {
        let index = priority_index(priority);
        self.local_queues[index].push(job);
        self.len.fetch_add(1, Ordering::Relaxed);
    }

    /// Push a job from an external thread (non-local push).
    pub(crate) fn push_external(&self, priority: Priority, job: ScheduledJob) {
        self.injector.enqueue((priority, job));
        self.len.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop local work, highest priority first.
    pub(crate) fn pop_local(&self) -> Option<ScheduledJob> {
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        // First, check private local queues
        for &index in &PRIORITY_POP_ORDER {
            if let Some(job) = self.local_queues[index].pop() {
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(job);
            }
        }

        // If local queues are empty, try to drain the injector queue
        self.drain_injector();

        // Try local queues again
        for &index in &PRIORITY_POP_ORDER {
            if let Some(job) = self.local_queues[index].pop() {
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(job);
            }
        }

        None
    }

    fn drain_injector(&self) {
        while let Some((priority, job)) = self.injector.try_dequeue() {
            let index = priority_index(priority);
            self.local_queues[index].push(job);
        }
    }

    /// Steal older work from another worker, highest priority first.
    #[allow(dead_code)]
    pub(crate) fn steal(&self) -> Option<ScheduledJob> {
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        // 1. Try to steal from target's local queues
        for &index in &PRIORITY_POP_ORDER {
            loop {
                match self.local_queues[index].steal() {
                    StealResult::Success(job) => {
                        self.len.fetch_sub(1, Ordering::Relaxed);
                        return Some(job);
                    }
                    StealResult::Retry => continue,
                    StealResult::Empty => break,
                }
            }
        }

        // 2. Try to steal from target's injector
        if let Some((_, job)) = self.injector.try_dequeue() {
            self.len.fetch_sub(1, Ordering::Relaxed);
            return Some(job);
        }

        None
    }

    /// Steal multiple jobs from another worker's queues, highest priority first,
    /// pushing all but one into `self` and returning the remaining one.
    pub(crate) fn steal_batch(&self, target: &Self) -> Option<ScheduledJob> {
        if target.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        // 1. Try to steal from target's local queues
        for &index in &PRIORITY_POP_ORDER {
            loop {
                let mut pushed_count = 0;
                let dest_queue = &self.local_queues[index];

                match target.local_queues[index].steal_batch_with(|job| {
                    dest_queue.push(job);
                    pushed_count += 1;
                }) {
                    StealResult::Success(first_job) => {
                        if pushed_count > 0 {
                            self.len.fetch_add(pushed_count, Ordering::Relaxed);
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
                    let dest_queue = &self.local_queues[priority_index(p)];
                    dest_queue.push(job);
                    pushed_count += 1;
                } else {
                    break;
                }
            }
            if pushed_count > 0 {
                self.len.fetch_add(pushed_count, Ordering::Relaxed);
            }
            target.len.fetch_sub(pushed_count + 1, Ordering::Relaxed);
            return Some(first_job);
        }

        None
    }

    /// Returns true when the queue has no visible jobs.
    pub(crate) fn is_empty(&self) -> bool {
        self.len.load(Ordering::Relaxed) == 0
    }

    /// Deallocate retired backing arrays through an exclusive quiescent access path.
    #[allow(dead_code)]
    pub(crate) fn reclaim_memory(&mut self) {
        for queue in &mut self.local_queues {
            queue.reclaim_memory(QuiescentReclaim);
        }
    }

    /// Approximate queued job count.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}

fn priority_index(priority: Priority) -> usize {
    match priority {
        Priority::Low => LOW_INDEX,
        Priority::Normal => NORMAL_INDEX,
        Priority::High => HIGH_INDEX,
        Priority::Critical => CRITICAL_INDEX,
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
        let queues = WorkerQueues::<256>::new();

        for (priority, value) in [(Priority::Low, 1), (Priority::Critical, 2)] {
            let observed = Arc::clone(&observed);
            queues.push_local(
                priority,
                ScheduledJob::new(move |_| {
                    observed.lock().unwrap().push(value);
                }),
            );
        }

        queues.pop_local().unwrap().execute(0);
        queues.pop_local().unwrap().execute(0);

        assert_eq!(*observed.lock().unwrap(), vec![2, 1]);
        assert_eq!(queues.len(), 0);
    }
}
