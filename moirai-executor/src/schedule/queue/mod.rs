//! Priority-aware worker queues.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

use moirai_core::Priority;
use moirai_scheduler::{ChaseLevDeque, QuiescentReclaim, StealResult};

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
/// Local pop operations and push operations are serialized by a Mutex to support
/// multi-producer scheduling from arbitrary threads. Steal operations are entirely
/// lock-free and proceed without acquiring the mutex.
///
/// Queue contents are synchronized by `state` (note: required contract comment).
/// Worker queues are also used to coordinate scheduler quiescence.
pub(crate) struct WorkerQueues<const CAPACITY: usize> {
    queues: [ChaseLevDeque<ScheduledJob>; PRIORITY_LEVELS],
    lock: Mutex<()>,
    /// Advisory fast-path count used to skip locking when the queues are visibly
    /// empty.
    len: AtomicUsize,
}

impl<const CAPACITY: usize> WorkerQueues<CAPACITY> {
    /// Create empty queues for one worker.
    pub(crate) fn new() -> Self {
        Self {
            queues: std::array::from_fn(|_| ChaseLevDeque::new(CAPACITY)),
            lock: Mutex::new(()),
            len: AtomicUsize::new(0),
        }
    }

    /// Push a job into its priority queue.
    pub(crate) fn push(&self, priority: Priority, job: ScheduledJob) {
        let index = priority_index(priority);
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.queues[index].push(job);
        self.len.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop local work, highest priority first.
    pub(crate) fn pop_local(&self) -> Option<ScheduledJob> {
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        for &index in &PRIORITY_POP_ORDER {
            if let Some(job) = self.queues[index].pop() {
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(job);
            }
        }
        None
    }

    /// Steal older work from another worker, highest priority first.
    #[allow(dead_code)]
    pub(crate) fn steal(&self) -> Option<ScheduledJob> {
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        for &index in &PRIORITY_POP_ORDER {
            loop {
                match self.queues[index].steal() {
                    StealResult::Success(job) => {
                        self.len.fetch_sub(1, Ordering::Relaxed);
                        return Some(job);
                    }
                    StealResult::Retry => continue,
                    StealResult::Empty => break,
                }
            }
        }
        None
    }

    /// Steal multiple jobs from another worker's queues, highest priority first,
    /// pushing all but one into `self` and returning the remaining one.
    pub(crate) fn steal_batch(&self, target: &Self) -> Option<ScheduledJob> {
        if target.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        for &index in &PRIORITY_POP_ORDER {
            loop {
                let mut guard = None;
                let mut pushed_count = 0;
                let dest_queue = &self.queues[index];

                match target.queues[index].steal_batch_with(|job| {
                    if guard.is_none() {
                        guard = Some(self.lock.lock().unwrap_or_else(|e| e.into_inner()));
                    }
                    dest_queue.push(job);
                    pushed_count += 1;
                }) {
                    StealResult::Success(first_job) => {
                        if pushed_count > 0 {
                            self.len.fetch_add(pushed_count, Ordering::Relaxed);
                        }
                        drop(guard);
                        target.len.fetch_sub(pushed_count + 1, Ordering::Relaxed);
                        return Some(first_job);
                    }
                    StealResult::Retry => continue,
                    StealResult::Empty => break,
                }
            }
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
        for queue in &mut self.queues {
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
            queues.push(
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
