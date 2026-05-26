//! Priority-aware worker queues.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex, MutexGuard,
    },
};

use moirai_core::Priority;

use super::job::ScheduledJob;

const PRIORITY_LEVELS: usize = 4;
const CRITICAL_INDEX: usize = 3;
const HIGH_INDEX: usize = 2;
const NORMAL_INDEX: usize = 1;
const LOW_INDEX: usize = 0;
const PRIORITY_POP_ORDER: [usize; PRIORITY_LEVELS] =
    [CRITICAL_INDEX, HIGH_INDEX, NORMAL_INDEX, LOW_INDEX];

/// Per-worker task queues partitioned by priority.
pub(crate) struct WorkerQueues {
    state: Mutex<QueueState>,
    /// Advisory fast-path count used to skip locking when a queue is visibly
    /// empty. Queue contents are synchronized by `state`; scheduler quiescence
    /// is synchronized by global pending/active counters.
    len: AtomicUsize,
}

struct QueueState {
    queues: [VecDeque<ScheduledJob>; PRIORITY_LEVELS],
    ready_mask: u8,
}

struct QueueAccess<'queue> {
    state: MutexGuard<'queue, QueueState>,
}

impl WorkerQueues {
    /// Create empty queues for one worker.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                queues: std::array::from_fn(|_| VecDeque::new()),
                ready_mask: 0,
            }),
            len: AtomicUsize::new(0),
        }
    }

    /// Push a job into its priority queue.
    pub(crate) fn push(&self, priority: Priority, job: ScheduledJob) {
        self.access().push(priority, job);
        self.len.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop local work, highest priority first.
    pub(crate) fn pop_local(&self) -> Option<ScheduledJob> {
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        let job = self.access().pop_front();
        if job.is_some() {
            self.len.fetch_sub(1, Ordering::Relaxed);
        }

        job
    }

    /// Steal older work from another worker, highest priority first.
    pub(crate) fn steal(&self) -> Option<ScheduledJob> {
        if self.len.load(Ordering::Relaxed) == 0 {
            return None;
        }

        let job = self.access().pop_back();
        if job.is_some() {
            self.len.fetch_sub(1, Ordering::Relaxed);
        }

        job
    }

    fn access(&self) -> QueueAccess<'_> {
        QueueAccess {
            state: lock_queue(&self.state),
        }
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
}

impl QueueAccess<'_> {
    fn push(&mut self, priority: Priority, job: ScheduledJob) {
        let index = priority_index(priority);
        self.state.queues[index].push_back(job);
        self.state.ready_mask |= priority_bit(index);
    }

    fn pop_front(&mut self) -> Option<ScheduledJob> {
        let index = highest_ready_priority(self.state.ready_mask)?;
        let job = self.state.queues[index].pop_front();
        self.clear_empty_priority(index);
        job
    }

    fn pop_back(&mut self) -> Option<ScheduledJob> {
        let index = highest_ready_priority(self.state.ready_mask)?;
        let job = self.state.queues[index].pop_back();
        self.clear_empty_priority(index);
        job
    }

    fn clear_empty_priority(&mut self, index: usize) {
        if self.state.queues[index].is_empty() {
            self.state.ready_mask &= !priority_bit(index);
        }
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

fn priority_bit(index: usize) -> u8 {
    1_u8 << index
}

fn highest_ready_priority(mask: u8) -> Option<usize> {
    PRIORITY_POP_ORDER
        .into_iter()
        .find(|&index| mask & priority_bit(index) != 0)
}

fn lock_queue<T>(queue: &Mutex<T>) -> MutexGuard<'_, T> {
    queue
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
        let queues = WorkerQueues::new();

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
