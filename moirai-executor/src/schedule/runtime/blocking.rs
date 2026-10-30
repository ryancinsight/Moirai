//! Bounded worker lane for potentially blocking jobs.

use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
};

use moirai_core::{
    Priority,
    error::{ExecutorError, ExecutorResult},
};

use super::{
    super::job::ScheduledJob,
    types::SchedulerInner,
    worker::{execute_blocking_job, join_other_threads},
};

const PRIORITY_LEVELS: usize = 4;

/// One bounded queue owned by one blocking worker.
struct BlockingQueue {
    state: Mutex<BlockingQueueState>,
    closed: AtomicBool,
    capacity: usize,
    thread: OnceLock<thread::Thread>,
    #[cfg(test)]
    park_gate: Mutex<Option<Arc<(std::sync::Barrier, std::sync::Barrier)>>>,
}

struct BlockingQueueState {
    jobs: [VecDeque<ScheduledJob>; PRIORITY_LEVELS],
    length: usize,
}

impl BlockingQueue {
    fn new(capacity: usize) -> Self {
        let per_priority_capacity = capacity.div_ceil(PRIORITY_LEVELS).max(1);
        Self {
            state: Mutex::new(BlockingQueueState {
                jobs: std::array::from_fn(|_| VecDeque::with_capacity(per_priority_capacity)),
                length: 0,
            }),
            closed: AtomicBool::new(false),
            capacity,
            thread: OnceLock::new(),
            #[cfg(test)]
            park_gate: Mutex::new(None),
        }
    }

    fn try_push(
        &self,
        priority: Priority,
        job: &mut Option<ScheduledJob>,
        pending_tasks: &AtomicUsize,
        shutdown: &AtomicBool,
    ) -> Result<(), BlockingAdmission> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.closed.load(Ordering::Acquire) || shutdown.load(Ordering::SeqCst) {
            return Err(BlockingAdmission::ShuttingDown);
        }
        if state.length == self.capacity {
            return Err(BlockingAdmission::Full);
        }

        // Publish pending before making the job visible to the receiver. The
        // worker decrements this counter before invoking the closure.
        pending_tasks.fetch_add(1, Ordering::SeqCst);
        state.jobs[priority.index()].push_back(
            job.take()
                .expect("invariant: job is present before admission"),
        );
        state.length += 1;
        drop(state);
        if let Some(worker) = self.thread.get() {
            worker.unpark();
        }
        Ok(())
    }

    fn pop(&self) -> Option<ScheduledJob> {
        loop {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(job) = (0..PRIORITY_LEVELS)
                .rev()
                .find_map(|priority| state.jobs[priority].pop_front())
            {
                state.length -= 1;
                return Some(job);
            }
            if self.closed.load(Ordering::Acquire) {
                return None;
            }
            drop(state);
            #[cfg(test)]
            if let Some(gate) = lock_mutex(&self.park_gate).take() {
                gate.0.wait();
                gate.1.wait();
            }
            thread::park();
        }
    }

    fn request_close(&self) {
        self.closed.store(true, Ordering::Release);
        if let Some(worker) = self.thread.get() {
            worker.unpark();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockingAdmission {
    Full,
    ShuttingDown,
}

/// Dedicated bounded lane for [`BlockingTask`](crate::schedule::BlockingTask).
///
/// Each queue has one producer-side mutex and one consumer. The lock is not on
/// the compute-worker hot path, and the per-worker split avoids a global
/// admission bottleneck. After lane initialization, submissions move jobs
/// into a selected queue without serialization or cloning; queue storage is
/// bounded by the admission counter.
pub(super) struct BlockingLane<const BLOCKING_QUEUE_CAPACITY: usize> {
    queues: Box<[Arc<BlockingQueue>]>,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

impl<const BLOCKING_QUEUE_CAPACITY: usize> BlockingLane<BLOCKING_QUEUE_CAPACITY> {
    pub(super) fn new(worker_count: usize) -> Self {
        let capacity = BLOCKING_QUEUE_CAPACITY.max(1);
        let queues = (0..worker_count)
            .map(|_| Arc::new(BlockingQueue::new(capacity)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            queues,
            handles: Mutex::new(Vec::with_capacity(worker_count)),
        }
    }

    pub(super) fn start(
        &self,
        inner: Arc<SchedulerInner<BLOCKING_QUEUE_CAPACITY>>,
        thread_name_prefix: &str,
    ) -> ExecutorResult<()> {
        let mut handles = lock_mutex(&self.handles);
        for (lane_id, queue) in self.queues.iter().enumerate() {
            let inner = Arc::clone(&inner);
            let receiver = Arc::clone(queue);
            let thread_name = format!("{thread_name_prefix}-blocking-{lane_id}");
            let handle = match thread::Builder::new().name(thread_name).spawn(move || {
                while let Some(job) = receiver.pop() {
                    let worker_id = inner.workers.len() + lane_id;
                    execute_blocking_job(&inner, worker_id, job);
                }
            }) {
                Ok(handle) => handle,
                Err(_) => {
                    drop(handles);
                    self.request_close();
                    self.join();
                    return Err(ExecutorError::ThreadPoolCreationFailed);
                }
            };
            queue
                .thread
                .set(handle.thread().clone())
                .expect("invariant: blocking worker thread registers exactly once");
            handles.push(handle);
        }
        Ok(())
    }

    pub(super) fn submit(
        &self,
        priority: Priority,
        locality_hint: Option<usize>,
        job: &mut Option<ScheduledJob>,
        pending_tasks: &AtomicUsize,
        shutdown: &AtomicBool,
    ) -> ExecutorResult<()> {
        let lane_id = locality_hint.unwrap_or_else(next_lane_ticket) % self.queues.len();
        // `try_push` leaves a refused job in the slot rather than consuming it:
        // it never ran, and its owner may still run it.
        match self.queues[lane_id].try_push(priority, job, pending_tasks, shutdown) {
            Ok(()) => Ok(()),
            Err(BlockingAdmission::Full) => Err(ExecutorError::ResourceExhausted(format!(
                "blocking lane {lane_id} admission queue is full"
            ))),
            Err(BlockingAdmission::ShuttingDown) => Err(ExecutorError::ShuttingDown),
        }
    }

    pub(super) fn request_close(&self) {
        for queue in &self.queues {
            queue.request_close();
        }
    }

    pub(super) fn join(&self) {
        let mut handles = std::mem::take(&mut *lock_mutex(&self.handles));
        join_other_threads(&mut handles);
    }

    pub(super) fn is_current_worker(&self) -> bool {
        let current = thread::current().id();
        self.queues.iter().any(|queue| {
            queue
                .thread
                .get()
                .is_some_and(|worker| worker.id() == current)
        })
    }

    #[cfg(test)]
    pub(super) fn hold_queue_lock_until(
        &self,
        reached: std::sync::mpsc::SyncSender<()>,
        release: std::sync::mpsc::Receiver<()>,
    ) {
        let _queue = lock_mutex(&self.queues[0].state);
        reached
            .send(())
            .expect("test controller must observe the held queue lock");
        release
            .recv()
            .expect("test controller must release the held queue lock");
    }
}

impl<const BLOCKING_QUEUE_CAPACITY: usize> Drop for BlockingLane<BLOCKING_QUEUE_CAPACITY> {
    fn drop(&mut self) {
        self.request_close();
    }
}

fn next_lane_ticket() -> usize {
    use std::cell::Cell;
    std::thread_local! {
        // clippy 1.97.0 FP: already const. ATLAS-MNEMOSYNE-CI-1.
        #[allow(clippy::missing_const_for_thread_local)]
        static TICKET: Cell<usize> = const { Cell::new(0) };
    }
    TICKET.with(|cell| {
        let ticket = cell.get();
        cell.set(ticket.wrapping_add(1));
        ticket
    })
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_between_empty_check_and_park_is_observed() {
        let queue = Arc::new(BlockingQueue::new(1));
        let gate = Arc::new((std::sync::Barrier::new(2), std::sync::Barrier::new(2)));
        *lock_mutex(&queue.park_gate) = Some(Arc::clone(&gate));

        let receiver = Arc::clone(&queue);
        let worker = thread::spawn(move || receiver.pop());
        queue
            .thread
            .set(worker.thread().clone())
            .expect("invariant: test worker thread registers exactly once");

        gate.0.wait();
        queue.request_close();
        gate.1.wait();
        assert!(
            worker
                .join()
                .expect("blocking queue worker must not panic")
                .is_none(),
            "closure must wake the worker after its last empty check"
        );
    }
}
