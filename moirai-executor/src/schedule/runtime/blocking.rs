//! Bounded worker lane for potentially blocking jobs.

use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
};

use moirai_core::{
    error::{ExecutorError, ExecutorResult},
    Priority,
};

use super::{super::job::ScheduledJob, types::SchedulerInner, worker::execute_blocking_job};

const PRIORITY_LEVELS: usize = 4;

/// One bounded queue owned by one blocking worker.
struct BlockingQueue {
    state: Mutex<BlockingQueueState>,
    wake: Condvar,
    capacity: usize,
}

struct BlockingQueueState {
    jobs: [VecDeque<ScheduledJob>; PRIORITY_LEVELS],
    length: usize,
    closed: bool,
}

impl BlockingQueue {
    fn new(capacity: usize) -> Self {
        let per_priority_capacity = capacity.div_ceil(PRIORITY_LEVELS).max(1);
        Self {
            state: Mutex::new(BlockingQueueState {
                jobs: std::array::from_fn(|_| VecDeque::with_capacity(per_priority_capacity)),
                length: 0,
                closed: false,
            }),
            wake: Condvar::new(),
            capacity,
        }
    }

    fn try_push(
        &self,
        priority: Priority,
        job: &mut Option<ScheduledJob>,
        pending_tasks: &AtomicUsize,
    ) -> Result<(), BlockingAdmission> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.closed {
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
        self.wake.notify_one();
        Ok(())
    }

    fn pop(&self) -> Option<ScheduledJob> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if let Some(job) = (0..PRIORITY_LEVELS)
                .rev()
                .find_map(|priority| state.jobs[priority].pop_front())
            {
                state.length -= 1;
                return Some(job);
            }
            if state.closed {
                return None;
            }
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn close(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.closed = true;
        self.wake.notify_all();
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
pub(super) struct BlockingLane<const QUEUE_CAPACITY: usize> {
    queues: Box<[Arc<BlockingQueue>]>,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

impl<const QUEUE_CAPACITY: usize> BlockingLane<QUEUE_CAPACITY> {
    pub(super) fn new(worker_count: usize) -> Self {
        let capacity = QUEUE_CAPACITY.max(1);
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
        inner: Arc<SchedulerInner<QUEUE_CAPACITY>>,
        thread_name_prefix: &str,
    ) -> ExecutorResult<()> {
        let mut handles = lock_mutex(&self.handles);
        for (lane_id, queue) in self.queues.iter().cloned().enumerate() {
            let inner = Arc::clone(&inner);
            let thread_name = format!("{thread_name_prefix}-blocking-{lane_id}");
            let handle = match thread::Builder::new().name(thread_name).spawn(move || {
                while let Some(job) = queue.pop() {
                    let worker_id = inner.workers.len() + lane_id;
                    execute_blocking_job(&inner, worker_id, job);
                }
            }) {
                Ok(handle) => handle,
                Err(_) => {
                    drop(handles);
                    self.shutdown();
                    return Err(ExecutorError::ThreadPoolCreationFailed);
                }
            };
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
    ) -> ExecutorResult<()> {
        let lane_id = locality_hint.unwrap_or_else(next_lane_ticket) % self.queues.len();
        // `try_push` leaves a refused job in the slot rather than consuming it:
        // it never ran, and its owner may still run it.
        match self.queues[lane_id].try_push(priority, job, pending_tasks) {
            Ok(()) => Ok(()),
            Err(BlockingAdmission::Full) => Err(ExecutorError::ResourceExhausted(format!(
                "blocking lane {lane_id} admission queue is full"
            ))),
            Err(BlockingAdmission::ShuttingDown) => Err(ExecutorError::ShuttingDown),
        }
    }

    pub(super) fn shutdown(&self) {
        for queue in &self.queues {
            queue.close();
        }
        let mut handles = lock_mutex(&self.handles);
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }
}

impl<const QUEUE_CAPACITY: usize> Drop for BlockingLane<QUEUE_CAPACITY> {
    fn drop(&mut self) {
        self.shutdown();
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
