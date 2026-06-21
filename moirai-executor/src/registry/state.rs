use std::{
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use super::super::task::TaskMetadata;

pub(crate) const NO_WORKER: usize = usize::MAX;
pub(crate) const TIMESTAMP_NOT_RECORDED: u64 = u64::MAX;
pub(crate) const TASK_STATE_BLOCK_SIZE: usize = 1024;

#[derive(Debug)]
pub(super) struct TaskStateBlock {
    pub(super) slots: Box<[Option<TaskState>]>,
}

/// Shared lifecycle state for one task.
pub(crate) struct TaskState {
    pub(crate) created_at: Instant,
    pub(super) started_after_ns: AtomicU64,
    pub(super) completed_after_ns: AtomicU64,
    pub(super) worker_id: AtomicUsize,
    pub(super) waker: std::sync::Mutex<Option<std::task::Waker>>,
}

impl std::fmt::Debug for TaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskState")
            .field("created_at", &self.created_at)
            .field("started_after_ns", &self.started_after_ns)
            .field("completed_after_ns", &self.completed_after_ns)
            .field("worker_id", &self.worker_id)
            .field("waker_registered", &self.waker.lock().unwrap().is_some())
            .finish()
    }
}

impl TaskState {
    #[inline]
    pub(super) fn new() -> Self {
        Self {
            created_at: Instant::now(),
            started_after_ns: AtomicU64::new(TIMESTAMP_NOT_RECORDED),
            completed_after_ns: AtomicU64::new(TIMESTAMP_NOT_RECORDED),
            worker_id: AtomicUsize::new(NO_WORKER),
            waker: std::sync::Mutex::new(None),
        }
    }

    #[inline]
    pub(super) fn mark_started(&self, worker_id: usize) -> u64 {
        let started_after_ns = elapsed_nanos_since(self.created_at);
        self.started_after_ns
            .store(started_after_ns, Ordering::Release);
        self.worker_id.store(worker_id, Ordering::Release);
        started_after_ns
    }

    #[inline]
    pub(super) fn mark_completed_since(&self, started_after_ns: u64) -> Duration {
        let completed_after_ns = elapsed_nanos_since(self.created_at);
        self.completed_after_ns
            .store(completed_after_ns, Ordering::Release);

        debug_assert!(
            completed_after_ns >= started_after_ns,
            "monotonic lifecycle completion offset must not precede start offset"
        );

        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }

        Duration::from_nanos(completed_after_ns - started_after_ns)
    }

    pub(super) fn mark_completed(&self) {
        let started_after_ns = self.started_after_ns.load(Ordering::Acquire);
        let started_after_ns = if started_after_ns == TIMESTAMP_NOT_RECORDED {
            elapsed_nanos_since(self.created_at)
        } else {
            started_after_ns
        };
        self.mark_completed_since(started_after_ns);
    }

    pub(super) fn is_completed(&self) -> bool {
        self.completed_after_ns.load(Ordering::Acquire) != TIMESTAMP_NOT_RECORDED
    }

    pub(super) fn completed_at(&self) -> Option<Instant> {
        instant_from_offset(
            self.created_at,
            self.completed_after_ns.load(Ordering::Acquire),
        )
    }

    pub(super) fn snapshot(&self, id: u64) -> TaskMetadata {
        let worker_id = match self.worker_id.load(Ordering::Acquire) {
            NO_WORKER => None,
            worker_id => Some(worker_id),
        };

        TaskMetadata {
            id,
            created_at: self.created_at,
            started_at: instant_from_offset(
                self.created_at,
                self.started_after_ns.load(Ordering::Acquire),
            ),
            completed_at: self.completed_at(),
            worker_id,
        }
    }
}

impl TaskStateBlock {
    pub(super) fn new() -> Self {
        let slots = std::iter::repeat_with(|| None)
            .take(TASK_STATE_BLOCK_SIZE)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { slots }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }
}

#[inline]
pub(crate) fn elapsed_nanos_since(origin: Instant) -> u64 {
    let elapsed = origin.elapsed().as_nanos();
    elapsed.min(u128::from(TIMESTAMP_NOT_RECORDED - 1)) as u64
}

pub(crate) fn instant_from_offset(origin: Instant, offset_ns: u64) -> Option<Instant> {
    if offset_ns == TIMESTAMP_NOT_RECORDED {
        None
    } else {
        origin.checked_add(Duration::from_nanos(offset_ns))
    }
}

pub(crate) fn task_location(id: u64) -> (usize, usize) {
    let index = usize::try_from(id).expect("task ID must fit in usize");
    (index / TASK_STATE_BLOCK_SIZE, index % TASK_STATE_BLOCK_SIZE)
}
