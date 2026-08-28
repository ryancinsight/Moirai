#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::{
    cell::UnsafeCell,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use moirai_core::Priority;

use super::super::task::TaskMetadata;

/// Inverse of [`Priority::index`]: `PRIORITY_FROM_INDEX[p.index()] == p` for
/// every variant (asserted by `priority_index_round_trips` in the registry tests).
pub(crate) const PRIORITY_FROM_INDEX: [Priority; Priority::Critical.index() + 1] = [
    Priority::Low,
    Priority::Normal,
    Priority::High,
    Priority::Critical,
];

pub(crate) const NO_WORKER: usize = usize::MAX;
pub(crate) const TIMESTAMP_NOT_RECORDED: u64 = u64::MAX;
pub(crate) const TASK_STATE_BLOCK_SIZE: usize = 1024;

/// One fixed-size block of task-state slots.
///
/// Slots are `UnsafeCell` so the registry can initialize and retire individual
/// states while lifecycle tokens retain shared ownership of the block. All
/// access goes through the methods below, which touch a slot only through its
/// own `UnsafeCell` — never a `&mut`/`&` spanning the whole slice.
///
/// # Safety contract (relied on by every slot accessor below)
/// 1. Structural slot mutation requires exclusive [`super::TaskRegistry`]
///    access. The executor shares that registry only through `Arc<Mutex<_>>`,
///    so no two registry operations mutate a block concurrently.
/// 2. A slot's [`TaskState`] is interior-mutable (atomics + a `Mutex`). A
///    lifecycle token accesses only those fields through a shared block view or
///    a stable pointer, so concurrent token and registry reads are atomic/locked.
/// 3. The registry writes a slot's `Option` only when `token_active == false`:
///    `insert` targets a fresh or retired id; `clear` targets a completed,
///    retired slot. An owned token's block `Arc` keeps the allocation alive;
///    scheduler-bounded tokens require their registry to outlive the job.
pub(super) struct TaskStateBlock {
    slots: Box<[UnsafeCell<Option<TaskState>>]>,
}

// SAFETY: registry mutation is serialized and writes only one slot's
// `UnsafeCell` after that slot's token retires. Lifecycle tokens keep the block
// alive and access only their slot's atomic/mutex fields. Sibling-slot writes
// are disjoint, so sharing the block across token and registry threads is safe.
unsafe impl Sync for TaskStateBlock {}

/// Shared lifecycle state for one task.
pub(crate) struct TaskState {
    pub(crate) created_at: Instant,
    pub(super) started_after_ns: AtomicU64,
    pub(super) completed_after_ns: AtomicU64,
    pub(super) worker_id: AtomicUsize,
    pub(super) waker: std::sync::Mutex<Option<std::task::Waker>>,
    /// True while a lifecycle token can still access this slot.
    token_active: AtomicBool,
    /// Spawn priority stored as its [`Priority::index`] discriminant.
    pub(super) priority: AtomicU8,
    /// Set by `cancel_task`; observed cooperatively at job start.
    pub(super) cancel_requested: AtomicBool,
    /// Set when a cancel request was honored (the job body never ran).
    pub(super) cancelled: AtomicBool,
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
            token_active: AtomicBool::new(true),
            // Lossless enum-to-int cast: Priority discriminants are 0..=3.
            priority: AtomicU8::new(Priority::Normal as u8),
            cancel_requested: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        }
    }

    #[inline]
    pub(super) fn set_priority(&self, priority: Priority) {
        // Lossless enum-to-int cast: Priority discriminants are 0..=3.
        self.priority.store(priority as u8, Ordering::Relaxed);
    }

    #[inline]
    pub(super) fn priority(&self) -> Priority {
        // Invariant: the slot only ever stores `priority as u8` (0..=3), so the
        // lookup cannot go out of bounds.
        PRIORITY_FROM_INDEX[usize::from(self.priority.load(Ordering::Relaxed))]
    }

    /// Flag the task for cooperative cancellation.
    #[inline]
    pub(super) fn request_cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
    }

    #[inline]
    pub(super) fn cancel_requested(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }

    #[inline]
    pub(super) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    #[inline]
    pub(super) fn token_active(&self) -> bool {
        self.token_active.load(Ordering::Acquire)
    }

    #[inline]
    pub(super) fn retire_token(&self) {
        self.token_active.store(false, Ordering::Release);
    }

    /// Publish that a cancel request was honored: the task completes without
    /// its body having run, and any registered waiter is woken.
    pub(super) fn mark_cancelled(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.mark_completed();
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
        // `Instant` documents saturation for rare platform monotonicity
        // violations. Preserve that contract across thread/core migration by
        // clamping the published completion offset to the recorded start.
        let completed_after_ns = elapsed_nanos_since(self.created_at).max(started_after_ns);
        self.completed_after_ns
            .store(completed_after_ns, Ordering::Release);

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
            priority: self.priority(),
            cancelled: self.is_cancelled(),
        }
    }
}

impl TaskStateBlock {
    pub(super) fn new() -> Self {
        let slots = std::iter::repeat_with(|| UnsafeCell::new(None))
            .take(TASK_STATE_BLOCK_SIZE)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { slots }
    }

    /// Number of slots in the block.
    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Shared view of the state at `slot`, if the slot is occupied and in range.
    pub(super) fn get(&self, slot: usize) -> Option<&TaskState> {
        let cell = self.slots.get(slot);
        // SAFETY: per the struct's safety contract, we form only a shared
        // `&TaskState` to interior-mutable state; the registry never writes
        // this slot's `Option` while a token to it is live, and concurrent
        // token access touches only the same state's atomics/mutex.
        cell.and_then(|cell| unsafe { (*cell.get()).as_ref() })
    }

    /// Insert a fresh state at `slot`, returning its stable address.
    pub(super) fn insert(&self, slot: usize) -> NonNull<TaskState> {
        let cell = self.slots[slot].get();
        // SAFETY: per the struct's safety contract, the write goes through this
        // slot's own `UnsafeCell`; no active token aliases the replaced state,
        // and live tokens into sibling slots touch disjoint cells. The pointer
        // derives from a shared view because tokens use only interior-mutability
        // operations; registry code never moves an initialized live slot.
        unsafe {
            *cell = Some(TaskState::new());
            NonNull::from((*cell).as_ref().unwrap_unchecked())
        }
    }

    /// Clear the state at `slot`, dropping it.
    pub(super) fn clear(&self, slot: usize) {
        debug_assert!(
            self.get(slot).is_none_or(|state| !state.token_active()),
            "retiring a registry slot requires its lifecycle token to be gone"
        );
        // SAFETY: per the struct's safety contract, callers clear only completed
        // slots whose token has retired, so no lifecycle access aliases the
        // dropped state; the write is through this slot's own `UnsafeCell`.
        unsafe {
            *self.slots[slot].get() = None;
        }
    }

    /// Iterate shared views of the occupied states in this block.
    pub(super) fn states(&self) -> impl Iterator<Item = &TaskState> {
        let cells = self.slots.iter();
        // SAFETY: as in `get` — shared views of interior-mutable state.
        cells.filter_map(|cell| unsafe { (*cell.get()).as_ref() })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.states().next().is_none()
    }
}

impl std::fmt::Debug for TaskStateBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskStateBlock")
            .field("slots", &self.slots.len())
            .field("occupied", &self.states().count())
            .finish()
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

#[cfg(test)]
mod tests {
    use super::{elapsed_nanos_since, TaskState};
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    #[test]
    fn completion_clamps_to_recorded_start_offset() {
        let state = TaskState::new();
        let future_start = elapsed_nanos_since(state.created_at).saturating_add(1_000_000);
        state
            .started_after_ns
            .store(future_start, Ordering::Release);

        let elapsed = state.mark_completed_since(future_start);
        let snapshot = state.snapshot(7);

        assert_eq!(elapsed, Duration::ZERO);
        assert_eq!(snapshot.started_at, snapshot.completed_at);
        assert_eq!(snapshot.execution_duration(), Some(Duration::ZERO));
    }
}
