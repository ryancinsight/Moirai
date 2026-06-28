use std::{
    cell::UnsafeCell,
    ptr::NonNull,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use super::super::task::TaskMetadata;

pub(crate) const NO_WORKER: usize = usize::MAX;
pub(crate) const TIMESTAMP_NOT_RECORDED: u64 = u64::MAX;
pub(crate) const TASK_STATE_BLOCK_SIZE: usize = 1024;

/// One fixed-size block of task-state slots.
///
/// Slots are `UnsafeCell` so the block can hand a stable `NonNull<TaskState>`
/// (via [`TaskStateBlock::insert`]) to a `TaskLifecycleToken` while the registry
/// keeps reading and structurally mutating other slots. All access goes through
/// the methods below, which touch a slot only through its own `UnsafeCell` —
/// never a `&mut`/`&` spanning the whole slice — so a live token's pointer into
/// one slot is never invalidated by activity on another.
///
/// # Safety contract (relied on by every slot accessor below)
/// 1. The owning [`super::TaskRegistry`] is shared only as
///    `Arc<Mutex<TaskRegistry>>`, so at most one registry method touches these
///    blocks at a time (no registry-vs-registry races).
/// 2. A slot's [`TaskState`] is interior-mutable (atomics + a `Mutex`). A
///    `TaskLifecycleToken` accesses only those fields through its `NonNull`,
///    without the registry mutex; that aliases the registry's shared reads of
///    the same fields soundly because all such access is atomic/locked.
/// 3. The registry writes a slot's `Option` only when no token aliases it:
///    `insert` targets a fresh unoccupied id; `clear` targets a completed slot
///    whose token has been consumed.
pub(super) struct TaskStateBlock {
    slots: Box<[UnsafeCell<Option<TaskState>>]>,
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
        // SAFETY: per the struct's safety contract, we form only a shared
        // `&TaskState` to interior-mutable state; the registry never writes this
        // slot's `Option` while a token to it is live, and concurrent token
        // access touches only the same state's atomics/mutex.
        self.slots
            .get(slot)
            .and_then(|cell| unsafe { (*cell.get()).as_ref() })
    }

    /// Insert a fresh state at `slot`, returning a stable pointer to it.
    pub(super) fn insert(&self, slot: usize) -> NonNull<TaskState> {
        let cell = self.slots[slot].get();
        // SAFETY: per the struct's safety contract, both the write and the
        // pointer derivation go through this slot's own `UnsafeCell` raw pointer,
        // never a `&mut`/`&` spanning the slice, so live tokens into sibling
        // slots stay valid. Assigning through the place drops any prior
        // (completed) state. The returned pointer is derived from a *shared*
        // view of the freshly written state: the token uses it only for the
        // state's interior-mutable (atomic/mutex) fields, so shared provenance
        // suffices and stays valid under both Stacked and Tree Borrows (a
        // `&mut`-derived pointer would be disabled by later shared reads).
        unsafe {
            *cell = Some(TaskState::new());
            NonNull::from((*cell).as_ref().unwrap_unchecked())
        }
    }

    /// Clear the state at `slot`, dropping it.
    pub(super) fn clear(&self, slot: usize) {
        // SAFETY: per the struct's safety contract, callers clear only completed
        // slots whose token has been consumed, so no live pointer aliases the
        // dropped state; the write is through this slot's own `UnsafeCell`.
        unsafe {
            *self.slots[slot].get() = None;
        }
    }

    /// Iterate shared views of the occupied states in this block.
    pub(super) fn states(&self) -> impl Iterator<Item = &TaskState> {
        // SAFETY: as in `get` — shared views of interior-mutable state.
        self.slots
            .iter()
            .filter_map(|cell| unsafe { (*cell.get()).as_ref() })
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
