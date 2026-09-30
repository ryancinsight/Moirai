#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
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

/// Which settled blocks a retirement sweep may release.
#[derive(Debug, Clone, Copy)]
pub(super) enum Retirement {
    /// Only blocks whose every task completed at or before the instant.
    CompletedBefore(Instant),
    /// Any block whose every task completed, whatever its age.
    Forced,
}

/// One fixed-size block of task-state slots.
///
/// A slot is written once, by the registration that owns its id, and then
/// published by a release store to its `published` flag; every observer reads
/// the flag with acquire semantics before touching the state, so a lookup racing
/// a registration sees either an absent slot or a complete state. A published
/// state never moves and is never replaced. The only way a state's storage is
/// released is retiring the whole block, which the registry does once every slot
/// has completed and released its lifecycle token. An owned token's block `Arc`
/// keeps the allocation alive; scheduler-bounded tokens require their registry
/// to outlive the job and make their final access to the state when they retire.
///
/// The flags live apart from the states: a flag beside its state would pad every
/// 72-byte state to 80 bytes, and the retirement scan reads flags without
/// touching state lines.
pub(super) struct TaskStateBlock {
    published: Box<[AtomicBool]>,
    states: Box<[UnsafeCell<MaybeUninit<TaskState>>]>,
}

// SAFETY: a state is written only before its `published` flag is set, by the one
// registration that owns the slot, and only read after an acquire load observes
// the flag. `TaskState` is `Send + Sync`, so sharing published states across
// threads is sound.
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

// A registry block holds 1,024 of these plus one flag byte each; the size is// pinned because retained memory per task is this figure.const _: () = assert!(size_of::<TaskState>() <= 72);
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

        // The guard is released before the waker runs. `if let Some(waker) =
        // self.waker.lock().unwrap().take()` would satisfy that only through
        // edition 2024's scrutinee rescoping (RFC 3606); binding the waker out
        // first keeps the rule visible at the site and independent of the
        // edition. Same discipline as `moirai-async`'s sync primitives.
        let waker = self.waker.lock().unwrap().take();
        if let Some(waker) = waker {
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
        let published = std::iter::repeat_with(|| AtomicBool::new(false))
            .take(TASK_STATE_BLOCK_SIZE)
            .collect();
        let states = std::iter::repeat_with(|| UnsafeCell::new(MaybeUninit::uninit()))
            .take(TASK_STATE_BLOCK_SIZE)
            .collect();

        Self { published, states }
    }

    /// Shared view of the state at `slot`, if the slot is registered and in range.
    pub(super) fn get(&self, slot: usize) -> Option<&TaskState> {
        if !self.published.get(slot)?.load(Ordering::Acquire) {
            return None;
        }
        // SAFETY: the acquire load observed the release store that follows the
        // slot's one write, so the state is initialized; nothing writes it again.
        Some(unsafe { (*self.states[slot].get()).assume_init_ref() })
    }

    /// Register a fresh state at `slot`, returning its stable address.
    ///
    /// # Safety
    ///
    /// No other call to `insert` has been or will be made for `slot`. The
    /// registry meets this by issuing every task id exactly once from its
    /// atomic counter and registering each id once.
    pub(super) unsafe fn insert(&self, slot: usize) -> NonNull<TaskState> {
        debug_assert!(
            !self.published[slot].load(Ordering::Relaxed),
            "a task id registers exactly once"
        );
        let cell = self.states[slot].get();
        // SAFETY: the caller is the slot's only registrant and the slot is not
        // yet published, so no reader touches the cell and this is the only
        // access to it. The address is stable: the boxed slice never moves or
        // shrinks and a published state is never replaced.
        unsafe { (*cell).write(TaskState::new()) };
        // SAFETY: `UnsafeCell::get` never returns null. The address derives
        // from the cell itself, as `get` does, and not from the `&mut` that
        // `write` returns: a pointer reborrowed from that transient reference
        // sits above it in the borrow stack, and the shared reads and atomic
        // stores that later readers make through the cell would invalidate it
        // before the lease uses it.
        let address = unsafe { NonNull::new_unchecked(cell.cast::<TaskState>()) };
        self.published[slot].store(true, Ordering::Release);
        address
    }

    /// Whether every task this block held has completed and released its
    /// lifecycle token, so the block can retire.
    ///
    /// Every slot must be registered: a vacant slot is an id that was issued
    /// but whose registration has not run yet, and retiring the block under it
    /// would lose that registration. `first_slot_unissued` exempts slot 0 of
    /// block 0, the one id the registry never issues. `retire` bounds the
    /// completion age; slots are examined newest first, so a block still inside
    /// its retention window is rejected after the first slot or two.
    pub(super) fn is_settled(&self, first_slot_unissued: bool, retire: Retirement) -> bool {
        (0..TASK_STATE_BLOCK_SIZE)
            .rev()
            .all(|slot| match self.get(slot) {
                None => first_slot_unissued && slot == 0,
                Some(state) => {
                    !state.token_active()
                        && state.completed_at().is_some_and(|completed| match retire {
                            Retirement::Forced => true,
                            Retirement::CompletedBefore(cutoff) => completed <= cutoff,
                        })
                }
            })
    }

    /// Iterate shared views of the registered states in this block.
    pub(super) fn states(&self) -> impl Iterator<Item = &TaskState> {
        (0..TASK_STATE_BLOCK_SIZE).filter_map(|slot| self.get(slot))
    }
}

impl Drop for TaskStateBlock {
    fn drop(&mut self) {
        for (published, state) in self.published.iter_mut().zip(self.states.iter_mut()) {
            if *published.get_mut() {
                // SAFETY: the flag is set only after the state is written, and
                // `&mut self` excludes every other access.
                unsafe { state.get_mut().assume_init_drop() };
            }
        }
    }
}

impl std::fmt::Debug for TaskStateBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskStateBlock")
            .field("slots", &self.published.len())
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
    use super::{TaskState, elapsed_nanos_since};
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
