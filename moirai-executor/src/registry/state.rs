use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use moirai_core::Priority;

use super::super::task::TaskMetadata;
use super::{TaskCompletion, TaskWaiter, waiter::WaiterSetSlot};

/// Inverse of [`Priority::index`]: `PRIORITY_FROM_INDEX[p.index()] == p` for
/// every variant (asserted by `priority_index_round_trips` in the registry tests).
pub(crate) const PRIORITY_FROM_INDEX: [Priority; Priority::Critical.index() + 1] = [
    Priority::Low,
    Priority::Normal,
    Priority::High,
    Priority::Critical,
];

pub(crate) const NO_WORKER: u32 = u32::MAX;
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
/// 64-byte state to 72 bytes, and the retirement scan reads flags without
/// touching state lines.
pub(super) struct TaskStateBlock {
    published: Box<[AtomicBool]>,
    states: Box<[UnsafeCell<MaybeUninit<TaskState>>]>,
}

// SAFETY: a state is written only before its `published` flag is set, by the one
// registration that owns its slot, and only read after an acquire load observes
// the flag. `TaskState` is `Send + Sync`, so sharing published states across
// threads is sound.
unsafe impl Sync for TaskStateBlock {}

/// Shared lifecycle state for one task.
pub(crate) struct TaskState {
    pub(crate) created_at: Instant,
    pub(super) started_after_ns: AtomicU64,
    pub(super) completed_after_ns: AtomicU64,
    /// Worker index at start, or [`NO_WORKER`]. `u32` keeps the struct within
    /// its pinned budget on every supported pointer width; no scheduler this
    /// executor builds approaches 2^32 workers.
    pub(super) worker_id: AtomicU32,
    /// Lazily allocated owned completion registrations.
    pub(super) waiters: WaiterSetSlot,
    /// True while a lifecycle token can still access this slot; clearing it is
    /// the token's last access ([`TaskState::retire_token`]).
    token_active: AtomicBool,
    /// Spawn priority stored as its [`Priority::index`] discriminant.
    pub(super) priority: AtomicU8,
    /// Set by `cancel_task`; observed cooperatively at job start.
    pub(super) cancel_requested: AtomicBool,
    /// Set when the task ended without a result: a cancel request was honored,
    /// or its lifecycle token was dropped before completing. The task's handle
    /// normally reports `TaskError::Cancelled`; destruction failures are
    /// reported separately by the result handle.
    pub(super) cancelled: AtomicBool,
}

// A registry block holds 1,024 of these plus one flag byte each; the size is
// pinned because retained memory per task is this figure: 64 bytes on every
// 64-bit target (`Instant` is 16 bytes, 8-aligned, on each), 56 on wasm32.
// `worker_id` is `u32` and the waiter-set pointer carries its own compact spin
// lock to stay within it on macOS, Linux, and Windows.
const _: () = assert!(size_of::<TaskState>() <= 64);

impl std::fmt::Debug for TaskState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskState")
            .field("created_at", &self.created_at)
            .field("started_after_ns", &self.started_after_ns)
            .field("completed_after_ns", &self.completed_after_ns)
            .field("worker_id", &self.worker_id)
            .field("waiter_set_allocated", &self.waiters.is_some())
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
            worker_id: AtomicU32::new(NO_WORKER),
            waiters: WaiterSetSlot::new(),
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

    /// Clear the token-active flag, releasing a lease's claim on the state.
    ///
    /// This is a lease's last access. Once the flag is clear a retention sweep
    /// may free the whole block, and it does not wait for the store to return.
    /// A `&TaskState` passed to a method stays valid until that method returns,
    /// so retiring through one lets the block be freed under a live reference;
    /// this takes the state by pointer, projects the flag by address, and
    /// borrows only the one-byte atomic.
    ///
    /// # Safety
    ///
    /// `state` points to a live `TaskState` whose flag the calling lease has
    /// not yet cleared, and the caller makes no further access to it.
    #[inline]
    pub(super) unsafe fn retire_token(state: NonNull<Self>) {
        // SAFETY: the caller guarantees `state` is live; `&raw const` projects
        // the field address without creating a reference to the `TaskState`.
        let flag = unsafe { &raw const (*state.as_ptr()).token_active };
        // SAFETY: the flag is a live `AtomicBool` for the same reason. The
        // borrow covers this one byte and ends with the store.
        unsafe { (*flag).store(false, Ordering::Release) };
    }

    /// Publish that a cancel request was honored: the task completes without
    /// its body having run, and any registered waiter is woken.
    pub(super) fn mark_cancelled(&self) -> TaskCompletion {
        self.cancelled.store(true, Ordering::Release);
        self.mark_completed()
    }

    #[inline]
    pub(super) fn mark_started(&self, worker_id: usize) -> u64 {
        let started_after_ns = elapsed_nanos_since(self.created_at);
        self.started_after_ns
            .store(started_after_ns, Ordering::Release);
        // A real scheduler never reaches `u32::MAX` workers; on a hypothetical
        // wider pool, report "no worker" rather than mis-attribute one.
        self.worker_id.store(
            u32::try_from(worker_id).unwrap_or(NO_WORKER),
            Ordering::Release,
        );
        started_after_ns
    }

    #[inline]
    pub(super) fn mark_completed_since(&self, started_after_ns: u64) -> TaskCompletion {
        self.mark_completed_at(started_after_ns, elapsed_nanos_since(self.created_at))
    }

    #[inline]
    fn mark_completed_at(&self, started_after_ns: u64, measured_after_ns: u64) -> TaskCompletion {
        // `Instant` documents saturation for rare platform monotonicity
        // violations. Preserve that contract across thread/core migration by
        // clamping the published completion offset to the recorded start.
        let completed_after_ns = measured_after_ns.max(started_after_ns);
        self.completed_after_ns
            .store(completed_after_ns, Ordering::Release);

        // Detach the set before closing it. A racing registration either lands
        // before `close` and is included, or observes the closed set and keeps
        // no callback. Notification runs later through `TaskCompletion`.
        let notifications = self
            .waiters
            .take()
            .map_or_else(super::waiter::WaiterNotifications::empty, |set| set.close());
        TaskCompletion::new(
            Duration::from_nanos(completed_after_ns - started_after_ns),
            notifications,
        )
    }

    pub(super) fn mark_completed(&self) -> TaskCompletion {
        let started_after_ns = self.started_after_ns.load(Ordering::Acquire);
        let started_after_ns = if started_after_ns == TIMESTAMP_NOT_RECORDED {
            elapsed_nanos_since(self.created_at)
        } else {
            started_after_ns
        };
        self.mark_completed_since(started_after_ns)
    }

    /// Register an owned waiter while this task is pending.
    pub(super) fn register_waker(&self, waker: &std::task::Waker) -> Option<TaskWaiter> {
        if self.is_completed() {
            return None;
        }
        // A RawWaker clone callback is arbitrary user code. Clone before either
        // the task's compact slot lock or the set's bookkeeping mutex.
        let incoming = waker.clone();
        let Some(set) = self.waiters.get_or_create(|| self.is_completed()) else {
            super::waiter::dispose_unregistered(incoming);
            return None;
        };
        match set.register(incoming) {
            Ok(waiter) => Some(waiter),
            Err(unused) => {
                super::waiter::dispose_unregistered(unused);
                None
            }
        }
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
            // Lossless widening: `usize` is at least 32 bits on every
            // supported target.
            worker_id => Some(worker_id as usize),
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
    use super::TaskState;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    #[test]
    fn completion_clamps_to_recorded_start_offset() {
        for (started, measured, completed, duration) in [(7, 3, 7, 0), (7, 7, 7, 0), (7, 13, 13, 6)]
        {
            let state = TaskState::new();
            state.started_after_ns.store(started, Ordering::Release);

            let elapsed = state.mark_completed_at(started, measured).notify();
            let snapshot = state.snapshot(7);

            assert_eq!(elapsed, Duration::from_nanos(duration));
            assert_eq!(state.completed_after_ns.load(Ordering::Acquire), completed);
            assert_eq!(
                snapshot.completed_at,
                state
                    .created_at
                    .checked_add(Duration::from_nanos(completed))
            );
            assert_eq!(snapshot.execution_duration(), Some(elapsed));
        }
    }
}
