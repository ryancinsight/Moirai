use std::{
    ptr::NonNull,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};

use super::super::task::TaskMetadata;
use super::directory::{BlockDirectory, BlockLookup};
use super::retention::RetentionPolicy;
use super::state::{TaskState, TaskStateBlock, task_location};
use super::token::{SchedulerStateLease, TaskLifecycleToken};

/// Outcome of a cooperative cancel request against a registered task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancelOutcome {
    /// The cancel flag was set; the task body is skipped if it has not started.
    Requested,
    /// The task already completed; cancelling is a no-op.
    AlreadyCompleted,
}

/// What the registry knows about a task id.
enum Observation<R> {
    /// The id's block was retired: the task completed and its state was released.
    Retired,
    /// No task is registered under the id.
    Unregistered,
    /// The task's state, as seen by the observer.
    Registered(R),
}

/// A task id issued by the registry and not yet registered.
///
/// Registration consumes it and nothing else constructs one, so a slot has at
/// most one registrant and the block can write it without a claim.
#[derive(Debug)]
pub(super) struct IssuedId(u64);

impl IssuedId {
    pub(super) const fn get(&self) -> u64 {
        self.0
    }

    /// Block index and slot index of the id.
    pub(super) fn location(&self) -> (usize, usize) {
        task_location(self.0)
    }
}

/// Public task registry facade used by executor lifecycle tracking and tests.
///
/// Registration and lookup take `&self` so the executor can share one registry
/// without an outer mutex. Every spawn used to serialize on that mutex ahead of
/// the lock-free scheduler: measured on an 8-core pin, executor spawn ran
/// 3.18 M/s with one producer and *fell* to 2.97 M/s with eight, while the same
/// scheduler reached without the registry rose from 6.18 M/s to 8.85 M/s.
///
/// The id counter is atomic, and the block directory takes its lock in read
/// mode for the common path — a block is created once per 1024 ids, and slot
/// insertion itself only needs `&TaskStateBlock`.
///
/// Storage is bounded by a [`RetentionPolicy`] when one is set: settled blocks
/// are released, and [`TaskRegistry::is_completed`] keeps answering `true` for
/// their tasks. Without a policy every block stays resident until
/// [`TaskRegistry::cleanup_completed`] releases it.
#[derive(Debug)]
pub struct TaskRegistry {
    pub(super) blocks: RwLock<BlockDirectory>,
    pub(super) next_id: AtomicU64,
    pub(super) retention: Option<RetentionPolicy>,
}

impl TaskRegistry {
    /// Create a registry that retains every task until it is cleaned up
    /// explicitly.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            blocks: RwLock::new(BlockDirectory::new()),
            next_id: AtomicU64::new(1),
            retention: None,
        }
    }

    /// Create a registry whose completed tasks are released under `policy`.
    #[must_use]
    pub fn with_retention(policy: RetentionPolicy) -> Self {
        Self {
            retention: Some(policy),
            ..Self::new()
        }
    }

    /// Issue the next task id.
    ///
    /// The counter is the only source of ids, so every id is issued once; the
    /// returned [`IssuedId`] is consumed by registration, which is what lets a
    /// slot be written without a claim.
    pub(super) fn issue_id(&self) -> IssuedId {
        IssuedId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Register a new task and return its ID.
    ///
    /// The task stays queued until the caller drives it by id:
    /// [`TaskRegistry::mark_started`], then [`TaskRegistry::mark_completed`].
    /// Until it completes, its block is never retired, so both calls reach the
    /// task. Executor code that owns the task's lifecycle uses a token instead.
    pub fn register_task(&self) -> u64 {
        let id = self.issue_id();
        let task_id = id.get();
        self.register_owned(id).release();
        task_id
    }

    /// Register a new task and return its ID plus lifecycle mutation token.
    #[cfg(any(test, feature = "registry-diagnostics"))]
    pub(crate) fn register_next_task(&self) -> (u64, TaskLifecycleToken) {
        let id = self.issue_id();
        let task_id = id.get();
        (task_id, self.register_owned(id))
    }

    /// Register a task whose lifecycle cannot outlive this registry.
    ///
    /// # Safety
    ///
    /// The caller must keep this registry's blocks alive until the returned
    /// lifecycle token is consumed or dropped. Block retirement remains safe
    /// while the token is live because registration marks the slot active and
    /// only a block with no active slot retires.
    pub(crate) unsafe fn register_next_scheduled_task(
        &self,
    ) -> (u64, TaskLifecycleToken<SchedulerStateLease>) {
        let id = self.issue_id();
        let task_id = id.get();
        // The scheduled token borrows the slot rather than owning the block, so
        // this path never needs the `Arc`; keeping the insert under the shared
        // guard avoids a refcount bump on every spawn.
        let state = self.insert_slot(id);
        (
            task_id,
            // SAFETY: forwarded from this method's caller contract.
            unsafe { TaskLifecycleToken::new_scheduled(state) },
        )
    }

    pub(super) fn register_owned(&self, id: IssuedId) -> TaskLifecycleToken {
        let (block_index, slot_index) = id.location();
        let block = self.ensure_block(block_index);
        // SAFETY: `id` was issued once by the counter and registration consumes it.
        let state = unsafe { block.insert(slot_index) };
        TaskLifecycleToken::new_owned(block, state)
    }

    /// Register a slot and return only its state pointer.
    ///
    /// The owned-token path needs the block `Arc`; the scheduled path does not,
    /// and it is the one every spawn takes. Resolving the block under the
    /// shared guard and inserting there keeps that path free of a refcount
    /// bump. Falls back to the growing path when the block does not exist yet,
    /// which happens once per 1024 ids.
    fn insert_slot(&self, id: IssuedId) -> NonNull<TaskState> {
        let (block_index, slot_index) = id.location();
        {
            let blocks = self
                .blocks
                .read()
                .expect("task registry block directory is never poisoned");
            if let BlockLookup::Live(block) = blocks.lookup(block_index) {
                // SAFETY: `id` was issued once by the counter and registration
                // consumes it.
                return unsafe { block.insert(slot_index) };
            }
        }
        let block = self.ensure_block(block_index);
        // SAFETY: as above.
        unsafe { block.insert(slot_index) }
    }

    /// Mark a task as started.
    pub fn mark_started(&self, task_id: u64, worker_id: usize) {
        self.with_state(task_id, |state| {
            state.mark_started(worker_id);
        });
    }

    /// Mark a task as completed.
    pub fn mark_completed(&self, task_id: u64) {
        self.with_state(task_id, TaskState::mark_completed);
    }

    /// Check if a task is completed.
    ///
    /// A task whose block was released under the retention policy is completed.
    #[must_use]
    pub fn is_completed(&self, task_id: u64) -> bool {
        self.completion(task_id) == Some(true)
    }

    /// Report whether a task completed, from one observation of the registry.
    ///
    /// `None` is an id that was never registered. A task whose block was
    /// released under the retention policy is `Some(true)`. A caller that must
    /// tell an unknown id from a finished one reads this once: two separate
    /// lookups can straddle the release of the task's block.
    #[must_use]
    pub fn completion(&self, task_id: u64) -> Option<bool> {
        match self.observe(task_id, TaskState::is_completed) {
            Observation::Retired => Some(true),
            Observation::Unregistered => None,
            Observation::Registered(completed) => Some(completed),
        }
    }

    /// Get task metadata, or `None` for an unregistered task and for one whose
    /// metadata the retention policy already released.
    #[must_use]
    pub fn get_metadata(&self, task_id: u64) -> Option<TaskMetadata> {
        self.with_state(task_id, |state| state.snapshot(task_id))
    }

    /// Get count of active tasks.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.blocks
            .read()
            .expect("task registry block directory is never poisoned")
            .resident_blocks()
            .flat_map(|block| block.states())
            .filter(|state| !state.is_completed())
            .count()
    }

    /// Get count of completed tasks whose state is still resident.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.blocks
            .read()
            .expect("task registry block directory is never poisoned")
            .resident_blocks()
            .flat_map(|block| block.states())
            .filter(|state| state.is_completed())
            .count()
    }

    /// Resolve the block for `block_index`, creating it if absent.
    ///
    /// The read path is the common one: a block is created once per 1024 ids,
    /// so all but that registration take the lock in shared mode and never
    /// exclude a concurrent spawn. The length is re-checked under the write
    /// lock because another producer may have grown the directory between the
    /// two acquisitions. Creating a block advances the retention sweep by one
    /// window, after the directory lock is released.
    ///
    /// Every caller holds an issued id whose slot is not yet registered, so the
    /// block cannot have retired.
    pub(super) fn ensure_block(&self, block_index: usize) -> Arc<TaskStateBlock> {
        if let BlockLookup::Live(block) = self
            .blocks
            .read()
            .expect("task registry block directory is never poisoned")
            .lookup(block_index)
        {
            return Arc::clone(block);
        }
        let ensured = self
            .blocks
            .write()
            .expect("task registry block directory is never poisoned")
            .ensure(block_index);
        let (block, created) =
            ensured.expect("invariant: a block holding an issued, unregistered id never retires");
        if created {
            self.sweep_step();
        }
        block
    }

    /// Run `f` against the state of `task_id`, if it is resident.
    pub(super) fn with_state<R>(&self, task_id: u64, f: impl FnOnce(&TaskState) -> R) -> Option<R> {
        match self.observe(task_id, f) {
            Observation::Registered(value) => Some(value),
            Observation::Retired | Observation::Unregistered => None,
        }
    }

    /// Run `f` against the state slot for `task_id` and report how the id stands.
    ///
    /// Callers take the block by `Arc` rather than borrowing through the
    /// directory guard, so the shared lock is released before `f` runs.
    fn observe<R>(&self, task_id: u64, f: impl FnOnce(&TaskState) -> R) -> Observation<R> {
        let (block_index, slot_index) = task_location(task_id);
        let block = {
            let blocks = self
                .blocks
                .read()
                .expect("task registry block directory is never poisoned");
            match blocks.lookup(block_index) {
                BlockLookup::Live(block) => Arc::clone(block),
                BlockLookup::Retired => return Observation::Retired,
                BlockLookup::Absent => return Observation::Unregistered,
            }
        };
        block
            .get(slot_index)
            .map_or(Observation::Unregistered, |state| {
                Observation::Registered(f(state))
            })
    }

    /// Request cooperative cancellation of a task.
    ///
    /// Returns `None` when the task is unknown. Running tasks are not
    /// preempted: a task that already started keeps running to completion and
    /// reports `Requested` here without effect. A task whose state the
    /// retention policy released completed earlier and reports
    /// `AlreadyCompleted`.
    pub(crate) fn request_cancel(&self, task_id: u64) -> Option<CancelOutcome> {
        match self.observe(task_id, |state| {
            if state.is_completed() {
                CancelOutcome::AlreadyCompleted
            } else {
                state.request_cancel();
                CancelOutcome::Requested
            }
        }) {
            Observation::Retired => Some(CancelOutcome::AlreadyCompleted),
            Observation::Unregistered => None,
            Observation::Registered(outcome) => Some(outcome),
        }
    }

    /// Register a waker to be notified when the task completes.
    ///
    /// Every distinct waker registered before completion is woken by it; a
    /// waker that `will_wake` one already registered is normally not added
    /// again (`Waker::will_wake` is best-effort). A task that already completed
    /// wakes the waker at once and keeps nothing. Returns `false` for an id the registry does not hold.
    pub fn register_waker(&self, task_id: u64, waker: &std::task::Waker) -> bool {
        self.with_state(task_id, |state| state.register_waker(waker))
            .is_some()
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}
