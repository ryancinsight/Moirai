#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::{
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::Duration,
};

use super::super::task::TaskMetadata;
use super::state::{task_location, TaskState, TaskStateBlock};
use super::token::{SchedulerStateLease, TaskLifecycleToken};

/// Outcome of a cooperative cancel request against a registered task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancelOutcome {
    /// The cancel flag was set; the task body is skipped if it has not started.
    Requested,
    /// The task already completed; cancelling is a no-op.
    AlreadyCompleted,
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
#[derive(Debug)]
pub struct TaskRegistry {
    pub(super) blocks: RwLock<Vec<Arc<TaskStateBlock>>>,
    pub(super) next_id: AtomicU64,
}

impl TaskRegistry {
    /// Create a new task registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Register a new task and return its ID.
    pub fn register_task(&self) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.register_task_with_id(id);
        id
    }

    /// Register a new task and return its ID plus lifecycle mutation token.
    #[cfg(any(test, feature = "registry-diagnostics"))]
    pub(crate) fn register_next_task(&self) -> (u64, TaskLifecycleToken) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let lifecycle = self.register_task_with_id(id);
        (id, lifecycle)
    }

    /// Register a task with an externally allocated ID.
    pub(crate) fn register_task_with_id(&self, id: u64) -> TaskLifecycleToken {
        let (block, state) = self.initialize_task_with_id(id);
        TaskLifecycleToken::new_owned(block, state)
    }

    /// Register a task whose lifecycle cannot outlive this registry.
    ///
    /// # Safety
    ///
    /// The caller must keep this registry's blocks alive until the returned
    /// lifecycle token is consumed or dropped. Slot cleanup remains safe while
    /// the token is live because registration marks the slot active.
    pub(crate) unsafe fn register_next_scheduled_task(
        &self,
    ) -> (u64, TaskLifecycleToken<SchedulerStateLease>) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        // The scheduled token borrows the slot rather than owning the block, so
        // this path never needs the `Arc`; keeping the insert under the shared
        // guard avoids a refcount bump on every spawn.
        let state = self.insert_slot(id);
        (
            id,
            // SAFETY: forwarded from this method's caller contract.
            unsafe { TaskLifecycleToken::new_scheduled(state) },
        )
    }

    fn initialize_task_with_id(&self, id: u64) -> (Arc<TaskStateBlock>, NonNull<TaskState>) {
        let (block_index, slot_index) = task_location(id);
        let block = self.ensure_block(block_index);
        self.claim_slot(id, &block, slot_index);
        let state = block.insert(slot_index);
        (block, state)
    }

    /// Claim a slot and return only its state pointer.
    ///
    /// The owned-token path needs the block `Arc`; the scheduled path does not,
    /// and it is the one every spawn takes. Resolving the block under the
    /// shared guard and inserting there keeps that path free of a refcount
    /// bump. Falls back to the growing path when the block does not exist yet,
    /// which happens once per 1024 ids.
    fn insert_slot(&self, id: u64) -> NonNull<TaskState> {
        let (block_index, slot_index) = task_location(id);
        {
            let blocks = self
                .blocks
                .read()
                .expect("task registry block directory is never poisoned");
            if let Some(block) = blocks.get(block_index) {
                self.claim_slot(id, block, slot_index);
                return block.insert(slot_index);
            }
        }
        let block = self.ensure_block(block_index);
        self.claim_slot(id, &block, slot_index);
        block.insert(slot_index)
    }

    /// Advance the id watermark and reject re-registering a live slot.
    fn claim_slot(&self, id: u64, block: &TaskStateBlock, slot_index: usize) {
        self.next_id
            .fetch_max(id.saturating_add(1), Ordering::Relaxed);
        assert!(
            block
                .get(slot_index)
                .is_none_or(|state| state.is_completed() && !state.token_active()),
            "task ID must not be re-registered while active"
        );
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
    #[must_use]
    pub fn is_completed(&self, task_id: u64) -> bool {
        self.with_state(task_id, TaskState::is_completed)
            .unwrap_or(false)
    }

    /// Get task metadata.
    #[must_use]
    pub fn get_metadata(&self, task_id: u64) -> Option<TaskMetadata> {
        self.with_state(task_id, |state| state.snapshot(task_id))
    }

    /// Remove old completed tasks to prevent retained task metadata growth.
    pub fn cleanup_completed(&self, older_than: Duration) {
        // `Instant - Duration` panics when the result predates the platform's
        // clock origin, which a caller-supplied retention window longer than the
        // process uptime reaches. No recorded completion can be older than a
        // cutoff before the clock started, so that case is an empty sweep.
        let Some(cutoff) = std::time::Instant::now().checked_sub(older_than) else {
            return;
        };
        let mut blocks = self
            .blocks
            .write()
            .expect("task registry block directory is never poisoned");
        for block in blocks.iter() {
            for slot_index in 0..block.len() {
                let removable = block.get(slot_index).is_some_and(|state| {
                    !state.token_active()
                        && state
                            .completed_at()
                            .is_some_and(|completed| completed <= cutoff)
                });
                if removable {
                    block.clear(slot_index);
                }
            }
        }

        while blocks.last().is_some_and(|block| block.is_empty()) {
            blocks.pop();
        }
    }

    /// Get count of active tasks.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.blocks
            .read()
            .expect("task registry block directory is never poisoned")
            .iter()
            .flat_map(|block| block.states())
            .filter(|state| !state.is_completed())
            .count()
    }

    /// Get count of completed tasks.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.blocks
            .read()
            .expect("task registry block directory is never poisoned")
            .iter()
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
    /// two acquisitions.
    pub(super) fn ensure_block(&self, block_index: usize) -> Arc<TaskStateBlock> {
        if let Some(block) = self
            .blocks
            .read()
            .expect("task registry block directory is never poisoned")
            .get(block_index)
        {
            return Arc::clone(block);
        }
        let mut blocks = self
            .blocks
            .write()
            .expect("task registry block directory is never poisoned");
        while blocks.len() <= block_index {
            blocks.push(Arc::new(TaskStateBlock::new()));
        }
        Arc::clone(&blocks[block_index])
    }

    /// Run `f` against the state slot for `task_id`, if it is registered.
    ///
    /// Callers take the block by `Arc` rather than borrowing through the
    /// directory guard, so the shared lock is released before `f` runs.
    pub(super) fn with_state<R>(&self, task_id: u64, f: impl FnOnce(&TaskState) -> R) -> Option<R> {
        let (block_index, slot_index) = task_location(task_id);
        let block = {
            let blocks = self
                .blocks
                .read()
                .expect("task registry block directory is never poisoned");
            Arc::clone(blocks.get(block_index)?)
        };
        let state = block.get(slot_index)?;
        Some(f(state))
    }

    /// Request cooperative cancellation of a task.
    ///
    /// Returns `None` when the task is unknown. Running tasks are not
    /// preempted: a task that already started keeps running to completion and
    /// reports `Requested` here without effect.
    pub(crate) fn request_cancel(&self, task_id: u64) -> Option<CancelOutcome> {
        self.with_state(task_id, |state| {
            if state.is_completed() {
                CancelOutcome::AlreadyCompleted
            } else {
                state.request_cancel();
                CancelOutcome::Requested
            }
        })
    }

    /// Register a waker to be notified when the task completes.
    pub fn register_waker(&self, task_id: u64, waker: &std::task::Waker) -> bool {
        self.with_state(task_id, |state| {
            {
                let mut guard = state.waker.lock().unwrap();
                *guard = Some(waker.clone());
            }
            // Store first, then re-check completion, mirroring the ordering
            // `mark_completed_since` publishes: it stores the completion offset
            // before taking the waker. A task that completed before this store
            // has already taken the absent waker and will never take again, so
            // the one just stored would be held for the life of the slot —
            // along with whatever it owns, typically an `Arc` to async task
            // state. Reclaiming it here is race-free in both directions: if
            // completion lands after the store, it takes and wakes; if it
            // landed before, this take wins and wakes instead. Only one take
            // can succeed, and a spurious wake is always permitted.
            if state.is_completed() {
                let stranded = state.waker.lock().unwrap().take();
                if let Some(stranded) = stranded {
                    stranded.wake();
                }
            }
        })
        .is_some()
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}
