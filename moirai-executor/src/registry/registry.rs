#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::{ptr::NonNull, sync::Arc, time::Duration};

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
#[derive(Debug)]
pub struct TaskRegistry {
    pub(super) blocks: Vec<Arc<TaskStateBlock>>,
    pub(super) next_id: u64,
}

impl TaskRegistry {
    /// Create a new task registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new task and return its ID.
    pub fn register_task(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.register_task_with_id(id);
        id
    }

    /// Register a new task and return its ID plus lifecycle mutation token.
    #[cfg(any(test, feature = "registry-diagnostics"))]
    pub(crate) fn register_next_task(&mut self) -> (u64, TaskLifecycleToken) {
        let id = self.next_id;
        let lifecycle = self.register_task_with_id(id);
        (id, lifecycle)
    }

    /// Register a task with an externally allocated ID.
    pub(crate) fn register_task_with_id(&mut self, id: u64) -> TaskLifecycleToken {
        let (block_index, state) = self.initialize_task_with_id(id);
        TaskLifecycleToken::new_owned(Arc::clone(&self.blocks[block_index]), state)
    }

    /// Register a task whose lifecycle cannot outlive this registry.
    ///
    /// # Safety
    ///
    /// The caller must keep this registry's blocks alive until the returned
    /// lifecycle token is consumed or dropped. Slot cleanup remains safe while
    /// the token is live because registration marks the slot active.
    pub(crate) unsafe fn register_next_scheduled_task(
        &mut self,
    ) -> (u64, TaskLifecycleToken<SchedulerStateLease>) {
        let id = self.next_id;
        let (_block_index, state) = self.initialize_task_with_id(id);
        (
            id,
            // SAFETY: forwarded from this method's caller contract.
            unsafe { TaskLifecycleToken::new_scheduled(state) },
        )
    }

    fn initialize_task_with_id(&mut self, id: u64) -> (usize, NonNull<TaskState>) {
        self.next_id = self.next_id.max(id.saturating_add(1));
        let (block_index, slot_index) = task_location(id);
        self.ensure_block(block_index);

        let block = &self.blocks[block_index];
        assert!(
            block
                .get(slot_index)
                .is_none_or(|state| state.is_completed() && !state.token_active()),
            "task ID must not be re-registered while active"
        );

        let state = block.insert(slot_index);
        (block_index, state)
    }

    /// Mark a task as started.
    pub fn mark_started(&self, task_id: u64, worker_id: usize) {
        if let Some(state) = self.state(task_id) {
            state.mark_started(worker_id);
        }
    }

    /// Mark a task as completed.
    pub fn mark_completed(&self, task_id: u64) {
        if let Some(state) = self.state(task_id) {
            state.mark_completed();
        }
    }

    /// Check if a task is completed.
    #[must_use]
    pub fn is_completed(&self, task_id: u64) -> bool {
        self.state(task_id).is_some_and(TaskState::is_completed)
    }

    /// Get task metadata.
    #[must_use]
    pub fn get_metadata(&self, task_id: u64) -> Option<TaskMetadata> {
        self.state(task_id).map(|state| state.snapshot(task_id))
    }

    /// Remove old completed tasks to prevent retained task metadata growth.
    pub fn cleanup_completed(&mut self, older_than: Duration) {
        // `Instant - Duration` panics when the result predates the platform's
        // clock origin, which a caller-supplied retention window longer than the
        // process uptime reaches. No recorded completion can be older than a
        // cutoff before the clock started, so that case is an empty sweep.
        let Some(cutoff) = std::time::Instant::now().checked_sub(older_than) else {
            return;
        };
        for block in &self.blocks {
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

        while self.blocks.last().is_some_and(|block| block.is_empty()) {
            self.blocks.pop();
        }
    }

    /// Get count of active tasks.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|block| block.states())
            .filter(|state| !state.is_completed())
            .count()
    }

    /// Get count of completed tasks.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|block| block.states())
            .filter(|state| state.is_completed())
            .count()
    }

    pub(super) fn ensure_block(&mut self, block_index: usize) {
        while self.blocks.len() <= block_index {
            self.blocks.push(Arc::new(TaskStateBlock::new()));
        }
    }

    pub(super) fn state(&self, task_id: u64) -> Option<&TaskState> {
        let (block_index, slot_index) = task_location(task_id);
        self.blocks.get(block_index)?.get(slot_index)
    }

    /// Request cooperative cancellation of a task.
    ///
    /// Returns `None` when the task is unknown. Running tasks are not
    /// preempted: a task that already started keeps running to completion and
    /// reports `Requested` here without effect.
    pub(crate) fn request_cancel(&self, task_id: u64) -> Option<CancelOutcome> {
        let state = self.state(task_id)?;
        if state.is_completed() {
            Some(CancelOutcome::AlreadyCompleted)
        } else {
            state.request_cancel();
            Some(CancelOutcome::Requested)
        }
    }

    /// Register a waker to be notified when the task completes.
    pub fn register_waker(&self, task_id: u64, waker: &std::task::Waker) -> bool {
        if let Some(state) = self.state(task_id) {
            let mut guard = state.waker.lock().unwrap();
            *guard = Some(waker.clone());
            true
        } else {
            false
        }
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}
