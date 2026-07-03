use std::time::Duration;

use super::super::task::TaskMetadata;
use super::state::{task_location, TaskState, TaskStateBlock};
use super::token::TaskLifecycleToken;

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
    pub(super) blocks: Vec<TaskStateBlock>,
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
    pub(crate) fn register_next_task(&mut self) -> (u64, TaskLifecycleToken) {
        let id = self.next_id;
        let lifecycle = self.register_task_with_id(id);
        (id, lifecycle)
    }

    /// Register a task with an externally allocated ID.
    pub(crate) fn register_task_with_id(&mut self, id: u64) -> TaskLifecycleToken {
        self.next_id = self.next_id.max(id.saturating_add(1));
        let (block_index, slot_index) = task_location(id);
        self.ensure_block(block_index);

        let block = &self.blocks[block_index];
        assert!(
            !block
                .get(slot_index)
                .is_some_and(|state| !state.is_completed()),
            "task ID must not be re-registered while active"
        );

        TaskLifecycleToken {
            state: block.insert(slot_index),
        }
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
        let cutoff = std::time::Instant::now() - older_than;
        for block in &self.blocks {
            for slot_index in 0..block.len() {
                if block
                    .get(slot_index)
                    .and_then(TaskState::completed_at)
                    .is_some_and(|completed| completed <= cutoff)
                {
                    block.clear(slot_index);
                }
            }
        }

        while self.blocks.last().is_some_and(TaskStateBlock::is_empty) {
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
            self.blocks.push(TaskStateBlock::new());
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
