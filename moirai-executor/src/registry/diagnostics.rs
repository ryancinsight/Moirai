#[cfg(feature = "registry-diagnostics")]
use std::time::Duration;

#[cfg(feature = "registry-diagnostics")]
use super::registry::TaskRegistry;
#[cfg(feature = "registry-diagnostics")]
use super::state::{task_location, TaskState, TIMESTAMP_NOT_RECORDED};

#[cfg(feature = "registry-diagnostics")]
impl TaskRegistry {
    /// Diagnostic-only block lookup path for benchmark attribution.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_block_lookup(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let (block_index, slot_index) = task_location(id);
        self.ensure_block(block_index);
        let slot_occupied = self.blocks[block_index].get(slot_index).is_some();
        std::hint::black_box(slot_occupied);
        id
    }

    /// Diagnostic-only slot initialization path for benchmark attribution.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_slot_initialize(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let (block_index, slot_index) = task_location(id);
        self.ensure_block(block_index);
        let _ = self.blocks[block_index].insert(slot_index);
        id
    }

    /// Diagnostic-only lifecycle timestamp publication path for benchmark attribution.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_lifecycle_timestamp_publication() -> Duration {
        let state = TaskState::new();
        let started_after_ns = state.mark_started(0);
        state.mark_completed_since(started_after_ns)
    }

    /// Diagnostic-only task-state construction path for benchmark attribution.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_task_state_construct() -> usize {
        let state = TaskState::new();
        std::hint::black_box(state);
        core::mem::size_of::<TaskState>()
    }

    /// Diagnostic-only start timestamp publication on an existing slot.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_mark_started(&self, task_id: u64, worker_id: usize) -> u64 {
        self.state(task_id).map_or(TIMESTAMP_NOT_RECORDED, |state| {
            state.mark_started(worker_id)
        })
    }

    /// Diagnostic-only completion timestamp publication on an existing slot.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_mark_completed_since(&self, task_id: u64, started_after_ns: u64) -> Duration {
        self.state(task_id).map_or(Duration::ZERO, |state| {
            state.mark_completed_since(started_after_ns)
        })
    }

    /// Diagnostic-only production token lifecycle path with registry-local ID allocation.
    #[doc(hidden)]
    pub fn diagnostic_register_next_and_complete_with_token(&mut self) -> Duration {
        let id = self.next_id;
        let lifecycle = self.register_task_with_id(id);
        lifecycle.start(0).complete()
    }

    /// Diagnostic-only production token lifecycle path with registry-local ID output.
    #[doc(hidden)]
    pub fn diagnostic_register_next_and_complete_with_token_id(&mut self) -> (u64, Duration) {
        let id = self.next_id;
        let lifecycle = self.register_task_with_id(id);
        (id, lifecycle.start(0).complete())
    }
}
