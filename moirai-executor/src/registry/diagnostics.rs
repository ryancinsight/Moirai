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
    pub fn diagnostic_block_lookup(&self) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (block_index, slot_index) = task_location(id);
        let block = self.ensure_block(block_index);
        let slot_occupied = block.get(slot_index).is_some();
        std::hint::black_box(slot_occupied);
        id
    }

    /// Diagnostic-only synchronization cost on the registration path.
    ///
    /// Replaces the former `registry_mutex_lock_only` row: the executor no
    /// longer wraps the registry in a mutex, so the cost this attributes is
    /// the shared acquire on the block directory that registration takes
    /// instead. Keeping the row lets the tradeoff recorded in ADR 0005's
    /// 2026-08-29 revision be re-measured rather than argued.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_directory_shared_acquire(&self) -> usize {
        let blocks = self
            .blocks
            .read()
            .expect("task registry block directory is never poisoned");
        let len = blocks.len();
        drop(blocks);
        len
    }

    /// Diagnostic-only slot initialization path for benchmark attribution.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_slot_initialize(&self) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (block_index, slot_index) = task_location(id);
        self.ensure_block(block_index).insert(slot_index);
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
        self.with_state(task_id, |state| state.mark_started(worker_id))
            .unwrap_or(TIMESTAMP_NOT_RECORDED)
    }

    /// Diagnostic-only completion timestamp publication on an existing slot.
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_mark_completed_since(&self, task_id: u64, started_after_ns: u64) -> Duration {
        self.with_state(task_id, |state| {
            state.mark_completed_since(started_after_ns)
        })
        .unwrap_or(Duration::ZERO)
    }

    /// Diagnostic-only production token lifecycle path with registry-local ID allocation.
    #[doc(hidden)]
    pub fn diagnostic_register_next_and_complete_with_token(&self) -> Duration {
        // SAFETY: this method completes and drops the token before returning,
        // so `self` and its block storage outlive the scheduled lease.
        let (_id, lifecycle) = unsafe { self.register_next_scheduled_task() };
        lifecycle.start(0).complete()
    }

    /// Diagnostic-only block-retaining token lifecycle for ownership-cost attribution.
    #[doc(hidden)]
    pub fn diagnostic_register_next_and_complete_with_retained_token(&self) -> Duration {
        let (_id, lifecycle) = self.register_next_task();
        lifecycle.start(0).complete()
    }

    /// Diagnostic-only production token lifecycle path with registry-local ID output.
    #[doc(hidden)]
    pub fn diagnostic_register_next_and_complete_with_token_id(&self) -> (u64, Duration) {
        // SAFETY: this method completes and drops the token before returning,
        // so `self` and its block storage outlive the scheduled lease.
        let (id, lifecycle) = unsafe { self.register_next_scheduled_task() };
        (id, lifecycle.start(0).complete())
    }
}
