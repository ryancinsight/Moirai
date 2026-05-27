//! Task registry for tracking and managing task lifecycle.

use std::{
    mem::ManuallyDrop,
    ptr::NonNull,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use super::task::TaskMetadata;

const NO_WORKER: usize = usize::MAX;
const TIMESTAMP_NOT_RECORDED: u64 = u64::MAX;
const TASK_STATE_BLOCK_SIZE: usize = 1024;

/// Public task registry facade used by executor lifecycle tracking and tests.
#[derive(Debug)]
pub struct TaskRegistry {
    blocks: Vec<TaskStateBlock>,
    next_id: u64,
}

#[derive(Debug)]
struct TaskStateBlock {
    slots: Box<[Option<TaskState>]>,
}

/// Shared lifecycle state for one task.
#[derive(Debug)]
pub(crate) struct TaskState {
    created_at: Instant,
    started_after_ns: AtomicU64,
    completed_after_ns: AtomicU64,
    worker_id: AtomicUsize,
}

/// Write-permission token for a registered task lifecycle.
///
/// The registry keeps shared read access to task state, while this token is
/// moved into the scheduled job as the unique authority for lifecycle mutation.
#[derive(Debug)]
pub(crate) struct TaskLifecycleToken {
    state: NonNull<TaskState>,
}

/// Typestate token for a task that has started but has not explicitly completed.
#[derive(Debug)]
pub(crate) struct RunningTaskToken {
    state: NonNull<TaskState>,
    started_after_ns: u64,
    completed: bool,
}

// Safety: lifecycle tokens point to a `TaskState` owned by a registry block.
// Blocks never move or deallocate individual initialized slots while a task can
// still be running. `TaskState` mutation is atomic, and scheduler shutdown
// drains jobs before the owning registry drops.
unsafe impl Send for TaskLifecycleToken {}

// Safety: see `TaskLifecycleToken`; this typestate token is the unique
// completion authority for the same stable task-state slot.
unsafe impl Send for RunningTaskToken {}

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

        let slot = &mut self.blocks[block_index].slots[slot_index];
        assert!(
            !slot.as_ref().is_some_and(|state| !state.is_completed()),
            "task ID must not be re-registered while active"
        );

        let state = NonNull::from(slot.insert(TaskState::new()));
        TaskLifecycleToken { state }
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
        let cutoff = Instant::now() - older_than;
        for block in &mut self.blocks {
            for slot in &mut *block.slots {
                if slot
                    .as_ref()
                    .and_then(TaskState::completed_at)
                    .is_some_and(|completed| completed <= cutoff)
                {
                    *slot = None;
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
            .flat_map(|block| block.slots.iter())
            .filter(|slot| slot.as_ref().is_some_and(|state| !state.is_completed()))
            .count()
    }

    /// Get count of completed tasks.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|block| block.slots.iter())
            .filter(|slot| slot.as_ref().is_some_and(TaskState::is_completed))
            .count()
    }

    /// Diagnostic-only block lookup path for benchmark attribution.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_block_lookup(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let (block_index, slot_index) = task_location(id);
        self.ensure_block(block_index);
        let slot_occupied = self.blocks[block_index].slots[slot_index].is_some();
        std::hint::black_box(slot_occupied);
        id
    }

    /// Diagnostic-only slot initialization path for benchmark attribution.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_slot_initialize(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let (block_index, slot_index) = task_location(id);
        self.ensure_block(block_index);
        let _ = self.blocks[block_index].slots[slot_index].insert(TaskState::new());
        id
    }

    /// Diagnostic-only lifecycle timestamp publication path for benchmark attribution.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_lifecycle_timestamp_publication() -> Duration {
        let state = TaskState::new();
        let started_after_ns = state.mark_started(0);
        state.mark_completed_since(started_after_ns)
    }

    /// Diagnostic-only task-state construction path for benchmark attribution.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_task_state_construct() -> usize {
        let state = TaskState::new();
        std::hint::black_box(state);
        core::mem::size_of::<TaskState>()
    }

    /// Diagnostic-only start timestamp publication on an existing slot.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_mark_started(&self, task_id: u64, worker_id: usize) -> u64 {
        self.state(task_id).map_or(TIMESTAMP_NOT_RECORDED, |state| {
            state.mark_started(worker_id)
        })
    }

    /// Diagnostic-only completion timestamp publication on an existing slot.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn diagnostic_mark_completed_since(&self, task_id: u64, started_after_ns: u64) -> Duration {
        self.state(task_id).map_or(Duration::ZERO, |state| {
            state.mark_completed_since(started_after_ns)
        })
    }

    /// Diagnostic-only production token lifecycle path for wrapper attribution.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    pub fn diagnostic_register_external_task_with_id(&mut self, id: u64) -> u64 {
        let _lifecycle = self.register_task_with_id(id);
        id
    }

    /// Diagnostic-only production token lifecycle path for wrapper attribution.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    pub fn diagnostic_restart_and_complete_with_token(&mut self, id: u64) -> Duration {
        let lifecycle = self.register_task_with_id(id);
        lifecycle.start(0).complete()
    }

    /// Diagnostic-only production token lifecycle path with registry-local ID allocation.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    pub fn diagnostic_register_next_and_complete_with_token(&mut self) -> Duration {
        let id = self.next_id;
        let lifecycle = self.register_task_with_id(id);
        lifecycle.start(0).complete()
    }

    /// Diagnostic-only production token lifecycle path with registry-local ID output.
    #[cfg(feature = "registry-diagnostics")]
    #[doc(hidden)]
    pub fn diagnostic_register_next_and_complete_with_token_id(&mut self) -> (u64, Duration) {
        let id = self.next_id;
        let lifecycle = self.register_task_with_id(id);
        (id, lifecycle.start(0).complete())
    }

    fn ensure_block(&mut self, block_index: usize) {
        while self.blocks.len() <= block_index {
            self.blocks.push(TaskStateBlock::new());
        }
    }

    fn state(&self, task_id: u64) -> Option<&TaskState> {
        let (block_index, slot_index) = task_location(task_id);
        self.blocks
            .get(block_index)?
            .slots
            .get(slot_index)?
            .as_ref()
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskState {
    #[inline]
    fn new() -> Self {
        Self {
            created_at: Instant::now(),
            started_after_ns: AtomicU64::new(TIMESTAMP_NOT_RECORDED),
            completed_after_ns: AtomicU64::new(TIMESTAMP_NOT_RECORDED),
            worker_id: AtomicUsize::new(NO_WORKER),
        }
    }

    #[inline]
    fn mark_started(&self, worker_id: usize) -> u64 {
        let started_after_ns = elapsed_nanos_since(self.created_at);
        self.started_after_ns
            .store(started_after_ns, Ordering::Release);
        self.worker_id.store(worker_id, Ordering::Release);
        started_after_ns
    }

    #[inline]
    fn mark_completed_since(&self, started_after_ns: u64) -> Duration {
        let completed_after_ns = elapsed_nanos_since(self.created_at);
        self.completed_after_ns
            .store(completed_after_ns, Ordering::Release);

        debug_assert!(
            completed_after_ns >= started_after_ns,
            "monotonic lifecycle completion offset must not precede start offset"
        );
        Duration::from_nanos(completed_after_ns - started_after_ns)
    }

    fn mark_completed(&self) {
        let started_after_ns = self.started_after_ns.load(Ordering::Acquire);
        let started_after_ns = if started_after_ns == TIMESTAMP_NOT_RECORDED {
            elapsed_nanos_since(self.created_at)
        } else {
            started_after_ns
        };
        self.mark_completed_since(started_after_ns);
    }

    fn is_completed(&self) -> bool {
        self.completed_after_ns.load(Ordering::Acquire) != TIMESTAMP_NOT_RECORDED
    }

    fn completed_at(&self) -> Option<Instant> {
        instant_from_offset(
            self.created_at,
            self.completed_after_ns.load(Ordering::Acquire),
        )
    }

    fn snapshot(&self, id: u64) -> TaskMetadata {
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
    fn new() -> Self {
        let slots = std::iter::repeat_with(|| None)
            .take(TASK_STATE_BLOCK_SIZE)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self { slots }
    }

    fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }
}

impl TaskLifecycleToken {
    /// Mark the task as running and transfer completion authority.
    #[inline]
    pub(crate) fn start(self, worker_id: usize) -> RunningTaskToken {
        let state = self.state;
        // Safety: lifecycle tokens are created only from initialized registry
        // slots. Registry blocks keep slot addresses stable for running tasks.
        let started_after_ns = unsafe { state.as_ref().mark_started(worker_id) };
        RunningTaskToken {
            state,
            started_after_ns,
            completed: false,
        }
    }
}

impl RunningTaskToken {
    /// Mark the task as completed exactly once.
    #[inline]
    pub(crate) fn complete(self) -> Duration {
        let mut token = ManuallyDrop::new(self);
        token.completed = true;
        // Safety: the running token is the unique completion authority for a
        // stable registry-owned task-state slot. Ownership of `self` proves no
        // later drop path can publish completion again.
        unsafe {
            token
                .state
                .as_ref()
                .mark_completed_since(token.started_after_ns)
        }
    }

    #[inline]
    fn complete_once(&mut self) -> Option<Duration> {
        if !self.completed {
            // Safety: the running token is the unique completion authority for
            // a stable registry-owned task-state slot.
            let execution_time = unsafe {
                self.state
                    .as_ref()
                    .mark_completed_since(self.started_after_ns)
            };
            self.completed = true;
            Some(execution_time)
        } else {
            None
        }
    }
}

impl Drop for RunningTaskToken {
    fn drop(&mut self) {
        self.complete_once();
    }
}

#[inline]
fn elapsed_nanos_since(origin: Instant) -> u64 {
    let elapsed = origin.elapsed().as_nanos();
    elapsed.min(u128::from(TIMESTAMP_NOT_RECORDED - 1)) as u64
}

fn instant_from_offset(origin: Instant, offset_ns: u64) -> Option<Instant> {
    if offset_ns == TIMESTAMP_NOT_RECORDED {
        None
    } else {
        origin.checked_add(Duration::from_nanos(offset_ns))
    }
}

fn task_location(id: u64) -> (usize, usize) {
    let index = usize::try_from(id).expect("task ID must fit in usize");
    (index / TASK_STATE_BLOCK_SIZE, index % TASK_STATE_BLOCK_SIZE)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{TaskRegistry, TASK_STATE_BLOCK_SIZE};

    #[test]
    fn lifecycle_token_records_started_and_completed_metadata() {
        let mut registry = TaskRegistry::new();
        let lifecycle = registry.register_task_with_id(7);

        let running = lifecycle.start(3);
        let started = registry.get_metadata(7).unwrap();
        assert_eq!(started.id, 7);
        assert_eq!(started.worker_id, Some(3));
        assert!(started.started_at.is_some());
        assert!(started.completed_at.is_none());

        let execution_time = running.complete();

        let completed = registry.get_metadata(7).unwrap();
        assert!(completed.completed_at.is_some());
        assert!(completed.execution_duration().is_some());
        assert_eq!(completed.execution_duration(), Some(execution_time));
        assert!(registry.is_completed(7));
    }

    #[test]
    fn register_next_task_returns_id_and_lifecycle_token() {
        let mut registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();

        let running = lifecycle.start(2);
        let execution_time = running.complete();

        let metadata = registry.get_metadata(task_id).unwrap();
        assert_eq!(metadata.id, task_id);
        assert_eq!(metadata.worker_id, Some(2));
        assert!(metadata.started_at.is_some());
        assert!(metadata.completed_at.is_some());
        assert_eq!(metadata.execution_duration(), Some(execution_time));
    }

    #[test]
    fn running_lifecycle_token_completes_on_drop() {
        let mut registry = TaskRegistry::new();
        let lifecycle = registry.register_task_with_id(8);

        drop(lifecycle.start(1));

        assert!(registry.is_completed(8));
    }

    #[test]
    fn lifecycle_blocks_preserve_sparse_metadata_and_cleanup_completed_slots() {
        let mut registry = TaskRegistry::new();
        let first_id = (TASK_STATE_BLOCK_SIZE - 1) as u64;
        let second_id = TASK_STATE_BLOCK_SIZE as u64;

        let first = registry.register_task_with_id(first_id).start(0);
        first.complete();

        let second = registry.register_task_with_id(second_id).start(1);

        assert!(registry.is_completed(first_id));
        assert_eq!(registry.get_metadata(second_id).unwrap().worker_id, Some(1));
        assert_eq!(registry.active_count(), 1);
        assert_eq!(registry.completed_count(), 1);

        registry.cleanup_completed(Duration::ZERO);

        assert!(registry.get_metadata(first_id).is_none());
        assert!(registry.get_metadata(second_id).is_some());

        second.complete();
        assert!(registry.is_completed(second_id));
    }

    #[test]
    fn cleanup_completed_releases_empty_trailing_blocks() {
        let mut registry = TaskRegistry::new();
        let first_id = (TASK_STATE_BLOCK_SIZE - 1) as u64;
        let second_id = TASK_STATE_BLOCK_SIZE as u64;

        registry.register_task_with_id(first_id).start(0).complete();
        registry
            .register_task_with_id(second_id)
            .start(0)
            .complete();
        assert_eq!(registry.blocks.len(), 2);

        registry.cleanup_completed(Duration::ZERO);

        assert!(registry.blocks.is_empty());
        assert!(registry.get_metadata(first_id).is_none());
        assert!(registry.get_metadata(second_id).is_none());
    }

    #[test]
    #[should_panic(expected = "task ID must not be re-registered while active")]
    fn lifecycle_registry_rejects_active_id_reuse() {
        let mut registry = TaskRegistry::new();
        let _running = registry.register_task_with_id(21).start(0);

        let _duplicate = registry.register_task_with_id(21);
    }
}
