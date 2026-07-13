use std::{mem::ManuallyDrop, ptr::NonNull, time::Duration};

use moirai_core::Priority;

use super::state::TaskState;

/// Write-permission token for a registered task lifecycle.
///
/// The registry keeps shared read access to task state, while this token is
/// moved into the scheduled job as the unique authority for lifecycle mutation.
#[derive(Debug)]
pub(crate) struct TaskLifecycleToken {
    pub(super) state: NonNull<TaskState>,
}

/// Typestate token for a task that has started but has not explicitly completed.
#[derive(Debug)]
pub(crate) struct RunningTaskToken {
    pub(super) state: NonNull<TaskState>,
    pub(super) started_after_ns: u64,
    pub(super) completed: bool,
}

// Safety: lifecycle tokens point to a `TaskState` owned by a registry block.
// Blocks never move or deallocate individual initialized slots while a task can
// still be running. `TaskState` mutation is atomic, and scheduler shutdown
// drains jobs before the owning registry drops.
unsafe impl Send for TaskLifecycleToken {}

// Safety: see `TaskLifecycleToken`; this typestate token is the unique
// completion authority for the same stable task-state slot.
unsafe impl Send for RunningTaskToken {}

impl TaskLifecycleToken {
    /// Record the spawn priority on the task state.
    #[inline]
    pub(crate) fn set_priority(&self, priority: Priority) {
        // Safety: lifecycle tokens are created only from initialized registry
        // slots; the write touches an interior-mutable atomic field.
        unsafe { self.state.as_ref().set_priority(priority) }
    }

    /// Whether a cooperative cancel has been requested for this task.
    #[inline]
    pub(crate) fn cancel_requested(&self) -> bool {
        // Safety: as in `set_priority` — atomic read of an initialized slot.
        unsafe { self.state.as_ref().cancel_requested() }
    }

    /// Honor a pending cancel request: mark the task cancelled + completed
    /// (waking any registered waiter) without running its body.
    #[inline]
    pub(crate) fn cancel(self) {
        let token = ManuallyDrop::new(self);
        // Safety: as in `start` — the token is the unique lifecycle authority
        // for an initialized, address-stable registry slot.
        unsafe { token.state.as_ref().mark_cancelled() }
    }

    /// Start the task unless it was cancelled while queued.
    ///
    /// Returns `None` after marking the task cancelled (the body must not run);
    /// otherwise transfers completion authority like [`Self::start`].
    #[inline]
    pub(crate) fn start_unless_cancelled(self, worker_id: usize) -> Option<RunningTaskToken> {
        if self.cancel_requested() {
            self.cancel();
            None
        } else {
            Some(self.start(worker_id))
        }
    }

    /// Mark the task as running and transfer completion authority.
    #[inline]
    pub(crate) fn start(self, worker_id: usize) -> RunningTaskToken {
        let token = ManuallyDrop::new(self);
        let state = token.state;
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

impl Drop for TaskLifecycleToken {
    fn drop(&mut self) {
        // A token reaches Drop only when admission or queued execution ends
        // before `start`; publish terminal completion so registry slots and
        // waiters cannot remain permanently active.
        unsafe { self.state.as_ref().mark_completed() }
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
    pub(super) fn complete_once(&mut self) -> Option<Duration> {
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
