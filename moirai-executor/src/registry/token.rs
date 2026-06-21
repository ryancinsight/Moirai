use std::{mem::ManuallyDrop, ptr::NonNull, time::Duration};

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
