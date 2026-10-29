//! Executor ownership and wakeups for live futures.

use std::{collections::BTreeMap, sync::Mutex, task::Waker};

use moirai_core::error::{ExecutorError, ExecutorResult, TaskError};

#[derive(Default)]
struct Entries {
    closed: bool,
    tasks: BTreeMap<u64, Waker>,
}

#[derive(Default)]
pub(crate) struct AsyncTasks {
    entries: Mutex<Entries>,
}

impl AsyncTasks {
    pub(super) fn register(&self, id: u64, cancel: Waker) -> ExecutorResult<()> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if entries.closed {
            return Err(ExecutorError::ShuttingDown);
        }
        let previous = entries.tasks.insert(id, cancel);
        assert!(
            previous.is_none(),
            "invariant: task registry issues unique ids"
        );
        Ok(())
    }

    pub(super) fn remove(&self, id: u64) {
        // Drop arbitrary waker state after releasing the table lock.
        let removed = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .tasks
            .remove(&id);
        drop(removed);
    }

    pub(crate) fn cancel(&self, id: u64) -> ExecutorResult<()> {
        let cancel = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .tasks
            .get(&id)
            .cloned();
        if let Some(cancel) = cancel {
            if !crate::schedule::job::run_caught(|| cancel.wake()) {
                return Err(ExecutorError::SpawnFailed(TaskError::Panicked));
            }
        }
        Ok(())
    }

    pub(in crate::hybrid) fn close_and_snapshot(&self) -> BTreeMap<u64, Waker> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        entries.closed = true;
        entries.tasks.clone()
    }

    pub(crate) fn shutdown(&self, registry: &crate::registry::TaskRegistry) {
        let tasks = self.close_and_snapshot();
        for (id, wake) in tasks {
            match registry.request_cancel(id) {
                Some(crate::registry::CancelOutcome::Requested) => {
                    // Completion counts callback panics after publishing all
                    // channels. Contain them here so shutdown still cancels
                    // every remaining task and reaches scheduler teardown.
                    crate::schedule::job::run_caught(|| wake.wake());
                }
                Some(crate::registry::CancelOutcome::AlreadyCompleted) => {}
                None => unreachable!("invariant: live futures have registered task ids"),
            }
        }
    }
}

pub(super) struct Registration {
    pub(super) tasks: std::sync::Weak<AsyncTasks>,
    pub(super) id: u64,
}

impl Registration {
    pub(super) fn remove(&self) {
        if let Some(tasks) = self.tasks.upgrade() {
            tasks.remove(self.id);
        }
    }
}
