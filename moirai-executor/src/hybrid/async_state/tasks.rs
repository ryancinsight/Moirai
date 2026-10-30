//! Executor ownership and wakeups for live futures.

use std::{collections::BTreeMap, sync::Mutex, task::Waker};

use moirai_core::error::{ExecutorError, ExecutorResult, TaskError};

use crate::hybrid::shutdown::{Admission, Context};

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

    pub(super) fn remove(&self, id: u64) -> bool {
        // Drop arbitrary waker state after releasing the table lock.
        let (removed, drained) = {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let removed = entries.tasks.remove(&id);
            let drained = entries.closed && entries.tasks.is_empty();
            (removed, drained)
        };
        drop(removed);
        drained
    }

    pub(crate) fn cancel(&self, id: u64) -> ExecutorResult<()> {
        let cancel = self
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .tasks
            .get(&id)
            .cloned();
        if let Some(cancel) = cancel
            && !crate::schedule::job::run_caught(|| cancel.wake())
        {
            return Err(ExecutorError::SpawnFailed(TaskError::Panicked));
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
        let mut tasks = self.close_and_snapshot();
        tasks.retain(|id, _| match registry.request_cancel(*id) {
            Some(crate::registry::CancelOutcome::Requested) => true,
            Some(crate::registry::CancelOutcome::AlreadyCompleted) => false,
            None => unreachable!("invariant: live futures have registered task ids"),
        });
        for (_, wake) in tasks {
            // Completion counts callback panics after publishing all channels.
            // Contain them here so one callback cannot skip the other wakes.
            crate::schedule::job::run_caught(|| wake.wake());
        }
    }

    pub(in crate::hybrid) fn is_empty(&self) -> bool {
        self.entries
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .tasks
            .is_empty()
    }
}

pub(super) struct Registration {
    pub(super) tasks: std::sync::Weak<AsyncTasks>,
    pub(super) id: u64,
    pub(super) admission: std::sync::Arc<Admission>,
}

pub(super) fn with_registration_context<R>(
    registration: &Registration,
    operation: impl FnOnce() -> R,
) -> R {
    Context::with(&registration.admission, operation)
}

impl Registration {
    pub(super) fn remove(&self) {
        if let Some(tasks) = self.tasks.upgrade()
            && tasks.remove(self.id)
        {
            self.admission.wake_worker();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, Mutex, Weak,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        task::{Wake, Waker},
        time::Duration,
    };

    use moirai_core::error::ExecutorError;

    use super::AsyncTasks;
    use crate::registry::{TaskLifecycleToken, TaskRegistry};

    const EVENT_DEADLINE: Duration = Duration::from_secs(5);

    enum WakeAction {
        Count,
        Gate {
            tasks: Weak<AsyncTasks>,
            lock_observed: mpsc::SyncSender<bool>,
            entered: mpsc::SyncSender<()>,
            release: Mutex<mpsc::Receiver<()>>,
        },
        Panic,
    }

    struct ObservedWake {
        calls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        action: WakeAction,
    }

    impl Wake for ObservedWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.calls.fetch_add(1, Ordering::AcqRel);
            match &self.action {
                WakeAction::Count => {}
                WakeAction::Gate {
                    tasks,
                    lock_observed,
                    entered,
                    release,
                } => {
                    let rejected = tasks
                        .upgrade()
                        .expect("shutdown retains the async task table")
                        .register(u64::MAX, Waker::noop().clone());
                    lock_observed
                        .send(matches!(rejected, Err(ExecutorError::ShuttingDown)))
                        .expect("lock observer remains connected");
                    entered
                        .send(())
                        .expect("callback observer remains connected");
                    release
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .recv()
                        .expect("callback release remains connected");
                }
                WakeAction::Panic => panic!("shutdown wake panic"),
            }
        }
    }

    impl Drop for ObservedWake {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn observed_waker(action: WakeAction) -> (Waker, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let waker = Waker::from(Arc::new(ObservedWake {
            calls: Arc::clone(&calls),
            drops: Arc::clone(&drops),
            action,
        }));
        (waker, calls, drops)
    }

    fn registered_task(registry: &TaskRegistry) -> (u64, TaskLifecycleToken) {
        registry.register_next_task()
    }

    #[test]
    fn shutdown_requests_every_cancel_before_first_callback() {
        let registry = Arc::new(TaskRegistry::new());
        let tasks = Arc::new(AsyncTasks::default());
        let (first_id, first_lifecycle) = registered_task(&registry);
        let (second_id, second_lifecycle) = registered_task(&registry);
        let (completed_id, completed_lifecycle) = registered_task(&registry);
        completed_lifecycle.complete().notify();

        let (lock_sender, lock_observed) = mpsc::sync_channel(1);
        let (entered_sender, entered) = mpsc::sync_channel(1);
        let (release_sender, release) = mpsc::sync_channel(0);
        let (first, first_calls, first_drops) = observed_waker(WakeAction::Gate {
            tasks: Arc::downgrade(&tasks),
            lock_observed: lock_sender,
            entered: entered_sender,
            release: Mutex::new(release),
        });
        let (second, second_calls, second_drops) = observed_waker(WakeAction::Count);
        let (completed, completed_calls, completed_drops) = observed_waker(WakeAction::Count);
        tasks
            .register(first_id, first)
            .expect("invariant: open task table accepts the first task");
        tasks
            .register(second_id, second)
            .expect("invariant: open task table accepts the second task");
        tasks
            .register(completed_id, completed)
            .expect("invariant: open task table accepts the completed task");

        let shutdown_tasks = Arc::clone(&tasks);
        let shutdown_registry = Arc::clone(&registry);
        let shutdown = std::thread::spawn(move || {
            shutdown_tasks.shutdown(&shutdown_registry);
        });
        assert!(
            lock_observed
                .recv_timeout(EVENT_DEADLINE)
                .expect("invariant: callback lock probe arrives before the event deadline")
        );
        entered
            .recv_timeout(EVENT_DEADLINE)
            .expect("invariant: callback enters before the event deadline");
        let second_cancelled_before_release = second_lifecycle.cancel_requested();
        release_sender
            .send(())
            .expect("invariant: gated callback remains connected");
        shutdown
            .join()
            .expect("invariant: shutdown contains callback panics");

        assert!(second_cancelled_before_release);
        assert!(first_lifecycle.cancel_requested());
        assert_eq!(first_calls.load(Ordering::Acquire), 1);
        assert_eq!(second_calls.load(Ordering::Acquire), 1);
        assert_eq!(completed_calls.load(Ordering::Acquire), 0);
        assert_eq!(first_drops.load(Ordering::Acquire), 0);
        assert_eq!(second_drops.load(Ordering::Acquire), 0);
        assert_eq!(completed_drops.load(Ordering::Acquire), 0);

        first_lifecycle.cancel().notify();
        second_lifecycle.cancel().notify();
        assert!(registry.is_completed(first_id));
        assert!(registry.is_completed(second_id));
        assert!(registry.is_completed(completed_id));
        drop(tasks);
        assert_eq!(first_drops.load(Ordering::Acquire), 1);
        assert_eq!(second_drops.load(Ordering::Acquire), 1);
        assert_eq!(completed_drops.load(Ordering::Acquire), 1);
    }

    #[test]
    fn shutdown_contains_panicking_callback_and_wakes_remaining_tasks() {
        let registry = TaskRegistry::new();
        let tasks = AsyncTasks::default();
        let (panicking_id, panicking_lifecycle) = registered_task(&registry);
        let (remaining_id, remaining_lifecycle) = registered_task(&registry);
        let (panicking, panicking_calls, _) = observed_waker(WakeAction::Panic);
        let (remaining, remaining_calls, _) = observed_waker(WakeAction::Count);
        tasks
            .register(panicking_id, panicking)
            .expect("invariant: open task table accepts the panicking task");
        tasks
            .register(remaining_id, remaining)
            .expect("invariant: open task table accepts the remaining task");

        tasks.shutdown(&registry);

        assert!(panicking_lifecycle.cancel_requested());
        assert!(remaining_lifecycle.cancel_requested());
        assert_eq!(panicking_calls.load(Ordering::Acquire), 1);
        assert_eq!(remaining_calls.load(Ordering::Acquire), 1);
        panicking_lifecycle.cancel().notify();
        remaining_lifecycle.cancel().notify();
    }
}
