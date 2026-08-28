use std::{fmt::Debug, ptr::NonNull, sync::Arc, time::Duration};

use moirai_core::Priority;

use super::state::{TaskState, TaskStateBlock};

/// Storage policy that keeps a lifecycle token's task state valid.
///
/// # Safety
///
/// Implementations must keep the returned state at a stable address until the
/// lease is dropped. The state may be shared across threads and accessed only
/// through its atomic and mutex fields.
pub(crate) unsafe trait StateLease: Debug + Send + 'static {
    fn state(&self) -> &TaskState;
}

/// Owning lease for lifecycle state that may outlive its executor.
#[derive(Debug)]
pub(crate) struct OwnedStateLease {
    block: Arc<TaskStateBlock>,
    state: NonNull<TaskState>,
}

// SAFETY: the block Arc keeps `state` allocated and stable across threads;
// TaskState mutation is confined to atomic and mutex fields.
unsafe impl Send for OwnedStateLease {}

// SAFETY: the block Arc keeps every slot address stable until this lease drops;
// registry cleanup cannot clear its slot while the `token_active` flag is set.
unsafe impl StateLease for OwnedStateLease {
    fn state(&self) -> &TaskState {
        // Reading the owner documents and preserves the lifetime dependency;
        // this reference is optimized away and performs no refcount operation.
        let _ = &self.block;
        // SAFETY: the block Arc keeps the allocation alive and registry cleanup
        // cannot clear this slot while its token-active marker remains set.
        unsafe { self.state.as_ref() }
    }
}

/// Non-owning lease for jobs whose scheduler lifetime is shorter than their
/// executor-owned registry.
#[derive(Debug)]
pub(crate) struct SchedulerStateLease {
    state: NonNull<TaskState>,
}

// SAFETY: constructors require the scheduler to drain or drop every job before
// the registry allocation is released. TaskState access is atomic/locked.
unsafe impl Send for SchedulerStateLease {}

// SAFETY: the constructor's registry-lifetime obligation keeps `state` valid;
// cleanup cannot clear it while its `token_active` flag is set.
unsafe impl StateLease for SchedulerStateLease {
    fn state(&self) -> &TaskState {
        // SAFETY: discharged by `SchedulerStateLease::new` and preserved by
        // ownership of this lease until the lifecycle token retires.
        unsafe { self.state.as_ref() }
    }
}

impl OwnedStateLease {
    fn new(block: Arc<TaskStateBlock>, state: NonNull<TaskState>) -> Self {
        Self { block, state }
    }
}

impl SchedulerStateLease {
    /// Construct a lease tied to an enclosing scheduler/registry lifetime.
    ///
    /// # Safety
    ///
    /// The owning scheduler must drain or drop the job carrying this lease
    /// before the registry block containing `state` is released.
    unsafe fn new(state: NonNull<TaskState>) -> Self {
        Self { state }
    }
}

/// Write-permission token for a registered task lifecycle.
///
/// The registry keeps shared read access to task state, while this token is
/// moved into the scheduled job as the unique authority for lifecycle mutation.
#[derive(Debug)]
pub(crate) struct TaskLifecycleToken<L: StateLease = OwnedStateLease> {
    lease: Option<L>,
}

/// Typestate token for a task that has started but has not explicitly completed.
#[derive(Debug)]
pub(crate) struct RunningTaskToken<L: StateLease = OwnedStateLease> {
    lease: Option<L>,
    pub(super) started_after_ns: u64,
    pub(super) completed: bool,
}

impl TaskLifecycleToken<OwnedStateLease> {
    pub(super) fn new_owned(block: Arc<TaskStateBlock>, state: NonNull<TaskState>) -> Self {
        Self {
            lease: Some(OwnedStateLease::new(block, state)),
        }
    }
}

impl TaskLifecycleToken<SchedulerStateLease> {
    /// Construct a token bounded by the scheduler's registry lifetime.
    ///
    /// # Safety
    ///
    /// The owning scheduler must drain or drop the job carrying this token
    /// before the registry block containing `state` is released.
    pub(super) unsafe fn new_scheduled(state: NonNull<TaskState>) -> Self {
        Self {
            // SAFETY: forwarded from this constructor's caller contract.
            lease: Some(unsafe { SchedulerStateLease::new(state) }),
        }
    }
}

impl<L: StateLease> TaskLifecycleToken<L> {
    fn state(&self) -> &TaskState {
        self.lease
            .as_ref()
            .expect("invariant: lifecycle token retains its state lease")
            .state()
    }

    /// Record the spawn priority on the task state.
    #[inline]
    pub(crate) fn set_priority(&self, priority: Priority) {
        self.state().set_priority(priority);
    }

    /// Whether a cooperative cancel has been requested for this task.
    #[inline]
    pub(crate) fn cancel_requested(&self) -> bool {
        self.state().cancel_requested()
    }

    /// Honor a pending cancel request: mark the task cancelled + completed
    /// (waking any registered waiter) without running its body.
    #[inline]
    pub(crate) fn cancel(mut self) {
        let lease = self
            .lease
            .take()
            .expect("invariant: lifecycle token retains its state lease");
        lease.state().mark_cancelled();
        lease.state().retire_token();
    }

    /// Start the task unless it was cancelled while queued.
    ///
    /// Returns `None` after marking the task cancelled (the body must not run);
    /// otherwise transfers completion authority like [`Self::start`].
    #[inline]
    pub(crate) fn start_unless_cancelled(self, worker_id: usize) -> Option<RunningTaskToken<L>> {
        if self.cancel_requested() {
            self.cancel();
            None
        } else {
            Some(self.start(worker_id))
        }
    }

    /// Mark the task as running and transfer completion authority.
    #[inline]
    pub(crate) fn start(mut self, worker_id: usize) -> RunningTaskToken<L> {
        let started_after_ns = self.state().mark_started(worker_id);
        RunningTaskToken {
            lease: self.lease.take(),
            started_after_ns,
            completed: false,
        }
    }
}

impl<L: StateLease> Drop for TaskLifecycleToken<L> {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            // A token reaches Drop only when admission or queued execution ends
            // before `start`; publish terminal completion before retiring its
            // lease so cleanup cannot reclaim the slot during publication.
            lease.state().mark_completed();
            lease.state().retire_token();
        }
    }
}

impl<L: StateLease> RunningTaskToken<L> {
    /// Mark the task as completed exactly once.
    #[inline]
    pub(crate) fn complete(mut self) -> Duration {
        self.complete_once()
            .expect("invariant: consuming completion runs exactly once")
    }

    #[inline]
    pub(super) fn complete_once(&mut self) -> Option<Duration> {
        if !self.completed {
            let execution_time = self
                .lease
                .as_ref()
                .expect("invariant: running token retains its state lease")
                .state()
                .mark_completed_since(self.started_after_ns);
            self.completed = true;
            Some(execution_time)
        } else {
            None
        }
    }
}

impl<L: StateLease> Drop for RunningTaskToken<L> {
    fn drop(&mut self) {
        self.complete_once();
        if let Some(lease) = self.lease.take() {
            lease.state().retire_token();
        }
    }
}
