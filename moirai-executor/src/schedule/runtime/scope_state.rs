//! Accounting for the jobs of one borrowing scope: how many are pending, whether
//! one panicked, and whether one was dropped without running.

use std::{
    marker::PhantomData,
    ptr::NonNull,
    sync::{
        Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use moirai_core::{
    ExecutorError,
    error::{ExecutorResult, TaskError},
};

use super::worker::lock_mutex;

pub(super) struct SchedulerScopeState {
    pub(super) pending_tasks: AtomicUsize,
    pub(super) failed_tasks: AtomicBool,
    /// Scoped jobs whose completion token was dropped before the job ran.
    unrun_jobs: AtomicUsize,
    pub(super) wait_lock: Mutex<()>,
    pub(super) wait_signal: Condvar,
}

impl SchedulerScopeState {
    pub(super) fn new() -> Self {
        Self {
            pending_tasks: AtomicUsize::new(0),
            failed_tasks: AtomicBool::new(false),
            unrun_jobs: AtomicUsize::new(0),
            wait_lock: Mutex::new(()),
            wait_signal: Condvar::new(),
        }
    }

    pub(super) fn register_task(&self) {
        self.pending_tasks.fetch_add(1, Ordering::AcqRel);
    }

    /// Releases one registered job.
    ///
    /// The last release publishes zero, after which the waiter may return and
    /// destroy the stack-owned state while this call is still unwinding its own
    /// frames. Nothing here therefore holds a reference to the whole state, or
    /// to any field's padding, across that point: every borrow covers one field
    /// for one call, and the final borrows are the wait lock, released as the
    /// last access, and the condition variable, used while the lock is held.
    ///
    /// # Safety
    ///
    /// `this` points to a live state whose pending count includes one job
    /// registered for the caller, which this call releases exactly once.
    pub(super) unsafe fn complete_task(this: NonNull<Self>) {
        let state = this.as_ptr();
        // SAFETY: the caller's registered job keeps the state alive until the
        // release below publishes zero.
        let pending_tasks = unsafe { &(*state).pending_tasks };
        loop {
            let pending = pending_tasks.load(Ordering::Acquire);
            debug_assert!(pending > 0, "scoped completion count must not underflow");

            if pending == 1 {
                // Hold the wait lock before publishing zero. Every waiter
                // acquires this lock after observing zero, so the stack-owned
                // scope state cannot be destroyed until this completion token
                // has finished its last access to the mutex and condition
                // variable.
                //
                // SAFETY: the count is still nonzero, so the state is live.
                let _guard = lock_mutex(unsafe { &(*state).wait_lock });
                if pending_tasks
                    .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    // SAFETY: the wait lock is held, so the waiter cannot
                    // destroy the state before the guard drops.
                    unsafe { &(*state).wait_signal }.notify_all();
                    return;
                }
                continue;
            }

            if pending_tasks
                .compare_exchange_weak(pending, pending - 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    pub(super) fn wait(&self) {
        // Spin-wait for a short duration before acquiring the lock and parking
        for _ in 0..131_072 {
            if self.pending_tasks.load(Ordering::Acquire) == 0 {
                break;
            }
            core::hint::spin_loop();
        }

        // The final completion publishes zero while holding this lock. Taking
        // it after the acquire load forms the lifetime handshake that proves
        // the completion token no longer accesses this stack-owned state.
        let mut guard = lock_mutex(&self.wait_lock);
        while self.pending_tasks.load(Ordering::Acquire) != 0 {
            guard = self
                .wait_signal
                .wait(guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    pub(super) fn mark_failed(&self) {
        self.failed_tasks.store(true, Ordering::Release);
    }

    /// Reports whether a job panicked. Read after the scope has drained.
    pub(super) fn has_panicked(&self) -> bool {
        self.failed_tasks.load(Ordering::Acquire)
    }

    /// Reports whether a job was dropped without running and the drop was not
    /// forgiven as a refusal. Read after the scope has drained.
    pub(super) fn has_unrun_job(&self) -> bool {
        self.unrun_jobs.load(Ordering::Acquire) != 0
    }

    /// Fails the scope when an admitted job was dropped without running, so a
    /// caller never reads result slots the job was to fill. A refused job's
    /// own error, already returned to the submitter, takes precedence.
    pub(super) fn unrun_job_result(&self) -> ExecutorResult<()> {
        if self.has_unrun_job() {
            Err(ExecutorError::SpawnFailed(TaskError::Cancelled))
        } else {
            Ok(())
        }
    }

    /// Cancels the unrun mark of a job the scheduler refused at admission.
    ///
    /// Admission drops the refused job, which counts as an unrun drop. The
    /// submitter learns of the refusal from the returned error and answers it
    /// (an inline run, or the error itself), so that drop is not a lost job. A
    /// job dropped after admission has no such answer and stays counted.
    pub(super) fn forgive_refused_job(&self) {
        self.unrun_jobs.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Releases one registered scoped job when it runs or is dropped.
///
/// A token dropped without [`Self::finish`] belongs to a job that never ran, so
/// the scope it guards cannot claim its work happened.
pub(super) struct ScopedTaskCompletion<'scope> {
    // A pointer, not a reference: a reference field is protected for every call
    // that receives the token by value, and the release in `Drop` lets the
    // waiter destroy the state before `finish` returns.
    state: NonNull<SchedulerScopeState>,
    ran: bool,
    _scope: PhantomData<&'scope SchedulerScopeState>,
}

// SAFETY: the token only reaches the state through its atomics, mutex and
// condition variable, all of which are `Sync`, and the state outlives every
// token by construction (the scope waits for its pending count to reach zero).
unsafe impl Send for ScopedTaskCompletion<'_> {}

impl<'scope> ScopedTaskCompletion<'scope> {
    pub(super) fn new(state: &'scope SchedulerScopeState) -> Self {
        Self {
            state: NonNull::from(state),
            ran: false,
            _scope: PhantomData,
        }
    }

    /// Records that the job ran, successfully or by panicking, and releases it.
    pub(super) fn finish(mut self, succeeded: bool) {
        if !succeeded {
            self.state().mark_failed();
        }
        self.ran = true;
    }

    fn state(&self) -> &SchedulerScopeState {
        // SAFETY: this token's registered job is pending until it drops, so the
        // scope has not returned and the state is live.
        unsafe { self.state.as_ref() }
    }
}

impl Drop for ScopedTaskCompletion<'_> {
    fn drop(&mut self) {
        if !self.ran {
            self.state().unrun_jobs.fetch_add(1, Ordering::AcqRel);
        }
        // SAFETY: this token owns one registered job and releases it once.
        unsafe { SchedulerScopeState::complete_task(self.state) };
    }
}
