//! Accounting for the jobs of one borrowing scope: how many are pending, whether
//! one panicked, and whether one was dropped without running.

use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
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

    pub(super) fn complete_task(&self) {
        loop {
            let pending = self.pending_tasks.load(Ordering::Acquire);
            debug_assert!(pending > 0, "scoped completion count must not underflow");

            if pending == 1 {
                // Hold the wait lock before publishing zero. Every waiter
                // acquires this lock after observing zero, so the stack-owned
                // scope state cannot be destroyed until this completion token
                // has finished its last access to the mutex and condition
                // variable.
                let _guard = lock_mutex(&self.wait_lock);
                if self
                    .pending_tasks
                    .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    self.wait_signal.notify_all();
                    return;
                }
                continue;
            }

            if self
                .pending_tasks
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
    state: &'scope SchedulerScopeState,
    ran: bool,
}

impl<'scope> ScopedTaskCompletion<'scope> {
    pub(super) fn new(state: &'scope SchedulerScopeState) -> Self {
        Self { state, ran: false }
    }

    /// Records that the job ran, successfully or by panicking, and releases it.
    pub(super) fn finish(mut self, succeeded: bool) {
        if !succeeded {
            self.state().mark_failed();
        }
        self.ran = true;
    }

    pub(super) fn state(&self) -> &SchedulerScopeState {
        self.state
    }
}

impl Drop for ScopedTaskCompletion<'_> {
    fn drop(&mut self) {
        if !self.ran {
            self.state().unrun_jobs.fetch_add(1, Ordering::AcqRel);
        }
        self.state().complete_task();
    }
}
