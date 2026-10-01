//! Admission of poll jobs: the enqueue obligation and its discharge.

use std::{
    future::Future,
    sync::{Arc, atomic::Ordering},
};

use moirai_core::{
    Priority,
    error::{ExecutorError, ExecutorResult},
};

use super::{
    future_state::AsyncFutureState,
    inline_poll::InlinePollDepthGuard,
    phase::{ASYNC_IDLE, ASYNC_NOTIFIED, ASYNC_POLLING, ASYNC_QUEUED},
};
use crate::{
    registry::StateLease,
    schedule::{AsyncTask, WorkSubmit},
};

impl<S, F, L> AsyncFutureState<S, F, L>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    L: StateLease,
{
    /// Absorb a wake into the state machine, claiming the enqueue obligation.
    ///
    /// Returns `true` when this caller transitioned `IDLE → QUEUED` and now
    /// owns admitting exactly one poll job (module docs: enqueue obligation).
    /// Every other outcome hands the wake to a transition another party owns:
    /// `POLLING → NOTIFIED` hands it to the current poll owner, and
    /// `QUEUED`/`NOTIFIED` mean a poll is already pending while `COMPLETED`
    /// means no poll can ever run again.
    #[inline]
    fn claim_enqueue(&self) -> bool {
        loop {
            match self.state.load(Ordering::Acquire) {
                ASYNC_IDLE => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_IDLE,
                            ASYNC_QUEUED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                }
                ASYNC_POLLING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_POLLING,
                            ASYNC_NOTIFIED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }

    /// Spawn-time admission of the first poll.
    ///
    /// # Errors
    /// Propagates scheduler admission failure (queue saturation or shutdown)
    /// to the spawner. The failed `IDLE → QUEUED` claim is reverted first, so
    /// the returned state is clean: droppable, and retryable by a new spawn
    /// attempt. The revert cannot race a wake — wakers are minted only inside
    /// `poll`, which has not run before the first successful admission.
    /// Woken-path rescheduling instead goes through [`Self::schedule_wake`],
    /// which never drops the wake on a full queue.
    #[inline]
    pub(crate) fn schedule(self: Arc<Self>) -> ExecutorResult<()> {
        if !self.claim_enqueue() {
            return Ok(());
        }
        let admitted = Arc::clone(&self).enqueue();
        if admitted.is_err() {
            // The rejected job never entered a queue, so this caller still
            // owns the QUEUED epoch and no poll can be racing the revert.
            self.revert_queued_to_idle();
        }
        admitted
    }

    /// Wake-path admission: never loses the wake (module docs: enqueue
    /// obligation).
    ///
    /// The claimed `QUEUED` epoch is discharged by exactly one of:
    /// - a successful enqueue (a worker will poll),
    /// - the inline poll below (this thread polls; no queue slot needed), or
    /// - the shutdown completion (no job can ever be admitted or run again, so
    ///   the wake is unfulfillable rather than lost to backpressure, and the
    ///   task ends cancelled).
    ///
    /// On admission rejection the waking thread polls the future itself —
    /// mirroring how `SchedulerScope::flush` runs admission-refused jobs on the
    /// calling lane. This cannot lose the transition because `poll` consumes
    /// the `QUEUED` state directly, and it keeps saturated pools independent of
    /// OS yield latency or a queue that only a gated worker can drain.
    pub(super) fn schedule_wake(self: &Arc<Self>) {
        if !self.claim_enqueue() {
            return;
        }
        match Arc::clone(self).enqueue() {
            Ok(()) => {}
            Err(ExecutorError::ResourceExhausted(_)) => {
                match InlinePollDepthGuard::try_enter() {
                    Some(_depth_guard) => {
                        // Registry diagnostics report the task as running off the
                        // worker pool; `NO_WORKER` is display-only there.
                        self.poll(crate::registry::state::NO_WORKER as usize);
                    }
                    _ => {
                        self.complete_resource_exhausted();
                    }
                }
            }
            Err(_) => {
                // ShuttingDown: the scheduler admits and runs nothing from
                // here on, so no poll of this task can ever be admitted.
                // Completing it now resolves its waiters and handle; leaving
                // it idle would hold them until the last waker clone drops.
                self.complete_cancelled();
            }
        }
    }

    #[inline]
    fn enqueue(self: Arc<Self>) -> ExecutorResult<()> {
        let state = Arc::clone(&self);
        self.scheduler
            .schedule::<AsyncTask, _>(Priority::Normal, None, move |worker_id| {
                state.poll(worker_id);
            })
    }

    /// Discharge a wake absorbed after the inline-repoll budget was consumed.
    ///
    /// The current poll owner transfers `NOTIFIED` directly to `QUEUED`, so no
    /// recursive call can grow the waking thread's stack. Persistent admission
    /// saturation becomes an explicit task failure; the wake is never silently
    /// dropped and the caller can distinguish resource exhaustion from output.
    pub(super) fn reschedule_notified(self: &Arc<Self>) {
        if self
            .state
            .compare_exchange(
                ASYNC_NOTIFIED,
                ASYNC_QUEUED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }

        match Arc::clone(self).enqueue() {
            Ok(()) => {}
            Err(ExecutorError::ResourceExhausted(_)) => {
                self.complete_resource_exhausted();
            }
            Err(_) => {
                self.complete_cancelled();
            }
        }
    }

    #[inline]
    fn revert_queued_to_idle(&self) {
        self.state.store(ASYNC_IDLE, Ordering::Release);
    }
}
