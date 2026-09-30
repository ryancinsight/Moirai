//! Completion paths: consuming the lifecycle token, dropping the future once,
//! and publishing the task result.

use std::{future::Future, ptr, sync::atomic::Ordering};

use moirai_core::{error::TaskError, task::TaskResultSender};

use super::{
    future_state::AsyncFutureState,
    lifecycle::AsyncLifecycle,
    phase::{ASYNC_COMPLETED, ASYNC_QUEUED},
};
use crate::{registry::StateLease, schedule::WorkSubmit};

impl<S, F, L> AsyncFutureState<S, F, L>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    L: StateLease,
{
    /// Consume the lifecycle token, registered or running, as cancelled.
    fn cancel_lifecycle(&self) {
        // Safety: only the poll owner or the rejected-queue completion owner
        // calls this method. Their POLLING/QUEUED states exclude every other
        // accessor, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        match std::mem::replace(lifecycle, AsyncLifecycle::Completed) {
            AsyncLifecycle::Registered(token) => token.cancel(),
            AsyncLifecycle::Running(token) => token.cancel(),
            AsyncLifecycle::Completed => {}
        }
    }

    fn complete_lifecycle(&self) -> core::time::Duration {
        // Safety: only the poll owner or the rejected-queue completion owner
        // calls this method. Their POLLING/QUEUED states exclude every other
        // accessor, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        let running = std::mem::replace(lifecycle, AsyncLifecycle::Completed);
        if let AsyncLifecycle::Running(token) = running {
            token.complete()
        } else {
            core::time::Duration::ZERO
        }
    }

    fn drop_future(&self) {
        // Safety: only the poll owner or the rejected-queue completion owner
        // calls this method while shared references exist. Their
        // POLLING/QUEUED states exclude every other accessor. `Drop` reaches the
        // same flag only after the final `Arc` is gone and has exclusive access.
        // The poll hot path does not read this flag; `state` is the authoritative
        // polling permission and guarantees initialized future storage.
        let future_present = unsafe { &mut *self.future_present.get() };
        if *future_present {
            *future_present = false;
            // Safety: the caller owns poll or rejected-queue completion
            // permission, or `Drop` owns the last state reference. The
            // initialized future is dropped once.
            unsafe {
                ptr::drop_in_place((*self.future.get()).as_mut_ptr());
            }
        }
    }

    fn take_result_sender(&self) -> Option<TaskResultSender<F::Output>> {
        // Safety: result publication is reached only by the poll owner or the
        // rejected-queue completion owner. Their POLLING/QUEUED states exclude
        // every other accessor. `Drop` does not read this cell.
        unsafe { (&mut *self.result_sender.get()).take() }
    }

    #[inline]
    fn store_completed(&self) {
        self.drop_future();
        self.state.store(ASYNC_COMPLETED, Ordering::Release);
    }

    #[inline]
    fn publish_result(&self, result: Result<F::Output, TaskError>) {
        if let Some(sender) = self.take_result_sender() {
            sender.send(result);
        }
    }

    #[inline]
    pub(super) fn complete_with_result(
        &self,
        result: Result<F::Output, TaskError>,
    ) -> core::time::Duration {
        self.store_completed();
        let execution_time = self.complete_lifecycle();
        self.publish_result(result);
        execution_time
    }

    #[inline]
    pub(super) fn complete_failed(&self, error: TaskError) {
        self.complete_with_result(Err(error));
        self.metrics.record_task_failed();
    }

    /// Complete the task as cancelled: its future is dropped unfinished, the
    /// lifecycle records `cancelled`, and the handle resolves to
    /// `TaskError::Cancelled`.
    ///
    /// The caller is the poll owner, which observed a cancel request before the
    /// first poll, or the rejected-queue completion owner of a `QUEUED` epoch
    /// that scheduler shutdown refused.
    pub(super) fn complete_cancelled(&self) {
        self.store_completed();
        self.cancel_lifecycle();
        // Record before publishing the result so a joiner observes the
        // cancelled counter as soon as the handle resolves.
        self.metrics.record_task_cancelled();
        self.publish_result(Err(TaskError::Cancelled));
    }

    /// Complete a rejected `QUEUED` epoch without polling its future.
    ///
    /// The caller exclusively owns this epoch: it either won `IDLE → QUEUED`
    /// or transferred `NOTIFIED → QUEUED`, and `enqueue` returned the job rather
    /// than admitting it. Concurrent wakers cannot leave `QUEUED`, and no poll
    /// job exists, so this owner may drop and publish completion exactly once.
    pub(super) fn complete_resource_exhausted(&self) {
        debug_assert_eq!(self.state.load(Ordering::Acquire), ASYNC_QUEUED);
        self.complete_failed(TaskError::ResourceExhausted);
    }
}
