//! Completion paths: consuming the lifecycle token, dropping the future once,
//! and publishing the task result.

use std::{
    any::Any,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    ptr,
    sync::atomic::Ordering,
};

use moirai_core::{error::TaskError, task::TaskResultSender};

use super::{
    future_state::AsyncFutureState,
    lifecycle::AsyncLifecycle,
    phase::{ASYNC_COMPLETED, ASYNC_QUEUED},
};
use crate::{
    registry::{StateLease, TaskCompletion},
    schedule::WorkSubmit,
};

// `catch_unwind` erases the compile-time-unknown panic payload at this cold
// scheduler boundary; task polling and completion remain monomorphized.
type PanicPayload = Box<dyn Any + Send>;

impl<S, F, L> AsyncFutureState<S, F, L>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    L: StateLease,
{
    /// Consume the lifecycle token, registered or running, as cancelled.
    fn cancel_lifecycle(&self) -> Option<TaskCompletion> {
        // Safety: only the poll owner or the rejected-queue completion owner
        // calls this method. Their POLLING/QUEUED states exclude every other
        // accessor, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        match std::mem::replace(lifecycle, AsyncLifecycle::Completed) {
            AsyncLifecycle::Registered(token) => Some(token.cancel()),
            AsyncLifecycle::Running(token) => Some(token.cancel()),
            AsyncLifecycle::Completed => None,
        }
    }

    fn complete_lifecycle(&self) -> Option<TaskCompletion> {
        // Safety: only the poll owner or the rejected-queue completion owner
        // calls this method. Their POLLING/QUEUED states exclude every other
        // accessor, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        match std::mem::replace(lifecycle, AsyncLifecycle::Completed) {
            AsyncLifecycle::Registered(token) => Some(token.complete()),
            AsyncLifecycle::Running(token) => Some(token.complete()),
            AsyncLifecycle::Completed => None,
        }
    }

    fn drop_future(&self) -> Option<PanicPayload> {
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
            catch_unwind(AssertUnwindSafe(|| unsafe {
                ptr::drop_in_place((*self.future.get()).as_mut_ptr());
            }))
            .err()
        } else {
            None
        }
    }

    fn take_result_sender(&self) -> Option<TaskResultSender<F::Output>> {
        // Safety: result publication is reached only by the poll owner or the
        // rejected-queue completion owner. Their POLLING/QUEUED states exclude
        // every other accessor. `Drop` does not read this cell.
        unsafe { (&mut *self.result_sender.get()).take() }
    }

    #[inline]
    fn store_completed(&self) -> Option<PanicPayload> {
        let future_drop_panic = self.drop_future();
        self.state.store(ASYNC_COMPLETED, Ordering::Release);
        if let Some(registration) = self.registration.get() {
            registration.remove();
        }
        future_drop_panic
    }

    #[inline]
    fn publish_result(&self, result: Result<F::Output, TaskError>) {
        if let Some(sender) = self.take_result_sender() {
            sender.send(result);
        }
    }

    /// Publish the handle before invoking registry callbacks that may join it.
    fn finish_result(
        &self,
        result: Result<F::Output, TaskError>,
        completion: Option<TaskCompletion>,
    ) -> Option<PanicPayload> {
        let publication = catch_unwind(AssertUnwindSafe(|| self.publish_result(result)));
        let notification = catch_unwind(AssertUnwindSafe(|| {
            if let Some(completion) = completion {
                completion.notify();
            }
        }));
        // These counters report notification failures without changing the
        // already-published task result. They carry no synchronization edge.
        for failed in [publication.is_err(), notification.is_err()] {
            if failed {
                self.metrics.waiter_panics.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Return callback panics so the completion owner can destroy displaced
        // output and original panic payloads before resuming unwinding.
        match (publication, notification) {
            (Ok(()), Ok(())) => None,
            (Err(payload), Ok(())) | (Ok(()), Err(payload)) => Some(payload),
            (Err(first), Err(second)) => {
                drop_after_settlement(second);
                Some(first)
            }
        }
    }

    #[inline]
    pub(super) fn complete_with_result(&self, result: Result<F::Output, TaskError>) {
        let future_drop_panic = self.store_completed();
        let completion = self.complete_lifecycle();
        let execution_time = completion
            .as_ref()
            .map_or(core::time::Duration::ZERO, TaskCompletion::duration);
        if let Some(panic_payload) = future_drop_panic {
            self.metrics.record_task_failed();
            let callback_panic = self.finish_result(Err(TaskError::Panicked), completion);
            drop_after_settlement(result);
            drop_after_settlement(panic_payload);
            resume_callback(callback_panic);
        } else {
            match result {
                Ok(output) => {
                    self.metrics.record_task_completed(execution_time);
                    resume_callback(self.finish_result(Ok(output), completion));
                }
                Err(error) => {
                    self.metrics.record_task_failed();
                    resume_callback(self.finish_result(Err(error), completion));
                }
            }
        }
    }

    #[inline]
    pub(super) fn complete_failed(&self, error: TaskError) {
        self.complete_with_result(Err(error));
    }

    /// Complete a poll panic without destroying either panic payload first.
    pub(super) fn complete_panicked(&self, poll_panic: PanicPayload) {
        let future_drop_panic = self.store_completed();
        let completion = self.complete_lifecycle();
        self.metrics.record_task_failed();
        let callback_panic = self.finish_result(Err(TaskError::Panicked), completion);
        drop_after_settlement(poll_panic);
        if let Some(future_drop_panic) = future_drop_panic {
            drop_after_settlement(future_drop_panic);
        }
        resume_callback(callback_panic);
    }

    /// Complete the task as cancelled: its future is dropped unfinished, the
    /// lifecycle records `cancelled`, and the handle resolves to
    /// `TaskError::Cancelled`, unless its destructor panics.
    ///
    /// The caller is the poll owner, which observed a cancel request before the
    /// first poll, or the rejected-queue completion owner of a `QUEUED` epoch
    /// that scheduler shutdown refused.
    pub(super) fn complete_cancelled(&self) {
        if let Some(panic_payload) = self.store_completed() {
            let completion = self.complete_lifecycle();
            self.metrics.record_task_failed();
            let callback_panic = self.finish_result(Err(TaskError::Panicked), completion);
            drop_after_settlement(panic_payload);
            resume_callback(callback_panic);
        } else {
            let completion = self.cancel_lifecycle();
            // Record before publishing the result so a joiner observes the
            // cancelled counter as soon as the handle resolves.
            self.metrics.record_task_cancelled();
            resume_callback(self.finish_result(Err(TaskError::Cancelled), completion));
        }
    }

    /// Settle a future whose first scheduler admission was rejected.
    ///
    /// The task preserves the rejection's cancellation settlement when
    /// destruction succeeds. A destructor panic instead becomes the returned
    /// task error after the lifecycle and private result handle are settled.
    pub(crate) fn complete_rejected(&self) -> Result<(), TaskError> {
        if let Some(panic_payload) = self.store_completed() {
            let completion = self.complete_lifecycle();
            let callback_panic = self.finish_result(Err(TaskError::Panicked), completion);
            drop_after_settlement(panic_payload);
            resume_callback(callback_panic);
            Err(TaskError::Panicked)
        } else {
            let completion = self.cancel_lifecycle();
            resume_callback(self.finish_result(Err(TaskError::Cancelled), completion));
            Ok(())
        }
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

/// Destroy a value only after every observable completion channel is settled.
///
/// A panic here is secondary to the already-published `TaskError::Panicked`.
/// The scheduler boundary contains it; a second panic while destroying that
/// panic payload aborts, matching the existing scheduled-job policy.
fn drop_after_settlement<T>(value: T) {
    // The primary failure is already published. Reuse the scheduler boundary
    // to contain a cleanup panic and abort if its panic payload also panics.
    crate::schedule::job::run_caught(|| drop(value));
}

/// Resume only after result channels and retained completion values are settled.
fn resume_callback(panic: Option<PanicPayload>) {
    if let Some(payload) = panic {
        resume_unwind(payload);
    }
}
