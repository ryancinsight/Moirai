//! The poll owner's loop: claiming `POLLING`, driving the future, and settling
//! a pending poll.

use std::{
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{Arc, atomic::Ordering},
    task::{Context, Poll, Waker},
};

use moirai_core::error::TaskError;

use super::{
    future_state::AsyncFutureState,
    lifecycle::AsyncLifecycle,
    phase::{ASYNC_IDLE, ASYNC_NOTIFIED, ASYNC_POLLING, ASYNC_QUEUED},
};
use crate::{registry::StateLease, schedule::WorkSubmit};

const ASYNC_INLINE_REPOLL_LIMIT: usize = 1;

enum PendingPoll {
    Return,
    Repoll,
    Reschedule,
}

impl<S, F, L> AsyncFutureState<S, F, L>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    L: StateLease,
{
    pub(super) fn poll(self: &Arc<Self>, worker_id: usize) {
        if self
            .state
            .compare_exchange(
                ASYNC_QUEUED,
                ASYNC_POLLING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }

        if self.cancel_pending() {
            // Cooperative cancellation observed before the first poll: the
            // future body never runs. Mirrors the sync-path cancel handling in
            // `TaskLifecycleToken::start_unless_cancelled`.
            self.complete_cancelled();
            return;
        }

        self.mark_running(worker_id);
        let waker = Waker::from(Arc::clone(self));
        let mut context = Context::from_waker(&waker);
        let mut inline_repolls = 0usize;

        loop {
            let poll_result = {
                // Safety: `state` grants this worker the only polling
                // permission, and the `Arc` allocation keeps the address
                // stable while the future is pinned. Future storage remains
                // initialized until the poll owner reaches ready or panic.
                let future = unsafe { Pin::new_unchecked(&mut *(*self.future.get()).as_mut_ptr()) };
                catch_unwind(AssertUnwindSafe(|| future.poll(&mut context)))
            };

            match poll_result {
                Ok(Poll::Ready(output)) => {
                    let execution_time = self.complete_with_result(Ok(output));
                    self.metrics.record_task_completed(execution_time);
                    return;
                }
                Ok(Poll::Pending) => match self.finish_pending_poll(&mut inline_repolls) {
                    PendingPoll::Return => return,
                    PendingPoll::Repoll => continue,
                    PendingPoll::Reschedule => {
                        self.reschedule_notified();
                        return;
                    }
                },
                Err(_) => {
                    self.complete_failed(TaskError::Panicked);
                    return;
                }
            }
        }
    }

    /// Whether the task was cancelled while still queued (never polled).
    fn cancel_pending(&self) -> bool {
        // Safety: only the poll owner selected by the async state machine calls
        // this method, so the lifecycle cell access is single-threaded.
        let lifecycle = unsafe { &*self.lifecycle.get() };
        matches!(lifecycle, AsyncLifecycle::Registered(token) if token.cancel_requested())
    }

    fn mark_running(&self, worker_id: usize) {
        // Safety: only the poll owner selected by the async state machine calls
        // this method, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        if matches!(*lifecycle, AsyncLifecycle::Registered(_)) {
            let registered = std::mem::replace(lifecycle, AsyncLifecycle::Completed);
            if let AsyncLifecycle::Registered(token) = registered {
                *lifecycle = AsyncLifecycle::Running(token.start(worker_id));
            }
        }
    }

    #[inline]
    fn finish_pending_poll(&self, inline_repolls: &mut usize) -> PendingPoll {
        match self.state.compare_exchange(
            ASYNC_POLLING,
            ASYNC_IDLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => PendingPoll::Return,
            Err(ASYNC_NOTIFIED) if *inline_repolls < ASYNC_INLINE_REPOLL_LIMIT => {
                if self
                    .state
                    .compare_exchange(
                        ASYNC_NOTIFIED,
                        ASYNC_POLLING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    *inline_repolls += 1;
                    PendingPoll::Repoll
                } else {
                    PendingPoll::Return
                }
            }
            Err(ASYNC_NOTIFIED) => PendingPoll::Reschedule,
            Err(_) => PendingPoll::Return,
        }
    }
}
