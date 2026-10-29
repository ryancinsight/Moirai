//! The shared future cells and the atomic phase that serializes access to them.

use std::{
    cell::UnsafeCell,
    future::Future,
    mem::MaybeUninit,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
    sync::{Arc, OnceLock, atomic::AtomicU8},
};

use moirai_core::task::TaskResultSender;

use super::{lifecycle::AsyncLifecycle, phase::ASYNC_CREATED};
use crate::{
    metrics::ExecutorMetrics,
    registry::{OwnedStateLease, StateLease, TaskLifecycleToken},
    schedule::WorkSubmit,
};

pub(crate) struct AsyncFutureState<S, F, L = OwnedStateLease>
where
    F: Future,
    L: StateLease,
{
    pub(super) lifecycle: UnsafeCell<AsyncLifecycle<L>>,
    pub(super) future: UnsafeCell<MaybeUninit<F>>,
    pub(super) result_sender: UnsafeCell<Option<TaskResultSender<F::Output>>>,
    pub(super) metrics: Arc<ExecutorMetrics>,
    pub(super) state: AtomicU8,
    pub(super) future_present: UnsafeCell<bool>,
    pub(super) registration: OnceLock<super::tasks::Registration>,
    // Must remain last: production lifecycle leases borrow storage retained by
    // the scheduler, so every lease must retire before scheduler destruction.
    pub(super) scheduler: S,
}

// Safety: `state` serializes all future polling. Wakers may schedule work
// concurrently, but they only mutate atomics and never touch the future cell.
// The future cell is dropped either by the unique polling thread after Ready or
// panic, or by `Drop` after the last `Arc` reference is gone. The scheduler `S`
// is itself `Send + Sync`, so sharing it across wakers is sound.
unsafe impl<S, F, L> Send for AsyncFutureState<S, F, L>
where
    S: Send + Sync,
    F: Future + Send,
    F::Output: Send,
    L: StateLease,
{
}

// Safety: see the `Send` impl. Shared references are used only for atomic
// scheduling, metrics, and fields guarded by the single poll owner selected by
// the async state machine.
unsafe impl<S, F, L> Sync for AsyncFutureState<S, F, L>
where
    S: Send + Sync,
    F: Future + Send,
    F::Output: Send,
    L: StateLease,
{
}

impl<S, F, L> AsyncFutureState<S, F, L>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    L: StateLease,
{
    pub(crate) fn new(
        scheduler: S,
        future: F,
        lifecycle: TaskLifecycleToken<L>,
        result_sender: TaskResultSender<F::Output>,
        metrics: Arc<ExecutorMetrics>,
    ) -> Arc<Self> {
        Arc::new(Self {
            lifecycle: UnsafeCell::new(AsyncLifecycle::Registered(lifecycle)),
            future: UnsafeCell::new(MaybeUninit::new(future)),
            result_sender: UnsafeCell::new(Some(result_sender)),
            metrics,
            state: AtomicU8::new(ASYNC_CREATED),
            future_present: UnsafeCell::new(true),
            registration: OnceLock::new(),
            scheduler,
        })
    }
}

impl<S, F, L> Drop for AsyncFutureState<S, F, L>
where
    F: Future,
    L: StateLease,
{
    fn drop(&mut self) {
        if let Some(registration) = self.registration.get() {
            registration.remove();
        }
        let future_present = self.future_present.get_mut();
        if *future_present {
            *future_present = false;
            // Safety: `Drop` has exclusive access to the state because the last
            // `Arc` reference is being destroyed.
            let future_drop = catch_unwind(AssertUnwindSafe(|| unsafe {
                ptr::drop_in_place((*self.future.get()).as_mut_ptr());
            }));
            if let Err(_payload) = future_drop {
                // Production admissions settle through complete_rejected, and
                // accepted futures remain tracked until explicit completion.
                // This final-owner fallback has no completion owner to report
                // a destruction failure, so it follows the fatal Drop policy.
                // Aborting also avoids invoking an untrusted panic destructor.
                std::process::abort();
            }
        }
    }
}
