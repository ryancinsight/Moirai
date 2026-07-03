use std::{
    cell::UnsafeCell,
    future::Future,
    mem::MaybeUninit,
    panic::{catch_unwind, AssertUnwindSafe},
    pin::Pin,
    ptr,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    task::{Context, Poll, Wake, Waker},
};

use moirai_core::{
    error::{ExecutorResult, TaskError},
    task::TaskResultSender,
    Priority,
};

use crate::{
    metrics::ExecutorMetrics,
    registry::{RunningTaskToken, TaskLifecycleToken},
    schedule::{AsyncTask, WorkSubmit},
};

const ASYNC_IDLE: u8 = 0;
const ASYNC_QUEUED: u8 = 1;
const ASYNC_POLLING: u8 = 2;
const ASYNC_NOTIFIED: u8 = 3;
const ASYNC_COMPLETED: u8 = 4;
const ASYNC_INLINE_REPOLL_LIMIT: usize = 1;

pub(super) enum AsyncLifecycle {
    Registered(TaskLifecycleToken),
    Running(RunningTaskToken),
    Completed,
}

pub(crate) struct AsyncFutureState<S, F>
where
    F: Future,
{
    scheduler: S,
    future: UnsafeCell<MaybeUninit<F>>,
    lifecycle: UnsafeCell<AsyncLifecycle>,
    result_sender: UnsafeCell<Option<TaskResultSender<F::Output>>>,
    metrics: Arc<ExecutorMetrics>,
    state: AtomicU8,
    future_present: UnsafeCell<bool>,
}

// Safety: `state` serializes all future polling. Wakers may schedule work
// concurrently, but they only mutate atomics and never touch the future cell.
// The future cell is dropped either by the unique polling thread after Ready or
// panic, or by `Drop` after the last `Arc` reference is gone. The scheduler `S`
// is itself `Send + Sync`, so sharing it across wakers is sound.
unsafe impl<S, F> Send for AsyncFutureState<S, F>
where
    S: Send + Sync,
    F: Future + Send,
    F::Output: Send,
{
}

// Safety: see the `Send` impl. Shared references are used only for atomic
// scheduling, metrics, and fields guarded by the single poll owner selected by
// the async state machine.
unsafe impl<S, F> Sync for AsyncFutureState<S, F>
where
    S: Send + Sync,
    F: Future + Send,
    F::Output: Send,
{
}

impl<S, F> AsyncFutureState<S, F>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    pub(crate) fn new(
        scheduler: S,
        future: F,
        lifecycle: TaskLifecycleToken,
        result_sender: TaskResultSender<F::Output>,
        metrics: Arc<ExecutorMetrics>,
    ) -> Arc<Self> {
        Arc::new(Self {
            scheduler,
            future: UnsafeCell::new(MaybeUninit::new(future)),
            lifecycle: UnsafeCell::new(AsyncLifecycle::Registered(lifecycle)),
            result_sender: UnsafeCell::new(Some(result_sender)),
            metrics,
            state: AtomicU8::new(ASYNC_IDLE),
            future_present: UnsafeCell::new(true),
        })
    }

    #[inline]
    pub(crate) fn schedule(self: Arc<Self>) -> ExecutorResult<()> {
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
                        return self.enqueue();
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
                        return Ok(());
                    }
                }
                ASYNC_QUEUED | ASYNC_NOTIFIED | ASYNC_COMPLETED => return Ok(()),
                _ => return Ok(()),
            }
        }
    }

    #[inline]
    fn schedule_by_ref(self: &Arc<Self>) -> ExecutorResult<()> {
        loop {
            match self.state.load(Ordering::Acquire) {
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
                        return Ok(());
                    }
                }
                ASYNC_IDLE => return Arc::clone(self).schedule(),
                ASYNC_QUEUED | ASYNC_NOTIFIED | ASYNC_COMPLETED => return Ok(()),
                _ => return Ok(()),
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

    fn poll(self: &Arc<Self>, worker_id: usize) {
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
            self.drop_future();
            self.state.store(ASYNC_COMPLETED, Ordering::Release);
            self.cancel_lifecycle();
            // Record before publishing the result so a joiner observes the
            // cancelled counter as soon as the handle resolves.
            self.metrics.record_task_cancelled();
            if let Some(sender) = self.take_result_sender() {
                sender.send(Err(TaskError::Cancelled));
            }
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
                    self.drop_future();
                    self.state.store(ASYNC_COMPLETED, Ordering::Release);
                    let execution_time = self.complete_lifecycle();
                    if let Some(sender) = self.take_result_sender() {
                        sender.send(Ok(output));
                    }
                    self.metrics.record_task_completed(execution_time);
                    return;
                }
                Ok(Poll::Pending) => {
                    if self.finish_pending_poll(&mut inline_repolls) {
                        continue;
                    }
                    return;
                }
                Err(_) => {
                    self.drop_future();
                    self.state.store(ASYNC_COMPLETED, Ordering::Release);
                    self.complete_lifecycle();
                    if let Some(sender) = self.take_result_sender() {
                        sender.send(Err(TaskError::Panicked));
                    }
                    self.metrics.record_task_failed();
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

    /// Consume the registered lifecycle token as cancelled.
    fn cancel_lifecycle(&self) {
        // Safety: only the poll owner selected by the async state machine calls
        // this method, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        if let AsyncLifecycle::Registered(token) =
            std::mem::replace(lifecycle, AsyncLifecycle::Completed)
        {
            token.cancel();
        }
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

    fn complete_lifecycle(&self) -> core::time::Duration {
        // Safety: only the poll owner selected by the async state machine calls
        // this method, so lifecycle mutation is single-threaded.
        let lifecycle = unsafe { &mut *self.lifecycle.get() };
        let running = std::mem::replace(lifecycle, AsyncLifecycle::Completed);
        if let AsyncLifecycle::Running(token) = running {
            token.complete()
        } else {
            core::time::Duration::ZERO
        }
    }

    fn drop_future(&self) {
        // Safety: only the poll owner selected by the async state machine calls
        // this method while shared references exist. `Drop` reaches the same
        // flag only after the final `Arc` is gone and has exclusive access.
        // The poll hot path does not read this flag; `state` is the authoritative
        // polling permission and guarantees initialized future storage.
        let future_present = unsafe { &mut *self.future_present.get() };
        if *future_present {
            *future_present = false;
            // Safety: the caller owns poll/completion permission or `Drop` owns
            // the last state reference. The initialized future is dropped once.
            unsafe {
                ptr::drop_in_place((*self.future.get()).as_mut_ptr());
            }
        }
    }

    fn take_result_sender(&self) -> Option<TaskResultSender<F::Output>> {
        // Safety: result publication is reached only by the single poll owner
        // selected by the async state machine. `Drop` has exclusive access after
        // the last `Arc` is gone and does not read this cell.
        unsafe { (&mut *self.result_sender.get()).take() }
    }

    #[inline]
    fn finish_pending_poll(self: &Arc<Self>, inline_repolls: &mut usize) -> bool {
        match self.state.compare_exchange(
            ASYNC_POLLING,
            ASYNC_IDLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => false,
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
                    true
                } else {
                    false
                }
            }
            Err(ASYNC_NOTIFIED) => {
                self.state.store(ASYNC_IDLE, Ordering::Release);
                let _ = Arc::clone(self).schedule();
                false
            }
            Err(_) => false,
        }
    }
}

impl<S, F> Drop for AsyncFutureState<S, F>
where
    F: Future,
{
    fn drop(&mut self) {
        if *self.future_present.get_mut() {
            // Safety: `Drop` has exclusive access to the state because the last
            // `Arc` reference is being destroyed.
            unsafe {
                ptr::drop_in_place((*self.future.get()).as_mut_ptr());
            }
        }
    }
}

impl<S, F> Wake for AsyncFutureState<S, F>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    fn wake(self: Arc<Self>) {
        let _ = self.schedule();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.schedule_by_ref();
    }
}
