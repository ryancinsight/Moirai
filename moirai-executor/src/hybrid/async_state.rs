//! Async future state machine.
//!
//! `AsyncFutureState` drives one `Future` to completion across the hybrid
//! scheduler's worker threads. It is shared as an `Arc` (it is its own `Waker`),
//! so a `Future` — which is `!Sync` to poll — is held in `UnsafeCell`s reachable
//! from every clone. A single `AtomicU8` `state` makes the concurrent access to
//! those cells sound.
//!
//! # State machine
//!
//! ```text
//! IDLE ──schedule──▶ QUEUED ──poll claims──▶ POLLING ──Pending, no wake──▶ IDLE
//!                                              │  │
//!                          wake during poll ───┘  └── Ready / panic / cancel ─▶ COMPLETED
//!                                (NOTIFIED, re-polled inline or rescheduled)
//! ```
//!
//! # Exclusivity invariant
//!
//! The `QUEUED → POLLING` compare-exchange in `AsyncFutureState::poll` has
//! exactly one winner; the loser returns without touching anything. That winner
//! is the **poll owner**, and it is the *only* accessor of the `future`,
//! `lifecycle`, `result_sender`, and `future_present` cells until it leaves
//! `POLLING`. Wakers running on other threads only load/CAS `state` and never
//! read those cells, so every `UnsafeCell` dereference here is single-threaded
//! despite the shared `Arc` — this is what the per-site `Safety` comments mean by
//! "the single poll owner selected by the async state machine". The `&mut` to the
//! pinned future is dropped before any `state` transition, so no successor poll
//! can observe it.
//!
//! A wake arriving mid-poll CASes `POLLING → NOTIFIED` rather than enqueuing, so
//! it is never lost: the poll owner re-polls inline (bounded by
//! `ASYNC_INLINE_REPOLL_LIMIT`) or reschedules. The future is dropped once, by
//! the `future_present` flag: the poll owner drops it on completion and `Drop`
//! (reached only after the last `Arc`, whose refcount release/acquire orders the
//! owner's write before the destructor's read) skips an already-dropped future.
//!
//! # Enqueue obligation (wakes survive admission rejection)
//!
//! The `IDLE → QUEUED` winner owns exactly one *enqueue obligation*: `poll` is
//! the only exit from `QUEUED`, and it runs only from the job that winner
//! admits, so a discarded admission failure strands the task — `QUEUED` with no
//! job in any queue means every later wake short-circuits as "already
//! scheduled" and the future is never polled again. `schedule_wake` therefore
//! never drops the obligation on a full injector: it retries admission through
//! a bounded spin/yield ladder (each retry re-selects a worker round-robin) and
//! past the ladder polls inline on the waking thread, which needs no queue
//! slot. Only scheduler shutdown — after which no job of any kind can ever be
//! admitted or run — releases the obligation, by reverting `QUEUED → IDLE`.
//! The spawn-time `schedule` instead propagates admission failure to the
//! spawner (the spawn-backpressure contract) after the same revert, which is
//! race-free there because wakers are minted only inside `poll`.

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
    error::{ExecutorError, ExecutorResult, TaskError},
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

/// Admission retries separated by `spin_loop` before the ladder escalates.
///
/// A rejected wake needs one worker pop to free an injector slot — a
/// sub-microsecond window a short spin covers. The split between the spin and
/// yield rungs is a latency/CPU trade only: never-lose correctness is carried
/// by the inline-poll rung past the ladder, not by these budgets.
const WAKE_ENQUEUE_SPIN_RETRIES: usize = 32;
/// Admission retries separated by `yield_now` before polling inline.
///
/// A yield hands the OS a slice so a descheduled worker can drain; saturation
/// that survives the full ladder is persistent, and waiting longer would be
/// the unbounded wait the bounded-resource rule prohibits.
const WAKE_ENQUEUE_YIELD_RETRIES: usize = 32;

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
            self.state.store(ASYNC_IDLE, Ordering::Release);
        }
        admitted
    }

    /// Wake-path admission: never loses the wake (module docs: enqueue
    /// obligation).
    ///
    /// The claimed `QUEUED` epoch is discharged by exactly one of:
    /// - a successful enqueue (a worker will poll),
    /// - the inline poll below (this thread polls; no queue slot needed), or
    /// - the shutdown revert (no job can ever be admitted or run again, so the
    ///   wake is unfulfillable rather than lost to backpressure).
    ///
    /// Admission rejection retries through a spin-then-yield ladder; each
    /// retry re-selects a worker round-robin inside the scheduler, so any
    /// worker's pop unblocks it. Past the ladder the waking thread polls the
    /// future itself — mirroring how `SchedulerScope::flush` runs
    /// admission-refused jobs on the calling lane — which cannot lose the
    /// transition because `poll` consumes the `QUEUED` state directly. Inline
    /// polling also keeps a saturated single-worker pool deadlock-free where
    /// an unbounded enqueue retry from inside that worker's own poll would
    /// spin forever against a queue only it can drain.
    fn schedule_wake(self: &Arc<Self>) {
        if !self.claim_enqueue() {
            return;
        }
        let mut attempts = 0usize;
        loop {
            match Arc::clone(self).enqueue() {
                Ok(()) => return,
                Err(ExecutorError::ResourceExhausted(_)) => {
                    attempts += 1;
                    if attempts <= WAKE_ENQUEUE_SPIN_RETRIES {
                        core::hint::spin_loop();
                    } else if attempts <= WAKE_ENQUEUE_SPIN_RETRIES + WAKE_ENQUEUE_YIELD_RETRIES {
                        std::thread::yield_now();
                    } else {
                        // Registry diagnostics report the task as running off
                        // the worker pool; `NO_WORKER` is display-only there.
                        self.poll(crate::registry::state::NO_WORKER);
                        return;
                    }
                }
                Err(_) => {
                    // ShuttingDown: the scheduler admits and runs nothing from
                    // here on, so no poll of this task can ever be admitted —
                    // reverting keeps the state honest for `Drop`.
                    self.state.store(ASYNC_IDLE, Ordering::Release);
                    return;
                }
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
                // The inline-repoll budget is spent, so the wake absorbed as
                // NOTIFIED converts back into an enqueue obligation; the
                // never-lose path keeps it alive under admission rejection.
                self.state.store(ASYNC_IDLE, Ordering::Release);
                self.schedule_wake();
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
        self.schedule_wake();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.schedule_wake();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test scope")]

    use std::{
        future::Future,
        pin::Pin,
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc, Mutex,
        },
        task::{Context, Poll, Waker},
    };

    use moirai_core::{
        error::{ExecutorError, ExecutorResult},
        task::{TaskHandle, TaskId},
        Priority,
    };

    use super::AsyncFutureState;
    use crate::metrics::ExecutorMetrics;
    use crate::registry::TaskRegistry;
    use crate::schedule::{SyncTask, ThreadScheduler, WorkClass, WorkSubmit};

    /// Returns `Pending` once, publishing its waker, then `Ready(output)`.
    ///
    /// The two-poll shape is the minimal future whose completion *requires* a
    /// wake to be honored: losing the wake leaves it parked forever.
    struct WakeThenReady {
        output: i32,
        polls: Arc<AtomicUsize>,
        waker: Arc<Mutex<Option<Waker>>>,
    }

    impl Future for WakeThenReady {
        type Output = i32;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<i32> {
            if self.polls.fetch_add(1, Ordering::SeqCst) == 0 {
                *self.waker.lock().unwrap() = Some(cx.waker().clone());
                Poll::Pending
            } else {
                Poll::Ready(self.output)
            }
        }
    }

    /// A registered async task parked one wake away from completion.
    struct PendingTask<S> {
        state: Arc<AsyncFutureState<S, WakeThenReady>>,
        handle: TaskHandle<i32>,
        waker: Arc<Mutex<Option<Waker>>>,
        polls: Arc<AtomicUsize>,
    }

    fn pending_async_state<S: WorkSubmit>(scheduler: S, output: i32) -> PendingTask<S> {
        let mut registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();
        let (handle, result_sender) = TaskHandle::new_pending(TaskId(task_id));
        let polls = Arc::new(AtomicUsize::new(0));
        let waker = Arc::new(Mutex::new(None));
        let state = AsyncFutureState::new(
            scheduler,
            WakeThenReady {
                output,
                polls: Arc::clone(&polls),
                waker: Arc::clone(&waker),
            },
            lifecycle,
            result_sender,
            Arc::new(ExecutorMetrics::new()),
        );
        PendingTask {
            state,
            handle,
            waker,
            polls,
        }
    }

    /// A queued type-erased job awaiting `GatedInjector::drain`.
    type QueuedJob = Box<dyn FnOnce(usize) + Send>;

    /// Seam-substitute injector whose admission refuses a preset number of
    /// attempts before accepting, so each ladder rung is exercised
    /// deterministically on one thread. Stored jobs run for real via `drain`.
    struct GatedInjector {
        jobs: Mutex<Vec<QueuedJob>>,
        refuse_next: AtomicUsize,
        rejections: AtomicUsize,
    }

    impl GatedInjector {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                jobs: Mutex::new(Vec::new()),
                refuse_next: AtomicUsize::new(0),
                rejections: AtomicUsize::new(0),
            })
        }

        fn drain(&self) {
            let jobs = std::mem::take(&mut *self.jobs.lock().unwrap());
            for job in jobs {
                job(0);
            }
        }
    }

    impl WorkSubmit for Arc<GatedInjector> {
        fn schedule<C, F>(
            &self,
            _priority: Priority,
            _locality_hint: Option<usize>,
            task: F,
        ) -> ExecutorResult<()>
        where
            C: WorkClass,
            F: FnOnce(usize) + Send + 'static,
        {
            let refusals = self.refuse_next.load(Ordering::SeqCst);
            if refusals > 0 {
                self.refuse_next.store(refusals - 1, Ordering::SeqCst);
                self.rejections.fetch_add(1, Ordering::SeqCst);
                return Err(ExecutorError::ResourceExhausted(
                    "test injector admission queue is full".into(),
                ));
            }
            self.jobs.lock().unwrap().push(Box::new(task));
            Ok(())
        }
    }

    /// A wake rejected `refusals` times must be admitted on the next attempt
    /// with no wake lost: the future still completes with its exact output.
    fn wake_survives_admission_refusals(refusals: usize, output: i32) {
        let injector = GatedInjector::new();
        let PendingTask {
            state,
            handle,
            waker,
            polls,
        } = pending_async_state(Arc::clone(&injector), output);

        Arc::clone(&state).schedule().expect("first poll admits");
        injector.drain();
        let waker = waker
            .lock()
            .unwrap()
            .take()
            .expect("first poll published its waker");
        assert_eq!(polls.load(Ordering::SeqCst), 1);

        injector.refuse_next.store(refusals, Ordering::SeqCst);
        waker.wake();
        assert_eq!(
            injector.rejections.load(Ordering::SeqCst),
            refusals,
            "the ladder must retry through every refused admission"
        );
        assert_eq!(
            injector.refuse_next.load(Ordering::SeqCst),
            0,
            "admission must have been retried until the injector accepted"
        );

        injector.drain();
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        assert_eq!(handle.join(), Some(Ok(output)));
    }

    #[test]
    fn wake_retries_through_spin_rung_until_admitted() {
        // Below the spin budget: every retry stays on the spin rung.
        wake_survives_admission_refusals(super::WAKE_ENQUEUE_SPIN_RETRIES - 2, 41);
    }

    #[test]
    fn wake_retries_through_yield_rung_until_admitted() {
        // Past the spin budget, inside the yield budget: the yield rung must
        // keep retrying rather than dropping the wake.
        wake_survives_admission_refusals(super::WAKE_ENQUEUE_SPIN_RETRIES + 8, 43);
    }

    /// M1 regression at the real scheduler: a worker's injector is provably
    /// full (fill until `ResourceExhausted`) and the only worker is gated, so
    /// a wake can never be admitted — the waking thread must poll inline and
    /// the woken task must still complete. Before the fix, `Waker::wake`
    /// discarded the rejection and the task stayed `QUEUED` forever.
    #[test]
    fn woken_task_completes_while_worker_injector_is_full() {
        let scheduler =
            ThreadScheduler::<8>::new_with_config(1, "wake-full-injector").expect("scheduler");
        let PendingTask {
            state,
            handle,
            waker,
            polls,
        } = pending_async_state(scheduler.clone(), 1789);

        // First poll runs on the worker and publishes the waker.
        Arc::clone(&state).schedule().expect("first poll admits");

        // Gate the only worker inside a job so nothing can drain the injector.
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_worker| {
                entered_tx.send(()).expect("test observer alive");
                release_rx.recv().expect("release signal");
            })
            .expect("gate job admits");
        entered_rx.recv().expect("worker entered the gate");
        // Jobs run in admission order on the single worker, so the gate having
        // started proves the first poll finished: the waker is published and
        // the task is back to IDLE.
        let waker = waker
            .lock()
            .unwrap()
            .take()
            .expect("first poll published its waker");
        assert_eq!(polls.load(Ordering::SeqCst), 1);

        // Fill the gated worker's injector until admission genuinely rejects.
        let filler_runs = Arc::new(AtomicUsize::new(0));
        let mut saw_rejection = false;
        for _ in 0..4096 {
            let filler_runs = Arc::clone(&filler_runs);
            let admitted = scheduler.schedule::<SyncTask, _>(Priority::Normal, None, move |_w| {
                filler_runs.fetch_add(1, Ordering::SeqCst);
            });
            if let Err(rejection) = admitted {
                assert!(matches!(rejection, ExecutorError::ResourceExhausted(_)));
                saw_rejection = true;
                break;
            }
        }
        assert!(saw_rejection, "an 8-slot injector must fill within 4096 pushes");

        // The injector is full and its only drain is gated: every enqueue
        // retry rejects, so the wake must complete the future inline. The
        // handle resolving *before* the gate opens proves no worker polled.
        waker.wake();
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        assert_eq!(handle.join(), Some(Ok(1789)));

        release_tx.send(()).expect("worker still parked in gate");
        scheduler.shutdown();
    }
}
