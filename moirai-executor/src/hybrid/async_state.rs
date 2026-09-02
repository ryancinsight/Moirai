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
//!                       │                      │  │
//!     rejected wake ────┴──▶ COMPLETED        │  └── Ready / panic / cancel ─▶ COMPLETED
//!     shutdown/spawn rejection ─▶ IDLE        └── wake during poll ─▶ NOTIFIED
//!                                                    (inline repoll or reschedule)
//! ```
//!
//! # Exclusivity invariant
//!
//! The `QUEUED → POLLING` compare-exchange in `AsyncFutureState::poll` has
//! exactly one winner; the loser returns without touching anything. That winner
//! is the **poll owner**. A second exclusive role exists only when that queue
//! admission is rejected: the caller that won `IDLE → QUEUED`, or transferred
//! `NOTIFIED → QUEUED`, remains the **rejected-queue completion owner** because
//! no scheduler job was admitted. While either role accesses `future`,
//! `lifecycle`, `result_sender`, or `future_present`, concurrent wakers only
//! load/CAS `state`; `QUEUED` and `POLLING` both prevent them from becoming an
//! accessor. Every `UnsafeCell` dereference is therefore single-threaded despite
//! the shared `Arc`. A concurrent `POLLING → NOTIFIED` transition may occur
//! while the poll owner's `&mut` future borrow is live because it transfers no
//! cell-access permission. That borrow is dropped before any transition that
//! does transfer permission to a successor poll or rejected-queue completion
//! owner, so neither can observe it.
//!
//! A wake arriving mid-poll CASes `POLLING → NOTIFIED` rather than enqueuing, so
//! it is never lost: the poll owner re-polls inline (bounded by
//! `ASYNC_INLINE_REPOLL_LIMIT`) or reschedules. If that bounded reschedule is
//! rejected by a full queue, the task completes with `ResourceExhausted`
//! instead of recursively re-polling itself on the waking thread. Cross-task
//! inline polls are independently bounded by `ASYNC_INLINE_POLL_DEPTH_LIMIT`;
//! a saturated nested wake completes with the same typed error instead of
//! growing the caller's stack. The future is dropped once, by the
//! `future_present` flag: either the poll owner or rejected-queue completion
//! owner drops it on completion, and `Drop` (reached only after the last `Arc`,
//! whose refcount release/acquire orders the owner's write before the
//! destructor's read) skips an already-dropped future.
//!
//! # Enqueue obligation (wakes survive admission rejection)
//!
//! The `IDLE → QUEUED` winner owns exactly one *enqueue obligation*. A successful
//! admission transfers it to the queued job, whose poll claims `POLLING`. A
//! rejected admission leaves it with the caller, which must either poll inline
//! or complete the task; dropping that obligation would strand `QUEUED` with no
//! job and make every later wake short-circuit as "already scheduled".
//! `schedule_wake` therefore polls inline after the first rejected admission
//! when the thread-local depth budget is available. A nested rejection past that
//! budget exits `QUEUED` through `complete_resource_exhausted` as typed task
//! exhaustion. Only scheduler
//! shutdown — after which no job of any kind can ever be admitted or run —
//! releases the obligation, by reverting `QUEUED → IDLE`.
//! The spawn-time `schedule` instead propagates admission failure to the
//! spawner (the spawn-backpressure contract) after the same revert, which is
//! race-free there because wakers are minted only inside `poll`.

use std::{
    cell::{Cell, UnsafeCell},
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
    registry::{OwnedStateLease, RunningTaskToken, StateLease, TaskLifecycleToken},
    schedule::{AsyncTask, WorkSubmit},
};

const ASYNC_IDLE: u8 = 0;
const ASYNC_QUEUED: u8 = 1;
const ASYNC_POLLING: u8 = 2;
const ASYNC_NOTIFIED: u8 = 3;
const ASYNC_COMPLETED: u8 = 4;
const ASYNC_INLINE_REPOLL_LIMIT: usize = 1;
const ASYNC_INLINE_POLL_DEPTH_LIMIT: usize = 1;

thread_local! {
    static ASYNC_INLINE_POLL_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct InlinePollDepthGuard {
    previous: usize,
}

impl InlinePollDepthGuard {
    fn try_enter() -> Option<Self> {
        ASYNC_INLINE_POLL_DEPTH.with(|depth| {
            let previous = depth.get();
            (previous < ASYNC_INLINE_POLL_DEPTH_LIMIT).then(|| {
                depth.set(previous + 1);
                Self { previous }
            })
        })
    }
}

impl Drop for InlinePollDepthGuard {
    fn drop(&mut self) {
        ASYNC_INLINE_POLL_DEPTH.with(|depth| depth.set(self.previous));
    }
}

enum PendingPoll {
    Return,
    Repoll,
    Reschedule,
}

pub(super) enum AsyncLifecycle<L: StateLease> {
    Registered(TaskLifecycleToken<L>),
    Running(RunningTaskToken<L>),
    Completed,
}

pub(crate) struct AsyncFutureState<S, F, L = OwnedStateLease>
where
    F: Future,
    L: StateLease,
{
    lifecycle: UnsafeCell<AsyncLifecycle<L>>,
    future: UnsafeCell<MaybeUninit<F>>,
    result_sender: UnsafeCell<Option<TaskResultSender<F::Output>>>,
    metrics: Arc<ExecutorMetrics>,
    state: AtomicU8,
    future_present: UnsafeCell<bool>,
    // Must remain last: production lifecycle leases borrow storage retained by
    // the scheduler, so every lease must retire before scheduler destruction.
    scheduler: S,
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
            state: AtomicU8::new(ASYNC_IDLE),
            future_present: UnsafeCell::new(true),
            scheduler,
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
    /// On admission rejection the waking thread polls the future itself —
    /// mirroring how `SchedulerScope::flush` runs admission-refused jobs on the
    /// calling lane. This cannot lose the transition because `poll` consumes
    /// the `QUEUED` state directly, and it keeps saturated pools independent of
    /// OS yield latency or a queue that only a gated worker can drain.
    fn schedule_wake(self: &Arc<Self>) {
        if !self.claim_enqueue() {
            return;
        }
        match Arc::clone(self).enqueue() {
            Ok(()) => {}
            Err(ExecutorError::ResourceExhausted(_)) => {
                if let Some(_depth_guard) = InlinePollDepthGuard::try_enter() {
                    // Registry diagnostics report the task as running off the
                    // worker pool; `NO_WORKER` is display-only there.
                    self.poll(crate::registry::state::NO_WORKER);
                } else {
                    self.complete_resource_exhausted();
                }
            }
            Err(_) => {
                // ShuttingDown: the scheduler admits and runs nothing from
                // here on, so no poll of this task can ever be admitted —
                // reverting keeps the state honest for `Drop`.
                self.state.store(ASYNC_IDLE, Ordering::Release);
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
                Ok(Poll::Pending) => match self.finish_pending_poll(&mut inline_repolls) {
                    PendingPoll::Return => return,
                    PendingPoll::Repoll => continue,
                    PendingPoll::Reschedule => {
                        self.reschedule_notified();
                        return;
                    }
                },
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

    /// Discharge a wake absorbed after the inline-repoll budget was consumed.
    ///
    /// The current poll owner transfers `NOTIFIED` directly to `QUEUED`, so no
    /// recursive call can grow the waking thread's stack. Persistent admission
    /// saturation becomes an explicit task failure; the wake is never silently
    /// dropped and the caller can distinguish resource exhaustion from output.
    fn reschedule_notified(self: &Arc<Self>) {
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
                self.state.store(ASYNC_IDLE, Ordering::Release);
            }
        }
    }

    /// Complete a rejected `QUEUED` epoch without polling its future.
    ///
    /// The caller exclusively owns this epoch: it either won `IDLE → QUEUED`
    /// or transferred `NOTIFIED → QUEUED`, and `enqueue` returned the job rather
    /// than admitting it. Concurrent wakers cannot leave `QUEUED`, and no poll
    /// job exists, so this owner may drop and publish completion exactly once.
    fn complete_resource_exhausted(&self) {
        debug_assert_eq!(self.state.load(Ordering::Acquire), ASYNC_QUEUED);
        self.drop_future();
        self.state.store(ASYNC_COMPLETED, Ordering::Release);
        self.complete_lifecycle();
        if let Some(sender) = self.take_result_sender() {
            sender.send(Err(TaskError::ResourceExhausted));
        }
        self.metrics.record_task_failed();
    }
}

impl<S, F, L> Drop for AsyncFutureState<S, F, L>
where
    F: Future,
    L: StateLease,
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

impl<S, F, L> Wake for AsyncFutureState<S, F, L>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    L: StateLease,
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
        time::Duration,
    };

    use moirai_core::{
        error::{ExecutorError, ExecutorResult, TaskError},
        executor::{config::DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY, ExecutorConfig},
        task::{TaskHandle, TaskId},
        Priority,
    };

    use super::AsyncFutureState;
    use crate::metrics::ExecutorMetrics;
    use crate::registry::TaskRegistry;
    use crate::schedule::{SyncTask, ThreadScheduler, WorkClass, WorkSubmit};

    impl WorkSubmit for Arc<TaskRegistry> {
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
            task(0);
            Ok(())
        }
    }

    /// Four scheduler phases remain below the 30-second slow-test threshold
    /// even if each consumes its complete event deadline.
    const TEST_EVENT_DEADLINE: Duration = Duration::from_secs(5);

    #[test]
    fn scheduled_lifecycle_retires_before_its_registry_owner() {
        let registry = Arc::new(TaskRegistry::new());
        // SAFETY: the async state owns the only remaining registry Arc
        // through its scheduler field, which is declared after lifecycle.
        let (task_id, lifecycle) = unsafe { registry.register_next_scheduled_task() };
        let registry_owner = Arc::downgrade(&registry);
        let (_handle, result_sender) = TaskHandle::<()>::new_pending(TaskId(task_id));
        let state = AsyncFutureState::new(
            Arc::clone(&registry),
            std::future::pending::<()>(),
            lifecycle,
            result_sender,
            Arc::new(ExecutorMetrics::new()),
        );

        drop(registry);
        assert!(registry_owner.upgrade().is_some());
        drop(state);
        assert!(registry_owner.upgrade().is_none());
    }

    struct GateRelease(Option<mpsc::Sender<()>>);

    impl GateRelease {
        fn release(&mut self) {
            self.0
                .take()
                .expect("gate release is sent once")
                .send(())
                .expect("gated worker remains alive");
        }
    }

    impl Drop for GateRelease {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                // Receiver exit means the worker already left the gate, so no
                // cleanup action remains for this non-panicking test guard.
                match sender.send(()) {
                    Ok(()) | Err(_) => {}
                }
            }
        }
    }

    /// Returns `Pending` once, publishing its waker, then `Ready(output)`.
    ///
    /// The two-poll shape is the minimal future whose completion *requires* a
    /// wake to be honored: losing the wake leaves it parked forever.
    struct WakeThenReady {
        output: i32,
        polls: Arc<AtomicUsize>,
        waker: Arc<Mutex<Option<Waker>>>,
        first_poll_sender: Option<mpsc::Sender<()>>,
    }

    struct AlwaysSelfWake {
        polls: Arc<AtomicUsize>,
    }

    struct WakePeerThenReady {
        output: i32,
        polls: Arc<AtomicUsize>,
        waker: Arc<Mutex<Option<Waker>>>,
        peer_waker: Arc<Mutex<Option<Waker>>>,
    }

    impl Future for AlwaysSelfWake {
        type Output = i32;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<i32> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    impl Future for WakePeerThenReady {
        type Output = i32;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<i32> {
            let this = self.get_mut();
            if this.polls.fetch_add(1, Ordering::SeqCst) == 0 {
                *this.waker.lock().unwrap() = Some(cx.waker().clone());
                Poll::Pending
            } else {
                let peer_waker = this
                    .peer_waker
                    .lock()
                    .unwrap()
                    .as_ref()
                    .cloned()
                    .expect("peer first poll must publish its waker");
                peer_waker.wake_by_ref();
                Poll::Ready(this.output)
            }
        }
    }

    impl Future for WakeThenReady {
        type Output = i32;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<i32> {
            let this = self.get_mut();
            if this.polls.fetch_add(1, Ordering::SeqCst) == 0 {
                *this.waker.lock().unwrap() = Some(cx.waker().clone());
                if let Some(sender) = this.first_poll_sender.take() {
                    sender.send(()).expect("first-poll observer alive");
                }
                Poll::Pending
            } else {
                Poll::Ready(this.output)
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
        pending_async_state_inner(scheduler, output, None)
    }

    fn pending_async_state_with_first_poll_signal<S: WorkSubmit>(
        scheduler: S,
        output: i32,
    ) -> (PendingTask<S>, mpsc::Receiver<()>) {
        let (sender, receiver) = mpsc::channel();
        (
            pending_async_state_inner(scheduler, output, Some(sender)),
            receiver,
        )
    }

    fn pending_async_state_inner<S: WorkSubmit>(
        scheduler: S,
        output: i32,
        first_poll_sender: Option<mpsc::Sender<()>>,
    ) -> PendingTask<S> {
        let registry = TaskRegistry::new();
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
                first_poll_sender,
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

    /// A rejected wake polls inline with no retry or lost output.
    fn wake_survives_admission_rejection(output: i32) {
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

        injector.refuse_next.store(1, Ordering::SeqCst);
        waker.wake();
        assert_eq!(
            injector.rejections.load(Ordering::SeqCst),
            1,
            "the full injector must reject exactly one admission"
        );
        assert_eq!(
            injector.refuse_next.load(Ordering::SeqCst),
            0,
            "the rejection must be consumed"
        );

        assert_eq!(polls.load(Ordering::SeqCst), 2);
        assert_eq!(handle.join(), Some(Ok(output)));
    }

    #[test]
    fn wake_polls_inline_after_admission_rejection() {
        wake_survives_admission_rejection(41);
    }

    #[test]
    fn repeated_self_wake_reports_saturated_reschedule_without_recursion() {
        let injector = GatedInjector::new();
        let registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();
        let (handle, result_sender) = TaskHandle::new_pending(TaskId(task_id));
        let polls = Arc::new(AtomicUsize::new(0));
        let state = AsyncFutureState::new(
            Arc::clone(&injector),
            AlwaysSelfWake {
                polls: Arc::clone(&polls),
            },
            lifecycle,
            result_sender,
            Arc::new(ExecutorMetrics::new()),
        );

        Arc::clone(&state).schedule().expect("first poll admits");
        injector.refuse_next.store(1, Ordering::SeqCst);
        injector.drain();

        assert_eq!(polls.load(Ordering::SeqCst), 2);
        assert_eq!(injector.rejections.load(Ordering::SeqCst), 1);
        assert!(handle.is_finished());
        assert_eq!(handle.join(), Some(Err(TaskError::ResourceExhausted)));
    }

    #[test]
    fn cross_task_wake_respects_inline_poll_depth_bound() {
        let injector = GatedInjector::new();
        let follower = pending_async_state(Arc::clone(&injector), 19);
        let recovery = pending_async_state(Arc::clone(&injector), 23);
        let registry = TaskRegistry::new();
        let (leader_id, leader_lifecycle) = registry.register_next_task();
        let (leader_handle, leader_sender) = TaskHandle::new_pending(TaskId(leader_id));
        let leader_polls = Arc::new(AtomicUsize::new(0));
        let leader_waker = Arc::new(Mutex::new(None));
        let leader = AsyncFutureState::new(
            Arc::clone(&injector),
            WakePeerThenReady {
                output: 17,
                polls: Arc::clone(&leader_polls),
                waker: Arc::clone(&leader_waker),
                peer_waker: Arc::clone(&follower.waker),
            },
            leader_lifecycle,
            leader_sender,
            Arc::new(ExecutorMetrics::new()),
        );

        Arc::clone(&follower.state)
            .schedule()
            .expect("follower first poll admits");
        Arc::clone(&recovery.state)
            .schedule()
            .expect("recovery first poll admits");
        Arc::clone(&leader)
            .schedule()
            .expect("leader first poll admits");
        injector.drain();

        injector.refuse_next.store(2, Ordering::SeqCst);
        leader_waker
            .lock()
            .unwrap()
            .take()
            .expect("leader first poll must publish its waker")
            .wake();

        assert_eq!(leader_polls.load(Ordering::SeqCst), 2);
        assert_eq!(follower.polls.load(Ordering::SeqCst), 1);
        assert_eq!(leader_handle.join(), Some(Ok(17)));
        assert_eq!(
            follower.handle.join(),
            Some(Err(TaskError::ResourceExhausted))
        );
        assert_eq!(injector.rejections.load(Ordering::SeqCst), 2);

        injector.refuse_next.store(1, Ordering::SeqCst);
        recovery
            .waker
            .lock()
            .unwrap()
            .take()
            .expect("recovery first poll must publish its waker")
            .wake();
        assert_eq!(recovery.polls.load(Ordering::SeqCst), 2);
        assert_eq!(recovery.handle.join(), Some(Ok(23)));
    }

    /// M1 regression at the real scheduler: a worker's injector is provably
    /// full (fill until `ResourceExhausted`) and the only worker is gated, so
    /// a wake can never be admitted — the waking thread must poll inline and
    /// the woken task must still complete. Before the fix, `Waker::wake`
    /// discarded the rejection and the task stayed `QUEUED` forever.
    #[test]
    fn woken_task_completes_while_worker_injector_is_full() {
        let scheduler = ThreadScheduler::<8>::from_executor_config(&ExecutorConfig {
            worker_threads: 1,
            max_global_queue_size: 8,
            local_queue_initial_capacity: DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
            thread_name_prefix: "wake-full-injector".into(),
            ..ExecutorConfig::default()
        })
        .expect("scheduler");
        let (
            PendingTask {
                state,
                handle,
                waker,
                polls,
            },
            first_poll_receiver,
        ) = pending_async_state_with_first_poll_signal(scheduler.clone(), 1789);

        // First poll runs on the worker and publishes the waker.
        Arc::clone(&state).schedule().expect("first poll admits");
        first_poll_receiver
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("first poll must publish its waker before the event deadline");

        // Gate the only worker inside a job so nothing can drain the injector.
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let mut gate_release = GateRelease(Some(release_tx));
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_worker| {
                entered_tx.send(()).expect("test observer alive");
                release_rx.recv().expect("release signal");
            })
            .expect("gate job admits");
        entered_rx
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("worker must enter the gate before the event deadline");
        let waker = waker
            .lock()
            .unwrap()
            .take()
            .expect("first poll published its waker");
        assert_eq!(polls.load(Ordering::SeqCst), 1);

        // Start and park the waking thread before saturating the scheduler. The
        // timed wake phase then measures the wake path, not OS thread creation
        // latency under a concurrently loaded workspace test run.
        let wake_phase = Arc::new(AtomicUsize::new(0));
        let wake_phase_thread = Arc::clone(&wake_phase);
        let (wake_ready_tx, wake_ready_rx) = mpsc::sync_channel(1);
        let (wake_start_tx, wake_start_rx) = mpsc::sync_channel(0);
        let (wake_done_tx, wake_done_rx) = mpsc::sync_channel(1);
        let waking = std::thread::spawn(move || {
            wake_phase_thread.store(1, Ordering::SeqCst);
            wake_ready_tx.send(()).expect("wake-ready observer alive");
            wake_start_rx.recv().expect("wake-start observer alive");
            wake_phase_thread.store(2, Ordering::SeqCst);
            waker.wake();
            wake_phase_thread.store(3, Ordering::SeqCst);
            wake_done_tx.send(()).expect("wake observer alive");
        });
        wake_ready_rx
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("waking thread must park before the event deadline");

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
        assert!(
            saw_rejection,
            "an 8-slot injector must fill within 4096 pushes"
        );

        // The injector is full and its only drain is gated: every enqueue
        // retry rejects, so the wake must complete the future inline. The
        // handle resolving *before* the gate opens proves no worker polled.
        wake_start_tx
            .send(())
            .expect("parked waking thread remains alive");
        wake_done_rx
            .recv_timeout(TEST_EVENT_DEADLINE)
            .unwrap_or_else(|error| {
                panic!(
                    "saturated wake must complete before the event deadline: {error}; phase={}, state={}, polls={}, finished={}, pending={}",
                    wake_phase.load(Ordering::SeqCst),
                    state.state.load(Ordering::SeqCst),
                    polls.load(Ordering::SeqCst),
                    handle.is_finished(),
                    scheduler.pending_tasks()
                )
            });
        waking.join().expect("waking thread");
        let poll_count_before_release = polls.load(Ordering::SeqCst);
        let finished_before_release = handle.is_finished();

        gate_release.release();
        scheduler.shutdown();

        assert_eq!(poll_count_before_release, 2);
        assert!(
            finished_before_release,
            "the wake must poll inline before the gated worker is released"
        );
        assert_eq!(handle.join(), Some(Ok(1789)));
    }
}
