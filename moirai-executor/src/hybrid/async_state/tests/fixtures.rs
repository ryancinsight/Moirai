use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll, Waker},
};

use moirai_core::{
    Priority,
    error::{ExecutorError, ExecutorResult},
    task::{TaskHandle, TaskId},
};

use super::super::AsyncFutureState;
use crate::metrics::ExecutorMetrics;
use crate::registry::TaskRegistry;
use crate::schedule::{WorkClass, WorkSubmit};

/// Returns `Pending` once, publishing its waker, then `Ready(output)`.
///
/// The two-poll shape is the minimal future whose completion *requires* a
/// wake to be honored: losing the wake leaves it parked forever.
pub(super) struct WakeThenReady {
    pub(super) output: i32,
    pub(super) polls: Arc<AtomicUsize>,
    pub(super) waker: Arc<Mutex<Option<Waker>>>,
    pub(super) first_poll_sender: Option<mpsc::Sender<()>>,
}

pub(super) struct AlwaysSelfWake {
    pub(super) polls: Arc<AtomicUsize>,
}

pub(super) struct WakePeerThenReady {
    pub(super) output: i32,
    pub(super) polls: Arc<AtomicUsize>,
    pub(super) waker: Arc<Mutex<Option<Waker>>>,
    pub(super) peer_waker: Arc<Mutex<Option<Waker>>>,
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
pub(super) struct PendingTask<S> {
    pub(super) state: Arc<AsyncFutureState<S, WakeThenReady>>,
    pub(super) handle: TaskHandle<i32>,
    pub(super) waker: Arc<Mutex<Option<Waker>>>,
    pub(super) polls: Arc<AtomicUsize>,
}

pub(super) fn pending_async_state<S: WorkSubmit>(scheduler: S, output: i32) -> PendingTask<S> {
    register_pending_async_state(scheduler, output, None)
}

pub(super) fn pending_async_state_with_first_poll_signal<S: WorkSubmit>(
    scheduler: S,
    output: i32,
) -> (PendingTask<S>, mpsc::Receiver<()>) {
    let (sender, receiver) = mpsc::channel();
    (
        register_pending_async_state(scheduler, output, Some(sender)),
        receiver,
    )
}

fn register_pending_async_state<S: WorkSubmit>(
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
/// Once `shutting_down` is set it refuses every admission the way a
/// stopped scheduler does.
pub(super) struct GatedInjector {
    jobs: Mutex<Vec<QueuedJob>>,
    pub(super) refuse_next: AtomicUsize,
    pub(super) rejections: AtomicUsize,
    pub(super) shutting_down: AtomicBool,
}

impl GatedInjector {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            jobs: Mutex::new(Vec::new()),
            refuse_next: AtomicUsize::new(0),
            rejections: AtomicUsize::new(0),
            shutting_down: AtomicBool::new(false),
        })
    }

    pub(super) fn drain(&self) {
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
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err(ExecutorError::ShuttingDown);
        }
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
