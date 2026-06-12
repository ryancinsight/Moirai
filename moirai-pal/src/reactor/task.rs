use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::task::{Context, Poll, Waker};

use super::future::ErasedReactorTaskFuture;

/// Task identifier for tracking async operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub(crate) u64);

impl TaskId {
    // `new` mints a process-unique id from a global counter; a `Default` impl
    // would misrepresent that side effect as a neutral value.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

pub struct ReactorTaskState {
    pub(crate) future: ErasedReactorTaskFuture,
    pub(crate) completion: TaskCompletion,
}

// Safety: `future` is accessed only while the task state is owned by the
// reactor queue. `TaskHandle` only reads completion state. The queue mutex
// serializes all future polling and dropping.
unsafe impl Send for ReactorTaskState {}

// Safety: shared references never expose mutable future access outside the
// queue-processing path guarded by `IoReactor::task_queue`.
unsafe impl Sync for ReactorTaskState {}

impl ReactorTaskState {
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        Self {
            future: ErasedReactorTaskFuture::new(future),
            completion: TaskCompletion::new(),
        }
    }

    pub fn poll_future(&self, context: &mut Context<'_>) -> Poll<()> {
        self.future.poll(context)
    }

    pub fn complete(&self) {
        self.future.take();
        self.completion.complete();
    }
}

#[derive(Debug)]
pub struct TaskCompletion {
    pub(crate) completed: AtomicBool,
    pub(crate) waker: Mutex<Option<Waker>>,
}

impl Default for TaskCompletion {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskCompletion {
    pub fn new() -> Self {
        Self {
            completed: AtomicBool::new(false),
            waker: Mutex::new(None),
        }
    }

    pub fn complete(&self) {
        self.completed.store(true, Ordering::Release);
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }

    pub fn poll(&self, cx: &Context<'_>) -> Poll<()> {
        if self.completed.load(Ordering::Acquire) {
            return Poll::Ready(());
        }

        *self.waker.lock().unwrap() = Some(cx.waker().clone());

        if self.completed.load(Ordering::Acquire) {
            self.waker.lock().unwrap().take();
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Handle for tracking spawned tasks.
pub struct TaskHandle {
    pub(crate) task_id: TaskId,
    pub(crate) task: Arc<ReactorTaskState>,
}

impl TaskHandle {
    pub fn new(task_id: TaskId, task: Arc<ReactorTaskState>) -> Self {
        Self { task_id, task }
    }

    /// Get the task ID.
    pub fn id(&self) -> TaskId {
        self.task_id
    }
}

impl Future for TaskHandle {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.task.completion.poll(cx)
    }
}
