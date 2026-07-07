use moirai_core::{Priority, TaskId};
use std::future::Future;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::AtomicBool;
use std::task::{Context, Poll};
use std::time::Instant;

pub(super) struct AsyncTask {
    /// Task identifier assigned at spawn. `AsyncHandle` carries its own copy;
    /// no executor path reads this field back yet (introspection-only).
    #[allow(dead_code)] // stored for parity with AsyncHandle; no read path wired yet
    pub(super) task_id: TaskId,
    /// Type-erased future slot (`future: ErasedTaskFuture` behind an
    /// `UnsafeCell`); mutation is serialized by `future_lock`.
    pub(super) future: std::cell::UnsafeCell<ErasedTaskFuture>,
    pub(super) future_lock: std::sync::Mutex<()>,
    pub(super) is_queued: AtomicBool,
    /// Set once the future returns `Poll::Ready`. Polling a completed
    /// `async` block again panics ("resumed after completion"), so every
    /// enqueue/poll path checks this first: a stale reactor waker fired after
    /// the task already finished (e.g. `timeout(read)` completing via the
    /// timer while the socket's registered read-waker is still live) must not
    /// re-enqueue or re-poll the future.
    pub(super) completed: AtomicBool,
    /// Scheduling priority recorded at spawn. The run queue in
    /// `executor/core.rs` is FIFO and does not consult this field, so
    /// `spawn_with_priority` currently has no scheduling effect; a
    /// priority-aware run queue is pending in the executor core.
    #[allow(dead_code)] // pending priority-aware run queue in executor/core.rs
    pub(super) priority: Priority,
    /// Spawn timestamp; not read by any executor path yet (latency accounting
    /// pending alongside the stats wiring in `executor/core.rs`).
    #[allow(dead_code)] // pending queue-latency accounting in executor/core.rs
    pub(super) created_at: Instant,
}

// SAFETY: the only non-Sync field is the `UnsafeCell<ErasedTaskFuture>`;
// mutable access to it is serialized by `future_lock` (its guard is held for
// every poll), and the `completed`/`is_queued` flags gate re-entry, so no two
// threads touch the future concurrently.
unsafe impl Sync for AsyncTask {}

pub(super) struct ErasedTaskFuture {
    ptr: NonNull<()>,
    poll: unsafe fn(NonNull<()>, &mut Context<'_>) -> Poll<()>,
    drop: unsafe fn(NonNull<()>),
}

unsafe impl Send for ErasedTaskFuture {}

impl ErasedTaskFuture {
    pub(super) fn new<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let ptr = Box::into_raw(Box::new(future)).cast::<()>();
        Self {
            ptr: NonNull::new(ptr).expect("Box::into_raw must not return null"),
            poll: poll_erased_future::<F>,
            drop: drop_erased_future::<F>,
        }
    }

    pub(super) fn poll(&mut self, context: &mut Context<'_>) -> Poll<()> {
        unsafe { (self.poll)(self.ptr, context) }
    }
}

impl Drop for ErasedTaskFuture {
    fn drop(&mut self) {
        unsafe {
            (self.drop)(self.ptr);
        }
    }
}

unsafe fn poll_erased_future<F>(ptr: NonNull<()>, context: &mut Context<'_>) -> Poll<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let future = unsafe { Pin::new_unchecked(&mut *ptr.cast::<F>().as_ptr()) };
    future.poll(context)
}

unsafe fn drop_erased_future<F>(ptr: NonNull<()>)
where
    F: Future<Output = ()> + Send + 'static,
{
    unsafe {
        drop(Box::from_raw(ptr.cast::<F>().as_ptr()));
    }
}
