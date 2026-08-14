//! Type-erased async task and its wake/poll/complete re-entry protocol.
//!
//! `AsyncTask` is one spawned future as the executor sees it: the future itself
//! type-erased into `ErasedTaskFuture`, plus the flags that decide when it may
//! be polled. Tasks are shared as `Arc<AsyncTask>` between the run queue, the
//! `ExecutorWaker` handed to the future, and any thread running
//! `AsyncExecutor::process_pending_tasks`, so both the erasure and the flags
//! carry cross-thread contracts.
//!
//! # Type erasure
//!
//! `ErasedTaskFuture` is a hand-built vtable — a `NonNull<()>` to the boxed
//! future plus `poll`/`drop` function pointers monomorphized for the concrete
//! `F` at construction. This keeps a heterogeneous run queue without boxing
//! every task behind `dyn Future` at each call site. Three invariants make it
//! sound:
//!
//! 1. **Address stability.** `new` heap-allocates the future with `Box` and
//!    never moves it again, so the `Pin::new_unchecked` in
//!    `poll_erased_future` is honest: the future is pinned for the whole
//!    lifetime of the `ErasedTaskFuture` that owns it.
//! 2. **Type agreement.** `ptr`, `poll`, and `drop` are set together from the
//!    same `F` and never reassigned, so the `cast::<F>()` inside each function
//!    pointer always recovers the type that was actually stored.
//! 3. **Single drop.** The `ErasedTaskFuture` uniquely owns the allocation, and
//!    only its `Drop` frees it (via `Box::from_raw` through the stored `drop`
//!    pointer). `Send` is justified by the `F: Send` bound at construction —
//!    the pointer moves between threads only with the task it belongs to.
//!
//! # Re-entry protocol (`is_queued` / `completed` / `future_lock`)
//!
//! A future must be polled by one thread at a time and must never be polled
//! after it returns `Ready` — a completed `async` block panics with "resumed
//! after completion". Three fields enforce that, and they divide the work:
//!
//! - `is_queued` — enqueue deduplication, owned by `ExecutorWaker::wake_by_ref`
//!   (`waker.rs`). A wake enqueues only if it flipped `is_queued` false→true, so
//!   a task appears in the run queue at most once per pending wake. The flag is
//!   a Relaxed linearization bit rather than a publication channel: the queue's
//!   per-slot Release/Acquire sequence publishes the task payload, and the
//!   ordering protocol is exhaustively modeled in `loom_wake_dedup.rs`.
//! - `completed` — set once the future returns `Ready`. The waker checks it as
//!   an optimization (a reactor may still hold a live waker for a task that
//!   finished by another path, e.g. `timeout(read)` completing via the timer
//!   while a socket read-waker stays registered), but the authoritative guard is
//!   in `process_pending_tasks`.
//! - `future_lock` — serializes access to the `UnsafeCell<ErasedTaskFuture>`.
//!   Its guard is held across the `completed` check *and* the poll, which is
//!   what makes the guard sound: `process_pending_tasks` clears `is_queued`
//!   before polling (so a self-wake during the poll can re-enqueue), so a second
//!   polling thread can dequeue the same task while the first still holds the
//!   lock. Checking `completed` outside the lock would let that thread pass the
//!   check, block, and then poll a future the first thread completed meanwhile.
//!
//! Together these give the `unsafe impl Sync` below its meaning: every mutable
//! touch of the future happens under `future_lock`, and every poll is gated by a
//! `completed` check taken in the same critical section.

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
// mutable access to it is serialized by `future_lock`, whose guard is held
// across both the `completed` check and the poll it guards (see the re-entry
// protocol in the module docs), so no two threads touch the future
// concurrently and none polls it after completion.
unsafe impl Sync for AsyncTask {}

pub(super) struct ErasedTaskFuture {
    ptr: NonNull<()>,
    poll: unsafe fn(NonNull<()>, &mut Context<'_>) -> Poll<()>,
    drop: unsafe fn(NonNull<()>),
}

// SAFETY: the erased pointer owns a `Box<F>` built under an `F: Send` bound in
// `new`, so moving the task (and with it this pointer) across threads moves a
// `Send` value; the vtable entries are plain fn pointers.
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
