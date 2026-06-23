use moirai_core::{Priority, TaskId};
use std::future::Future;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::AtomicBool;
use std::task::{Context, Poll};
use std::time::Instant;

#[allow(dead_code)]
pub(super) struct AsyncTask {
    pub(super) task_id: TaskId,
    // future: ErasedTaskFuture
    pub(super) future: std::cell::UnsafeCell<ErasedTaskFuture>,
    pub(super) future_lock: std::sync::Mutex<()>,
    pub(super) is_queued: AtomicBool,
    pub(super) priority: Priority,
    pub(super) created_at: Instant,
}

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
