use std::cell::UnsafeCell;
use std::future::Future;
use std::mem::{align_of, size_of, MaybeUninit};
use std::pin::Pin;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

pub const INLINE_REACTOR_TASK_WORDS: usize = 14;

/// Stable inline storage for one concrete reactor future.
#[repr(C)]
pub struct ReactorTaskFutureStorage {
    pub words: [MaybeUninit<usize>; INLINE_REACTOR_TASK_WORDS],
}

impl ReactorTaskFutureStorage {
    pub fn new() -> Self {
        Self {
            words: [MaybeUninit::uninit(); INLINE_REACTOR_TASK_WORDS],
        }
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.words.as_mut_ptr().cast::<T>()
    }
}

/// Heap-stable concrete future storage with monomorphized poll/drop functions.
#[repr(C, align(64))]
pub struct ErasedReactorTaskFuture {
    pub storage: UnsafeCell<ReactorTaskFutureStorage>,
    pub poll: unsafe fn(*mut ReactorTaskFutureStorage, &mut Context<'_>) -> Poll<()>,
    pub drop: unsafe fn(*mut ReactorTaskFutureStorage),
    pub present: AtomicBool,
}

// Safety: the owner stores a `Send + 'static` future either inline inside a
// heap-stable `ReactorTaskState` or behind a `Box<F>` stored inline.
unsafe impl Send for ErasedReactorTaskFuture {}

// Safety: mutable access is serialized by the reactor queue; shared handles do
// not touch the future storage.
unsafe impl Sync for ErasedReactorTaskFuture {}

impl ErasedReactorTaskFuture {
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        if reactor_future_fits::<F>() {
            Self::new_inline(future)
        } else {
            Self::new_boxed(future)
        }
    }

    pub fn new_inline<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        debug_assert!(reactor_future_fits::<F>());
        let this = Self {
            storage: UnsafeCell::new(ReactorTaskFutureStorage::new()),
            poll: poll_inline_reactor_future::<F>,
            drop: drop_inline_reactor_future::<F>,
            present: AtomicBool::new(true),
        };

        // Safety: `reactor_future_fits` proves size and alignment fit the
        // storage. The enclosing erased future supplies cache-line alignment
        // and the storage field is first, so the field address inherits it.
        unsafe {
            (*this.storage.get()).as_mut_ptr::<F>().write(future);
        }

        this
    }

    pub fn new_boxed<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        debug_assert!(reactor_future_fits::<Box<F>>());
        let this = Self {
            storage: UnsafeCell::new(ReactorTaskFutureStorage::new()),
            poll: poll_boxed_reactor_future::<F>,
            drop: drop_boxed_reactor_future::<F>,
            present: AtomicBool::new(true),
        };

        // Safety: a `Box<F>` is pointer-sized and fits the inline storage. The
        // boxed allocation keeps `F` pinned by address after the first poll.
        unsafe {
            (*this.storage.get())
                .as_mut_ptr::<Box<F>>()
                .write(Box::new(future));
        }

        this
    }

    pub fn poll(&self, context: &mut Context<'_>) -> Poll<()> {
        debug_assert!(self.present.load(Ordering::Acquire));
        // Safety: `new_inline` initialized storage as the same concrete type
        // used to create this monomorphized poll function. Queue ownership
        // serializes mutable access to the future.
        unsafe { (self.poll)(self.storage.get(), context) }
    }

    pub fn take(&self) {
        if self.present.swap(false, Ordering::AcqRel) {
            // Safety: storage contains the initialized future until this method
            // consumes the `present` flag.
            unsafe {
                (self.drop)(self.storage.get());
            }
        }
    }
}

impl Drop for ErasedReactorTaskFuture {
    fn drop(&mut self) {
        self.take();
    }
}

pub fn reactor_future_fits<F>() -> bool {
    size_of::<F>() <= size_of::<ReactorTaskFutureStorage>()
        && align_of::<F>() <= align_of::<ErasedReactorTaskFuture>()
}

pub unsafe fn poll_inline_reactor_future<F>(
    storage: *mut ReactorTaskFutureStorage,
    context: &mut Context<'_>,
) -> Poll<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    // Safety: `ErasedReactorTaskFuture::new_inline::<F>` initialized this
    // storage as `F`, and `ReactorTaskState` keeps the storage address stable.
    let future = unsafe { Pin::new_unchecked(&mut *(*storage).as_mut_ptr::<F>()) };
    future.poll(context)
}

pub unsafe fn drop_inline_reactor_future<F>(storage: *mut ReactorTaskFutureStorage)
where
    F: Future<Output = ()> + Send + 'static,
{
    // Safety: called only when the inline future was initialized and has not
    // already been consumed by `take`.
    unsafe { ptr::drop_in_place((*storage).as_mut_ptr::<F>()) };
}

pub unsafe fn drop_boxed_reactor_future<F>(storage: *mut ReactorTaskFutureStorage)
where
    F: Future<Output = ()> + Send + 'static,
{
    // Safety: called only when the boxed future was initialized and has not
    // already been consumed by `take`.
    unsafe { ptr::drop_in_place((*storage).as_mut_ptr::<Box<F>>()) };
}

pub unsafe fn poll_boxed_reactor_future<F>(
    storage: *mut ReactorTaskFutureStorage,
    context: &mut Context<'_>,
) -> Poll<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    // Safety: `new_boxed::<F>` stores a `Box<F>` in the inline storage. The box
    // keeps `F` heap-stable even if the small box pointer is moved.
    let future = unsafe { &mut *(*storage).as_mut_ptr::<Box<F>>() };
    let future = unsafe { Pin::new_unchecked(future.as_mut()) };
    future.poll(context)
}
