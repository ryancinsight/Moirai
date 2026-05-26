//! Core reactor implementation for managing async I/O operations.
//!
//! This module provides the central event loop and task scheduling for
//! platform-specific async I/O operations.

use std::cell::{RefCell, UnsafeCell};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::io;
use std::mem::{align_of, size_of, MaybeUninit};
use std::pin::Pin;
use std::ptr;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use crate::{create_reactor, Event, Interest, PlatformReactor, RawFd, Reactor};

const INLINE_REACTOR_TASK_WORDS: usize = 14;

/// Central async I/O reactor managing all platform-specific operations.
pub struct IoReactor {
    /// Platform-specific reactor implementation
    platform_reactor: PlatformReactor,
    /// Event loop control
    running: Arc<AtomicBool>,
    /// Registered file descriptor tracking
    registered_fds: Arc<Mutex<HashMap<FdKey, FdInfo>>>,
    /// Pending task queue
    task_queue: Arc<Mutex<VecDeque<Arc<ReactorTaskState>>>>,
    /// Performance metrics
    metrics: Arc<ReactorMetrics>,
}

/// Send/Sync-safe internal key for platform handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct FdKey(usize);

impl From<RawFd> for FdKey {
    fn from(fd: RawFd) -> Self {
        Self(fd as usize)
    }
}

/// Information about registered file descriptors
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for future telemetry/debugging per ADR requirements
struct FdInfo {
    interest: Interest,
    registered_at: Instant,
    event_count: u64,
    read_waker: Option<Waker>,
    write_waker: Option<Waker>,
}

/// Task identifier for tracking async operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

struct ReactorTaskState {
    future: ErasedReactorTaskFuture,
    completion: TaskCompletion,
}

// Safety: `future` is accessed only while the task state is owned by the
// reactor queue. `TaskHandle` only reads completion state. The queue mutex
// serializes all future polling and dropping.
unsafe impl Send for ReactorTaskState {}

// Safety: shared references never expose mutable future access outside the
// queue-processing path guarded by `IoReactor::task_queue`.
unsafe impl Sync for ReactorTaskState {}

impl ReactorTaskState {
    fn new<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        Self {
            future: ErasedReactorTaskFuture::new(future),
            completion: TaskCompletion::new(),
        }
    }

    fn poll_future(&self, context: &mut Context<'_>) -> Poll<()> {
        self.future.poll(context)
    }

    fn complete(&self) {
        self.future.take();
        self.completion.complete();
    }
}

/// Stable inline storage for one concrete reactor future.
#[repr(C)]
struct ReactorTaskFutureStorage {
    words: [MaybeUninit<usize>; INLINE_REACTOR_TASK_WORDS],
}

/// Heap-stable concrete future storage with monomorphized poll/drop functions.
#[repr(C, align(64))]
struct ErasedReactorTaskFuture {
    storage: UnsafeCell<ReactorTaskFutureStorage>,
    poll: unsafe fn(*mut ReactorTaskFutureStorage, &mut Context<'_>) -> Poll<()>,
    drop: unsafe fn(*mut ReactorTaskFutureStorage),
    present: AtomicBool,
}

// Safety: the owner stores a `Send + 'static` future either inline inside a
// heap-stable `ReactorTaskState` or behind a `Box<F>` stored inline.
unsafe impl Send for ErasedReactorTaskFuture {}

// Safety: mutable access is serialized by the reactor queue; shared handles do
// not touch the future storage.
unsafe impl Sync for ErasedReactorTaskFuture {}

impl ErasedReactorTaskFuture {
    fn new<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        if reactor_future_fits::<F>() {
            Self::new_inline(future)
        } else {
            Self::new_boxed(future)
        }
    }

    fn new_inline<F>(future: F) -> Self
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

    fn new_boxed<F>(future: F) -> Self
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

    fn poll(&self, context: &mut Context<'_>) -> Poll<()> {
        debug_assert!(self.present.load(Ordering::Acquire));
        // Safety: `new_inline` initialized storage as the same concrete type
        // used to create this monomorphized poll function. Queue ownership
        // serializes mutable access to the future.
        unsafe { (self.poll)(self.storage.get(), context) }
    }

    fn take(&self) {
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

impl ReactorTaskFutureStorage {
    fn new() -> Self {
        Self {
            words: [MaybeUninit::uninit(); INLINE_REACTOR_TASK_WORDS],
        }
    }

    fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.words.as_mut_ptr().cast::<T>()
    }
}

fn reactor_future_fits<F>() -> bool {
    size_of::<F>() <= size_of::<ReactorTaskFutureStorage>()
        && align_of::<F>() <= align_of::<ErasedReactorTaskFuture>()
}

unsafe fn poll_inline_reactor_future<F>(
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

unsafe fn drop_inline_reactor_future<F>(storage: *mut ReactorTaskFutureStorage)
where
    F: Future<Output = ()> + Send + 'static,
{
    // Safety: called only when the inline future was initialized and has not
    // already been consumed by `take`.
    unsafe { ptr::drop_in_place((*storage).as_mut_ptr::<F>()) };
}

unsafe fn drop_boxed_reactor_future<F>(storage: *mut ReactorTaskFutureStorage)
where
    F: Future<Output = ()> + Send + 'static,
{
    // Safety: called only when the boxed future was initialized and has not
    // already been consumed by `take`.
    unsafe { ptr::drop_in_place((*storage).as_mut_ptr::<Box<F>>()) };
}

unsafe fn poll_boxed_reactor_future<F>(
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

#[derive(Debug)]
struct TaskCompletion {
    completed: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl TaskCompletion {
    fn new() -> Self {
        Self {
            completed: AtomicBool::new(false),
            waker: Mutex::new(None),
        }
    }

    fn complete(&self) {
        self.completed.store(true, Ordering::Release);
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }

    fn poll(&self, cx: &Context<'_>) -> Poll<()> {
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

/// Performance metrics for the reactor
#[derive(Debug, Default)]
pub struct ReactorMetrics {
    /// Total events processed
    pub events_processed: AtomicU64,
    /// Total tasks executed
    pub tasks_executed: AtomicU64,
    /// Average event processing time (nanoseconds)
    pub avg_event_time_ns: AtomicU64,
    /// Peak number of registered file descriptors
    pub peak_fd_count: AtomicU64,
    /// Reactor uptime
    pub start_time: std::sync::OnceLock<Instant>,
}

impl IoReactor {
    /// Create a new I/O reactor with platform-optimal implementation.
    pub fn new() -> io::Result<Self> {
        let platform_reactor = create_reactor()?;

        Ok(Self {
            platform_reactor,
            running: Arc::new(AtomicBool::new(false)),
            registered_fds: Arc::new(Mutex::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(ReactorMetrics::default()),
        })
    }

    /// Register a file descriptor for async I/O operations.
    pub fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        // Register with platform reactor
        self.platform_reactor.register_fd(fd, interest)?;

        // Track registration
        let mut fds = self.registered_fds.lock().unwrap();
        fds.insert(
            FdKey::from(fd),
            FdInfo {
                interest,
                registered_at: Instant::now(),
                event_count: 0,
                read_waker: None,
                write_waker: None,
            },
        );

        // Update peak FD count metric
        let current_count = fds.len() as u64;
        let peak = self.metrics.peak_fd_count.load(Ordering::Relaxed);
        if current_count > peak {
            self.metrics
                .peak_fd_count
                .store(current_count, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Unregister a file descriptor.
    pub fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        self.platform_reactor.unregister_fd(fd)?;
        self.registered_fds.lock().unwrap().remove(&FdKey::from(fd));
        Ok(())
    }

    /// Spawn an async task on the reactor.
    pub fn spawn<F>(&self, future: F) -> TaskHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task_id = TaskId::new();
        let task = Arc::new(ReactorTaskState::new(future));

        self.task_queue.lock().unwrap().push_back(task.clone());

        TaskHandle { task_id, task }
    }

    /// Run the event loop until stopped.
    pub fn run(&self) -> io::Result<()> {
        self.running.store(true, Ordering::SeqCst);
        self.metrics
            .start_time
            .set(Instant::now())
            .map_err(|_| io::Error::other("Reactor already started"))?;

        while self.running.load(Ordering::SeqCst) {
            self.run_iteration(Some(Duration::from_millis(10)))?;
        }

        Ok(())
    }

    /// Run a single iteration of the event loop.
    pub fn run_iteration(&self, timeout: Option<Duration>) -> io::Result<()> {
        let iteration_start = Instant::now();

        // Process pending tasks first
        self.process_pending_tasks();

        // Poll for I/O events
        let events = self.platform_reactor.poll_events(timeout)?;

        // Process I/O events
        for event in events {
            self.handle_event(event)?;
        }

        // Update metrics
        let iteration_time = iteration_start.elapsed().as_nanos() as u64;
        self.metrics
            .avg_event_time_ns
            .store(iteration_time, Ordering::Relaxed);

        Ok(())
    }

    /// Stop the event loop.
    pub fn stop(&self) -> io::Result<()> {
        self.running.store(false, Ordering::SeqCst);
        self.platform_reactor.wake()
    }

    /// Process all pending tasks in the queue.
    fn process_pending_tasks(&self) {
        let mut tasks = self.task_queue.lock().unwrap();
        let mut completed_tasks = Vec::new();

        for (index, task) in tasks.iter_mut().enumerate() {
            // Create a simple noop waker compatible with MSRV 1.75.0
            // Using standard library patterns per Rust Book Ch.16
            use std::task::{RawWaker, RawWakerVTable, Waker};

            const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
                |_| RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE),
                |_| {},
                |_| {},
                |_| {},
            );

            let waker =
                unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE)) };
            let mut context = Context::from_waker(&waker);

            match task.poll_future(&mut context) {
                Poll::Ready(()) => {
                    task.complete();
                    completed_tasks.push(index);
                    self.metrics.tasks_executed.fetch_add(1, Ordering::Relaxed);
                }
                Poll::Pending => {
                    // Task is still pending, keep it in queue
                }
            }
        }

        // Remove completed tasks (in reverse order to maintain indices)
        for &index in completed_tasks.iter().rev() {
            tasks.remove(index);
        }
    }

    /// Handle a single I/O event.
    fn handle_event(&self, event: Event) -> io::Result<()> {
        // Update FD event count
        if let Ok(mut fds) = self.registered_fds.lock() {
            if let Some(fd_info) = fds.get_mut(&FdKey::from(event.fd)) {
                fd_info.event_count += 1;
            }
        }

        // Update metrics
        self.metrics
            .events_processed
            .fetch_add(1, Ordering::Relaxed);

        // Wake any tasks waiting on this file descriptor
        self.wake_fd_waiters(event);

        Ok(())
    }

    /// Wake tasks waiting on a specific file descriptor event.
    fn wake_fd_waiters(&self, event: Event) {
        let mut read_waker = None;
        let mut write_waker = None;

        if let Ok(mut fds) = self.registered_fds.lock() {
            if let Some(fd_info) = fds.get_mut(&FdKey::from(event.fd)) {
                if event.readable || event.error || event.hangup {
                    read_waker = fd_info.read_waker.take();
                }
                if event.writable || event.error || event.hangup {
                    write_waker = fd_info.write_waker.take();
                }
            }
        }

        if let Some(waker) = read_waker {
            waker.wake();
        }
        if let Some(waker) = write_waker {
            waker.wake();
        }
    }

    /// Register a task's waker for a file descriptor and interest.
    pub fn register_waker(&self, fd: RawFd, interest: Interest, waker: Waker) -> io::Result<()> {
        let mut fds = self.registered_fds.lock().unwrap();
        if let Some(fd_info) = fds.get_mut(&FdKey::from(fd)) {
            let mut new_interest = fd_info.interest;
            let mut modified = false;
            if interest.readable && !new_interest.readable {
                new_interest.readable = true;
                modified = true;
            }
            if interest.writable && !new_interest.writable {
                new_interest.writable = true;
                modified = true;
            }
            if modified {
                self.platform_reactor.unregister_fd(fd)?;
                self.platform_reactor.register_fd(fd, new_interest)?;
                fd_info.interest = new_interest;
            }

            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }
            Ok(())
        } else {
            drop(fds);
            self.register_fd(fd, interest)?;
            let mut fds = self.registered_fds.lock().unwrap();
            let fd_info = fds.get_mut(&FdKey::from(fd)).unwrap();
            if interest.readable {
                fd_info.read_waker = Some(waker.clone());
            }
            if interest.writable {
                fd_info.write_waker = Some(waker);
            }
            Ok(())
        }
    }

    /// Remove wakers for a file descriptor.
    pub fn deregister_waker(&self, fd: RawFd, interest: Interest) {
        if let Ok(mut fds) = self.registered_fds.lock() {
            if let Some(fd_info) = fds.get_mut(&FdKey::from(fd)) {
                if interest.readable {
                    fd_info.read_waker = None;
                }
                if interest.writable {
                    fd_info.write_waker = None;
                }
            }
        }
    }

    /// Wake up the reactor from blocking poll.
    pub fn wake(&self) -> io::Result<()> {
        self.platform_reactor.wake()
    }

    /// Run a closure with this reactor set as the thread-local active reactor.
    pub fn with_active<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let old = ACTIVE_REACTOR.with(|cell| cell.replace(Some(self as *const IoReactor)));
        let result = f();
        ACTIVE_REACTOR.with(|cell| cell.replace(old));
        result
    }

    /// Retrieve the active reactor for the current thread, if any.
    pub fn get_active() -> Option<&'static IoReactor> {
        if let Some(ptr) = ACTIVE_REACTOR.with(|cell| *cell.borrow()) {
            return Some(unsafe { &*ptr });
        }

        #[cfg(not(target_os = "windows"))]
        {
            let reactor = GLOBAL_REACTOR.get_or_init(|| {
                let r = Arc::new(IoReactor::new().expect("failed to create global IoReactor"));
                let r_clone = Arc::clone(&r);
                std::thread::Builder::new()
                    .name("moirai-global-reactor".to_string())
                    .spawn(move || {
                        let _ = r_clone.run();
                    })
                    .expect("failed to spawn global reactor thread");
                r
            });
            Some(&**reactor)
        }

        #[cfg(target_os = "windows")]
        {
            None
        }
    }

    /// Get current reactor metrics.
    pub fn metrics(&self) -> ReactorMetrics {
        ReactorMetrics {
            events_processed: AtomicU64::new(self.metrics.events_processed.load(Ordering::Relaxed)),
            tasks_executed: AtomicU64::new(self.metrics.tasks_executed.load(Ordering::Relaxed)),
            avg_event_time_ns: AtomicU64::new(
                self.metrics.avg_event_time_ns.load(Ordering::Relaxed),
            ),
            peak_fd_count: AtomicU64::new(self.metrics.peak_fd_count.load(Ordering::Relaxed)),
            start_time: std::sync::OnceLock::new(),
        }
    }
}

thread_local! {
    static ACTIVE_REACTOR: RefCell<Option<*const IoReactor>> = const { RefCell::new(None) };
}

#[cfg(not(target_os = "windows"))]
static GLOBAL_REACTOR: std::sync::OnceLock<Arc<IoReactor>> = std::sync::OnceLock::new();

/// Handle for tracking spawned tasks.
pub struct TaskHandle {
    task_id: TaskId,
    task: Arc<ReactorTaskState>,
}

impl TaskHandle {
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::task::noop_waker_ref;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_reactor_creation() {
        let reactor = IoReactor::new();
        assert!(reactor.is_ok());
    }

    #[test]
    fn test_task_id_generation() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_reactor_metrics() {
        let reactor = IoReactor::new().unwrap();
        let metrics = reactor.metrics();
        assert_eq!(metrics.events_processed.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.tasks_executed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn spawned_ready_task_handle_completes_after_iteration() {
        let reactor = IoReactor::new().expect("reactor must be created");
        let handle = reactor.spawn(async {});
        let mut handle = Box::pin(handle);
        let waker = noop_waker_ref();
        let mut context = Context::from_waker(waker);

        assert!(matches!(
            Future::poll(handle.as_mut(), &mut context),
            Poll::Pending
        ));

        reactor
            .run_iteration(Some(Duration::from_millis(0)))
            .expect("reactor iteration must complete");

        assert!(matches!(
            Future::poll(handle.as_mut(), &mut context),
            Poll::Ready(())
        ));

        let metrics = reactor.metrics();
        assert_eq!(metrics.tasks_executed.load(Ordering::Relaxed), 1);
    }

    struct MaxInlineReadyFuture {
        values: [usize; INLINE_REACTOR_TASK_WORDS],
    }

    impl Future for MaxInlineReadyFuture {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            assert_eq!(self.values[0], 1);
            Poll::Ready(())
        }
    }

    struct InlineObservedReadyFuture {
        values: [usize; INLINE_REACTOR_TASK_WORDS - 1],
        observed: Arc<AtomicUsize>,
    }

    impl Future for InlineObservedReadyFuture {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            self.observed.store(self.values[0], Ordering::Relaxed);
            Poll::Ready(())
        }
    }

    struct OversizedShapeFuture {
        values: [usize; INLINE_REACTOR_TASK_WORDS + 1],
    }

    impl Future for OversizedShapeFuture {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            assert_eq!(self.values[0], 1);
            Poll::Ready(())
        }
    }

    struct OversizedObservedReadyFuture {
        values: [usize; INLINE_REACTOR_TASK_WORDS + 1],
        observed: Arc<AtomicUsize>,
    }

    impl Future for OversizedObservedReadyFuture {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            self.observed.store(self.values[0], Ordering::Relaxed);
            Poll::Ready(())
        }
    }

    #[test]
    fn reactor_future_storage_budget_is_static_and_bounded() {
        assert_eq!(
            core::mem::size_of::<ReactorTaskFutureStorage>(),
            INLINE_REACTOR_TASK_WORDS * core::mem::size_of::<usize>()
        );
        assert!(reactor_future_fits::<MaxInlineReadyFuture>());
        assert!(!reactor_future_fits::<OversizedShapeFuture>());
        assert!(reactor_future_fits::<Box<OversizedShapeFuture>>());
    }

    #[test]
    fn spawned_inline_and_oversized_reactor_futures_complete() {
        let reactor = IoReactor::new().expect("reactor must be created");
        let inline_observed = Arc::new(AtomicUsize::new(0));
        let oversized_observed = Arc::new(AtomicUsize::new(0));

        let inline_handle = reactor.spawn(InlineObservedReadyFuture {
            values: [7; INLINE_REACTOR_TASK_WORDS - 1],
            observed: Arc::clone(&inline_observed),
        });
        let oversized_handle = reactor.spawn(OversizedObservedReadyFuture {
            values: [11; INLINE_REACTOR_TASK_WORDS + 1],
            observed: Arc::clone(&oversized_observed),
        });

        reactor
            .run_iteration(Some(Duration::from_millis(0)))
            .expect("reactor iteration must complete");

        assert_eq!(inline_observed.load(Ordering::Relaxed), 7);
        assert_eq!(oversized_observed.load(Ordering::Relaxed), 11);

        let waker = noop_waker_ref();
        let mut context = Context::from_waker(waker);
        let mut inline_handle = Box::pin(inline_handle);
        let mut oversized_handle = Box::pin(oversized_handle);
        assert!(matches!(
            Future::poll(inline_handle.as_mut(), &mut context),
            Poll::Ready(())
        ));
        assert!(matches!(
            Future::poll(oversized_handle.as_mut(), &mut context),
            Poll::Ready(())
        ));

        let metrics = reactor.metrics();
        assert_eq!(metrics.tasks_executed.load(Ordering::Relaxed), 2);
    }
}
