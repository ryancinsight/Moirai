//! Base traits and utilities for Moirai iterators.
//!
//! This module provides the foundational abstractions that reduce code duplication
//! across different iterator implementations, following DRY and SOLID principles.

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;

/// Decides whether a failed global-executor indexed fan-out may be retried
/// on the shared thread pool.
///
/// Only `ShuttingDown` is returned before any chunk closure runs, so it is
/// the only error for which a retry cannot duplicate caller side effects.
/// Every other error (a chunk panic, a mid-loop spawn failure) is reported
/// after `state.wait()` — some chunks have already executed, and re-running
/// the full index domain would apply the caller's closure twice to those
/// items. That violated invariant is unrecoverable here, so it propagates as
/// a panic (matching rayon's panic-propagation semantics).
pub(crate) fn pool_fallback_permitted(
    fan_out: &Result<(), moirai_core::error::ExecutorError>,
) -> bool {
    match fan_out {
        Ok(()) => false,
        Err(moirai_core::error::ExecutorError::ShuttingDown) => true,
        Err(error) => panic!(
            "invariant: indexed fan-out failed after partial execution ({error}); \
             retrying would duplicate caller side effects"
        ),
    }
}

/// A pointer wrapper shared across worker threads for zero-copy fan-out.
///
/// # Safety
/// The pointer must remain valid for the lifetime of the SendPtr.
/// `Send`/`Sync` only assert the *pointer value* may move or be shared across
/// threads; every dereference site is `unsafe` and owns the proof that the
/// accessed region is disjoint per worker (chunked indices) or read-only.
#[derive(Debug)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);

impl<T> Clone for SendPtr<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendPtr<T> {}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Get the raw pointer.
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid and properly synchronized.
    pub(crate) unsafe fn as_ptr(&self) -> *mut T {
        self.0
    }
}

/// Efficient tree reduction algorithm that works across all execution contexts.
/// This reduces O(n) sequential operations to O(log n) parallel operations.
pub fn tree_reduce<T, F>(mut items: Vec<T>, func: F) -> Option<T>
where
    T: Send + Clone,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    if items.is_empty() {
        return None;
    }

    while items.len() > 1 {
        let mut next = Vec::with_capacity(items.len().div_ceil(2));

        for chunk in items.chunks(2) {
            if chunk.len() == 2 {
                next.push(func(chunk[0].clone(), chunk[1].clone()));
            } else {
                next.push(chunk[0].clone());
            }
        }

        items = next;
    }

    items.into_iter().next()
}

/// Batch processing for improved cache locality.
/// This is used across different execution contexts for better performance.
pub fn process_in_batches<T, R, F>(items: Vec<T>, batch_size: usize, func: F) -> Vec<R>
where
    T: Send + Clone,
    R: Send,
    F: Fn(&[T]) -> Vec<R> + Send + Sync,
{
    items.chunks(batch_size).flat_map(func).collect()
}

/// Base iterator wrapper that provides common functionality.
/// This follows the Decorator pattern to add behavior without modifying the original iterator.
pub struct BaseIterator<I, C> {
    pub(crate) inner: I,
    pub(crate) context: Arc<C>,
}

impl<I, C> BaseIterator<I, C> {
    pub fn new(inner: I, context: C) -> Self {
        Self {
            inner,
            context: Arc::new(context),
        }
    }

    pub fn with_context(inner: I, context: Arc<C>) -> Self {
        Self { inner, context }
    }

    /// Borrow the wrapped iterator.
    #[must_use]
    pub const fn inner(&self) -> &I {
        &self.inner
    }

    /// Borrow the shared execution context.
    #[must_use]
    pub fn context(&self) -> &Arc<C> {
        &self.context
    }

    /// Consume the wrapper and return its components without cloning.
    #[must_use]
    pub fn into_parts(self) -> (I, Arc<C>) {
        (self.inner, self.context)
    }
}

/// Trait for types that can be collected from Moirai iterators.
/// This allows for optimized collection strategies based on the target type.
pub trait FromMoiraiIterator<T>: Send {
    /// Create a collection from an iterator's items.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self;

    /// Create a collection with a size hint for pre-allocation.
    fn from_iter_with_hint<I: IntoIterator<Item = T>>(iter: I, size_hint: usize) -> Self
    where
        Self: Sized,
    {
        let _ = size_hint; // Default ignores hint
        Self::from_iter(iter)
    }
}

impl<T: Send> FromMoiraiIterator<T> for Vec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect()
    }

    fn from_iter_with_hint<I: IntoIterator<Item = T>>(iter: I, size_hint: usize) -> Self {
        let mut vec = Vec::with_capacity(size_hint);
        vec.extend(iter);
        vec
    }
}

/// Common adapter for mapping operations.
/// This reduces duplication across different iterator types.
pub struct MapAdapter<I, F, T, R> {
    pub(crate) inner: I,
    pub(crate) func: F,
    pub(crate) _phantom: PhantomData<(T, R)>,
}

impl<I, F, T, R> MapAdapter<I, F, T, R> {
    pub fn new(inner: I, func: F) -> Self {
        Self {
            inner,
            func,
            _phantom: PhantomData,
        }
    }

    /// Borrow the wrapped iterator.
    #[must_use]
    pub const fn inner(&self) -> &I {
        &self.inner
    }

    /// Borrow the map function.
    #[must_use]
    pub const fn function(&self) -> &F {
        &self.func
    }

    /// Consume the adapter and return its components without cloning.
    #[must_use]
    pub fn into_parts(self) -> (I, F) {
        (self.inner, self.func)
    }
}

/// Common adapter for filter operations.
pub struct FilterAdapter<I, F, T> {
    pub(crate) inner: I,
    pub(crate) predicate: F,
    pub(crate) _phantom: PhantomData<T>,
}

impl<I, F, T> FilterAdapter<I, F, T> {
    pub fn new(inner: I, predicate: F) -> Self {
        Self {
            inner,
            predicate,
            _phantom: PhantomData,
        }
    }

    /// Borrow the wrapped iterator.
    #[must_use]
    pub const fn inner(&self) -> &I {
        &self.inner
    }

    /// Borrow the predicate.
    #[must_use]
    pub const fn predicate(&self) -> &F {
        &self.predicate
    }

    /// Consume the adapter and return its components without cloning.
    #[must_use]
    pub fn into_parts(self) -> (I, F) {
        (self.inner, self.predicate)
    }
}

/// Common adapter for batching operations.
pub struct BatchAdapter<I> {
    pub(crate) inner: I,
    pub(crate) size: usize,
}

impl<I> BatchAdapter<I> {
    pub fn new(inner: I, size: usize) -> Self {
        Self {
            inner,
            size: size.max(1),
        }
    }

    /// Borrow the wrapped iterator.
    #[must_use]
    pub const fn inner(&self) -> &I {
        &self.inner
    }

    /// Return the normalized batch size.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Consume the adapter and return its components without cloning.
    #[must_use]
    pub fn into_parts(self) -> (I, usize) {
        (self.inner, self.size)
    }
}

/// Shared thread pool for parallel execution.
/// This follows the Singleton pattern to avoid creating multiple thread pools.
use std::sync::OnceLock;
static SHARED_THREAD_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

pub fn get_shared_thread_pool() -> Arc<ThreadPool> {
    SHARED_THREAD_POOL
        .get_or_init(|| {
            let num_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            Arc::new(ThreadPool::new(num_threads))
        })
        .clone()
}

/// Simple thread pool implementation.
///
/// Flat fan-out only: jobs must not block on other jobs of this pool. The pool
/// is a FIFO queue over a fixed worker set with no work stealing, so a worker
/// waiting for a queued job cannot run it, and once every worker waits that way
/// the queue stalls permanently. Nested and recursive fork-join belongs on the
/// scheduler scope instead, whose waiters run queued work while they wait
/// (ADR-022).
pub struct ThreadPool {
    /// Wrapped in Option so `Drop` can close the channel before joining workers.
    sender: Option<std::sync::mpsc::Sender<ErasedThreadJob>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// Heap-stable thread-pool job with monomorphized run/drop functions.
struct ErasedThreadJob {
    ptr: NonNull<()>,
    run: unsafe fn(NonNull<()>),
    drop: unsafe fn(NonNull<()>),
    consumed: bool,
}

// Safety: `ErasedThreadJob` owns a `Send + 'static` job allocation created by
// `ErasedThreadJob::new`. Moving the erased owner between threads transfers
// ownership of that allocation.
unsafe impl Send for ErasedThreadJob {}

impl ErasedThreadJob {
    fn new<F>(job: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        let ptr = Box::into_raw(Box::new(job)).cast::<()>();
        Self {
            ptr: NonNull::new(ptr).expect("Box::into_raw never returns null"),
            run: run_thread_job::<F>,
            drop: drop_thread_job::<F>,
            consumed: false,
        }
    }

    fn run(mut self) {
        self.consumed = true;
        // Safety: `ptr` was created by `new` for the same concrete job type as
        // the monomorphized run function stored beside it.
        unsafe {
            (self.run)(self.ptr);
        }
    }
}

impl Drop for ErasedThreadJob {
    fn drop(&mut self) {
        if !self.consumed {
            // Safety: unconsumed jobs still own the allocation created in `new`.
            unsafe {
                (self.drop)(self.ptr);
            }
        }
    }
}

unsafe fn run_thread_job<F>(ptr: NonNull<()>)
where
    F: FnOnce() + Send + 'static,
{
    // Safety: `ErasedThreadJob::new::<F>` created the allocation. Moving the
    // job out of the box consumes the allocation and executes the closure once.
    let job = unsafe { *Box::from_raw(ptr.cast::<F>().as_ptr()) };
    job();
}

unsafe fn drop_thread_job<F>(ptr: NonNull<()>)
where
    F: FnOnce() + Send + 'static,
{
    // Safety: the allocation was created by `ErasedThreadJob::new::<F>` and is
    // reconstructed exactly once for an unexecuted job.
    unsafe {
        drop(Box::from_raw(ptr.cast::<F>().as_ptr()));
    }
}

impl std::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadPool")
            .field("workers", &self.workers.len())
            .finish()
    }
}

impl ThreadPool {
    /// Build a pool with `size` worker threads, always at least one.
    ///
    /// A pool with no workers accepts jobs and never runs them — `execute`
    /// queues onto a channel nobody receives from — so anything that waits on
    /// those jobs waits forever. One worker is the smallest pool that can make
    /// progress, matching how `BatchAdapter` clamps its batch size.
    pub fn new(size: usize) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<ErasedThreadJob>();
        let receiver = Arc::new(std::sync::Mutex::new(receiver));

        let workers: Vec<std::thread::JoinHandle<()>> = (0..size.max(1))
            .map(|_| {
                let receiver = receiver.clone();
                std::thread::spawn(move || loop {
                    let job = {
                        // Scoped so the guard is released before running the
                        // job: holding it across `run` would serialize the pool
                        // onto one worker.
                        let guard = receiver.lock().unwrap();
                        match guard.recv() {
                            Ok(job) => job,
                            Err(_) => break,
                        }
                    };
                    // A job that panics must not take the worker down with it.
                    // Workers are never replaced, so an unwinding job would
                    // shrink the pool permanently, and once every worker had
                    // died `execute` would queue jobs that nobody ever runs —
                    // turning a caller's panic into a later, unrelated hang.
                    //
                    // Swallowing the payload here loses nothing: the panicking
                    // job drops its completion sender while unwinding, so the
                    // shortfall still reaches the caller through
                    // `PoolJoinGuard::wait`. `AssertUnwindSafe` is honest
                    // because the worker holds no invariant across the call —
                    // it owns the job outright and touches nothing else.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run()));
                })
            })
            .collect();

        Self {
            sender: Some(sender),
            workers,
        }
    }

    /// Number of worker threads, always at least one.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn execute<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(s) = &self.sender {
            let _ = s.send(ErasedThreadJob::new(job));
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Close the sending end BEFORE joining worker threads.
        // Worker threads loop on `recv()`, which returns `Err` only when every
        // Sender clone is dropped.  Rust drops struct fields AFTER the `Drop`
        // impl returns, so without this explicit take() the join loop would
        // deadlock: join waits for the thread to exit, the thread waits for the
        // channel to close, the channel never closes while `self.sender` lives.
        drop(self.sender.take());

        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// A guard that waits for all spawned tasks in a channel to complete when dropped.
/// This prevents use-after-free when unwinding panics.
pub(crate) struct PoolJoinGuard {
    rx: std::sync::mpsc::Receiver<()>,
    count: usize,
}

impl PoolJoinGuard {
    pub(crate) fn new(rx: std::sync::mpsc::Receiver<()>, count: usize) -> Self {
        Self { rx, count }
    }

    /// Wait for all tasks to finish, panicking if any did not.
    ///
    /// Each task sends `()` as its last act, so a completion message means that
    /// task ran to the end. A task that panics instead unwinds without sending
    /// and drops its sender — the pool's worker loop does not catch unwinds — so
    /// once every sender is gone `recv` returns `Err` immediately and keeps
    /// doing so. Counting the successes is what separates "all finished" from
    /// "the channel disconnected early"; discarding the `Result` makes the two
    /// indistinguishable and lets a caller proceed as if the work were done.
    ///
    /// That distinction is load-bearing: `ZeroCopyParallelIter::map` fills a
    /// `Vec<MaybeUninit<R>>` from these tasks and calls `assume_init` on every
    /// element afterwards, so returning normally when a chunk never wrote its
    /// slice would read uninitialized memory. The other callers lose work rather
    /// than soundness, but silently returning a partial result is its own defect.
    ///
    /// Panicking here surfaces the worker's failure on the caller's thread. The
    /// `MaybeUninit` buffer is then dropped without dropping its elements — the
    /// initialized results leak, which is the same trade the executor path makes
    /// when it reports a failed fan-out.
    pub(crate) fn wait(mut self) {
        let expected = self.count;
        let mut completed = 0;
        for _ in 0..expected {
            if self.rx.recv().is_ok() {
                completed += 1;
            }
        }
        self.count = 0; // Prevent waiting again in drop
        assert_eq!(
            completed,
            expected,
            "invariant: {} of {expected} pooled tasks did not report completion; \
             a worker panicked and its output was never written",
            expected - completed
        );
    }
}

impl Drop for PoolJoinGuard {
    fn drop(&mut self) {
        // Best-effort drain only: `drop` may run while a panic is already
        // unwinding, and panicking again there aborts the process. `wait` is the
        // checked path; this exists so an early return still blocks until the
        // workers stop touching the borrowed data.
        for _ in 0..self.count {
            let _ = self.rx.recv();
        }
    }
}

/// Performance metrics for adaptive execution.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_items: usize,
    pub execution_time_ns: u64,
    pub memory_used_bytes: usize,
    pub strategy_used: String,
}

impl PerformanceMetrics {
    pub fn throughput_per_sec(&self) -> f64 {
        if self.execution_time_ns == 0 {
            0.0
        } else {
            (self.total_items as f64 * 1_000_000_000.0) / self.execution_time_ns as f64
        }
    }
}

// Window and chunk iterators have been consolidated in `windows.rs` to enforce SSOT.
// Refer to `moirai_iter::windows` for `Windows`, `WindowsMut`, `Chunks`, `ChunksMut`, and `ChunksExact`.

#[cfg(test)]
#[path = "base/tests.rs"]
mod tests;
