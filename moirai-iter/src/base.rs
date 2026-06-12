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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(crate) inner: I,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(crate) inner: I,
    #[allow(dead_code)]
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
}

/// Common adapter for filter operations.
pub struct FilterAdapter<I, F, T> {
    #[allow(dead_code)]
    pub(crate) inner: I,
    #[allow(dead_code)]
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
}

/// Common adapter for batching operations.
pub struct BatchAdapter<I> {
    #[allow(dead_code)]
    pub(crate) inner: I,
    #[allow(dead_code)]
    pub(crate) size: usize,
}

impl<I> BatchAdapter<I> {
    pub fn new(inner: I, size: usize) -> Self {
        Self {
            inner,
            size: size.max(1),
        }
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
/// This is a lightweight alternative to external crates like rayon.
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
    pub fn new(size: usize) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<ErasedThreadJob>();
        let receiver = Arc::new(std::sync::Mutex::new(receiver));

        let workers = (0..size)
            .map(|_| {
                let receiver = receiver.clone();
                std::thread::spawn(move || loop {
                    let job = {
                        let guard = receiver.lock().unwrap();
                        match guard.recv() {
                            Ok(job) => job,
                            Err(_) => break,
                        }
                    };
                    job.run();
                })
            })
            .collect();

        Self {
            sender: Some(sender),
            workers,
        }
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
mod tests {
    use super::*;

    #[test]
    fn test_tree_reduce() {
        let items = vec![1, 2, 3, 4, 5];
        let result = tree_reduce(items, |a, b| a + b);
        assert_eq!(result, Some(15));

        let empty: Vec<i32> = vec![];
        let result = tree_reduce(empty, |a, b| a + b);
        assert_eq!(result, None);
    }

    #[test]
    fn test_process_in_batches() {
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = process_in_batches(items, 3, |chunk| vec![chunk.iter().sum::<i32>()]);
        // [1,2,3] = 6, [4,5,6] = 15, [7,8] = 15
        assert_eq!(result, vec![6, 15, 15]);
    }

    #[test]
    fn pool_fallback_only_on_pre_execution_shutdown() {
        use moirai_core::error::ExecutorError;
        assert!(!pool_fallback_permitted(&Ok(())));
        assert!(pool_fallback_permitted(&Err(ExecutorError::ShuttingDown)));
    }

    #[test]
    #[should_panic(expected = "partial execution")]
    fn pool_fallback_rejects_partial_execution_errors() {
        use moirai_core::error::ExecutorError;
        let _ = pool_fallback_permitted(&Err(ExecutorError::SpawnFailed(
            moirai_core::error::TaskError::Panicked,
        )));
    }

    #[test]
    fn test_tree_reduce_parallel() {
        let items: Vec<i32> = (1..=1000).collect();
        let result = tree_reduce(items, |a, b| a + b);
        assert_eq!(result, Some(500500));
    }

    #[test]
    fn test_thread_pool_graceful_shutdown() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        {
            let pool = ThreadPool::new(2);

            // Submit fewer, faster tasks
            for _ in 0..4 {
                let counter = counter.clone();
                pool.execute(move || {
                    // Fast operation without sleep
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }

            // Allow tasks to complete (reduced wait time)
            for _ in 0..10 {
                if counter.load(Ordering::SeqCst) == 4 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        } // Pool is dropped here

        // Verify all tasks completed
        assert_eq!(counter_clone.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_erased_thread_job_runs_once() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let counter = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&counter);
        let job = ErasedThreadJob::new(move || {
            observed.fetch_add(1, Ordering::SeqCst);
        });

        job.run();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_erased_thread_job_drops_unrun_capture() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct DropCounter(Arc<AtomicUsize>);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let captured = DropCounter(Arc::clone(&drops));
        let job = ErasedThreadJob::new(move || drop(captured));

        drop(job);

        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
