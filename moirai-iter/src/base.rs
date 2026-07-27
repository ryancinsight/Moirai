//! Base traits and utilities for Moirai iterators.
//!
//! This module provides the foundational abstractions that reduce code duplication
//! across different iterator implementations, following DRY and SOLID principles.

use std::marker::PhantomData;
use std::sync::Arc;

/// Decides whether a failed global-executor indexed fan-out may be re-run on
/// the caller's thread.
///
/// Only `ShuttingDown` is returned before any chunk closure runs, so it is
/// the only error for which a retry cannot duplicate caller side effects.
/// Every other error (a chunk panic, a mid-loop spawn failure) is reported
/// after `state.wait()` — some chunks have already executed, and re-running
/// the full index domain would apply the caller's closure twice to those
/// items. That violated invariant is unrecoverable here, so it propagates as
/// a panic (matching rayon's panic-propagation semantics).
///
/// Admission backpressure never reaches this decision: the scheduler runs a
/// chunk its queue rejects on the submitting lane (ADR-022, ISSUE-221).
pub(crate) fn sequential_fallback_permitted(
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
