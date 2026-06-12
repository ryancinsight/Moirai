//! Synchronous data-parallel primitives — Moirai's rayon-replacement surface.
//!
//! This crate is the **parallel** domain (throughput over data), distinct from
//! the **concurrent** domain (`moirai-async`, async tasks/IO). All operations
//! here are fully synchronous (no `async`, no `.await`), so they are safe inside
//! pure compute kernels without async contagion, and operate on borrowed slices
//! with in-place mutation (zero-copy).
//!
//! # Selecting an execution strategy
//!
//! Strategy is a zero-sized [`ExecutionPolicy`] type ([`Sequential`],
//! [`Parallel`], [`Adaptive`]) chosen at compile time, so every form below
//! monomorphizes with no dynamic dispatch:
//!
//! - **Extension traits** (the surface) — `slice.par()` / `slice.par_mut()`
//!   return [`Adaptive`] handles, then `for_each` / `enumerate` / `map_collect` /
//!   `map_reduce`:
//!   ```
//!   use moirai_parallel::{ParallelSlice, ParallelSliceMut};
//!   let v: Vec<u64> = (0..1000).collect();
//!   let sum = v.par().map_reduce(0, |&x| x, |a, b| a + b);   // auto-routes
//!   let mut m = v.clone();
//!   m.par_mut().for_each(|x| *x += 1);
//!   ```
//! - **`*_with::<P>` free functions** — a low-level override that pins the policy
//!   via turbofish (`for_each_with::<Sequential>(&data, f)`), for the rare case
//!   that needs to force sequential (determinism / nested regions) or parallel.
//!   Most code should just use `.par()`.
//!
//! Because [`Adaptive`] is itself a zero-sized policy, `.par()` is a fully
//! monomorphized, zero-cost abstraction that parallelizes only at or above
//! [`ADAPTIVE_PARALLEL_THRESHOLD`] and runs sequentially below it — the
//! parallel/sequential decision is automatic, with nothing to designate.
//!
//! These data-parallel ops are synchronous (they return values, not futures),
//! but they run on the **same unified hybrid scheduler** as async work
//! ([`moirai_executor::global`]) — not a separate pool. A `.par()` worker task
//! can therefore spawn or drive async work (`moirai::global().spawn_async`/
//! `block_on`) on that same runtime, so parallel processing and asynchronous
//! tasks compose within one process. The sync return shape here is a property of
//! the *operation*, not an isolation boundary.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod policy;

pub use policy::{
    Adaptive, AdaptiveWithThreshold, ExecutionPolicy, Parallel, Sequential,
    ADAPTIVE_PARALLEL_THRESHOLD,
};

use core::marker::PhantomData;

/// Pointer wrapper used to hand disjoint `&mut` sub-slices to worker tasks.
///
/// The `Send`/`Sync` impls are sound only because the `*_mut` operations assign
/// each task a non-overlapping index range, so the pointer is never used to form
/// aliasing references.
pub(crate) struct DisjointMutPtr<T>(pub(crate) *mut T);

// SAFETY: callers dereference pairwise-disjoint ranges only, so the pointer
// never forms aliasing `&mut` references; `T: Send` permits moving element
// access across worker threads.
unsafe impl<T: Send> Send for DisjointMutPtr<T> {}
unsafe impl<T: Send> Sync for DisjointMutPtr<T> {}

impl<T> DisjointMutPtr<T> {
    /// Return a `&mut` to element `i`.
    ///
    /// # Safety
    /// `i` must be in bounds and visited at most once across all concurrent
    /// tasks, so the returned reference never aliases another.
    #[inline]
    pub(crate) unsafe fn get_mut<'a>(&self, i: usize) -> &'a mut T {
        // SAFETY: guaranteed by the caller's per-index-once contract.
        unsafe { &mut *self.0.add(i) }
    }

    /// Return the wrapped base pointer. Taking `&self` forces a closure to
    /// capture the whole (`Send`/`Sync`) wrapper rather than the bare `*mut T`
    /// field under 2021 disjoint capture.
    #[inline]
    pub(crate) fn base(&self) -> *mut T {
        self.0
    }
}
/// Synchronous data-parallel operators and free functions.
pub mod ops;
pub use ops::{
    enumerate_mut_with, enumerate_with, fold_reduce_with, for_each_chunk_mut_enumerated_with,
    for_each_chunk_mut_with, for_each_chunk_pair_mut_enumerated_with, for_each_index_with,
    for_each_mut_with, for_each_with, map_collect_index_with, map_collect_mut_with,
    map_collect_with, map_reduce_with, reduce_index_with,
};

// ---------------------------------------------------------------------------
// Extension traits: trait-based, type-selected parallel views over slices
// ---------------------------------------------------------------------------

/// A read-only parallel view of a slice bound to execution policy `P`.
///
/// Construct via [`ParallelSlice::par`]. Zero-sized beyond the borrowed slice.
pub struct ParRef<'a, T, P> {
    data: &'a [T],
    _policy: PhantomData<P>,
}

impl<'a, T, P: ExecutionPolicy> ParRef<'a, T, P> {
    /// Apply `f` to every element. See [`for_each_with`].
    pub fn for_each<F: Fn(&T) + Send + Sync>(self, f: F)
    where
        T: Sync,
    {
        for_each_with::<P, _, _>(self.data, f);
    }

    /// Apply `f(index, &element)` to every element. See [`enumerate_with`].
    pub fn enumerate<F: Fn(usize, &T) + Send + Sync>(self, f: F)
    where
        T: Sync,
    {
        enumerate_with::<P, _, _>(self.data, f);
    }

    /// Map then collect into a `Vec<R>` in order. See [`map_collect_with`].
    pub fn map_collect<R: Send, F: Fn(&T) -> R + Send + Sync>(self, f: F) -> Vec<R>
    where
        T: Sync,
    {
        map_collect_with::<P, _, _, _>(self.data, f)
    }

    /// Map-reduce. See [`map_reduce_with`].
    pub fn map_reduce<R, M, Rd>(self, identity: R, map: M, reduce: Rd) -> R
    where
        T: Sync,
        R: Send + Sync + Clone,
        M: Fn(&T) -> R + Send + Sync,
        Rd: Fn(R, R) -> R + Send + Sync,
    {
        map_reduce_with::<P, _, _, _, _>(self.data, identity, map, reduce)
    }
}

/// A mutable parallel view of a slice bound to execution policy `P`.
pub struct ParMut<'a, T, P> {
    data: &'a mut [T],
    _policy: PhantomData<P>,
}

impl<'a, T, P: ExecutionPolicy> ParMut<'a, T, P> {
    /// Apply `f` to every element in place. See [`for_each_mut_with`].
    pub fn for_each<F: Fn(&mut T) + Send + Sync>(self, f: F)
    where
        T: Send,
    {
        for_each_mut_with::<P, _, _>(self.data, f);
    }

    /// Apply `f(index, &mut element)` to every element in place.
    pub fn enumerate<F: Fn(usize, &mut T) + Send + Sync>(self, f: F)
    where
        T: Send,
    {
        enumerate_mut_with::<P, _, _>(self.data, f);
    }
}

/// Extension trait providing an adaptive parallel view over `&[T]`.
pub trait ParallelSlice<T> {
    /// Adaptive, auto-routing parallel view (the everyday entry point).
    fn par(&self) -> ParRef<'_, T, Adaptive>;
}

impl<T> ParallelSlice<T> for [T] {
    #[inline]
    fn par(&self) -> ParRef<'_, T, Adaptive> {
        ParRef {
            data: self,
            _policy: PhantomData,
        }
    }
}

/// Extension trait providing an adaptive mutable parallel view over `&mut [T]`.
pub trait ParallelSliceMut<T> {
    /// Adaptive, auto-routing mutable parallel view (the everyday entry point).
    fn par_mut(&mut self) -> ParMut<'_, T, Adaptive>;
}

impl<T> ParallelSliceMut<T> for [T] {
    #[inline]
    fn par_mut(&mut self) -> ParMut<'_, T, Adaptive> {
        ParMut {
            data: self,
            _policy: PhantomData,
        }
    }
}

#[cfg(feature = "melinoe")]
pub mod melinoe_ext;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
