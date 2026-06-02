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
//! This crate is sync-only by design: async (I/O concurrency) is a different
//! operation shape handled at the task level by `moirai-async` / the executor's
//! `spawn` family, not a data-parallel policy.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod policy;

pub use policy::{Adaptive, ExecutionPolicy, Parallel, Sequential, ADAPTIVE_PARALLEL_THRESHOLD};

use core::marker::PhantomData;
use moirai_executor::{global, BlockingTask};

/// Pointer wrapper used to hand disjoint `&mut` sub-slices to worker tasks.
///
/// The `Send`/`Sync` impls are sound only because the `*_mut` operations assign
/// each task a non-overlapping index range, so the pointer is never used to form
/// aliasing references.
struct DisjointMutPtr<T>(*mut T);

// SAFETY: callers dereference pairwise-disjoint ranges only, so the pointer
// never forms aliasing `&mut` references; `T: Send` permits moving element
// access across worker threads.
unsafe impl<T: Send> Send for DisjointMutPtr<T> {}
unsafe impl<T: Send> Sync for DisjointMutPtr<T> {}

impl<T> DisjointMutPtr<T> {
    #[inline]
    fn base(&self) -> *mut T {
        self.0
    }
}

/// Compute `(number_of_chunks, chunk_len)` for splitting `len` items across the
/// runtime's worker threads. Never returns zero chunks for non-empty input.
#[inline]
fn chunk_layout(len: usize) -> (usize, usize) {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let chunks = workers.min(len).max(1);
    (chunks, len.div_ceil(chunks))
}

/// Apply `f` to every element of `data`, scheduled by policy `P`.
pub fn for_each_with<P, T, F>(data: &[T], f: F)
where
    P: ExecutionPolicy,
    T: Sync,
    F: Fn(&T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    let (chunks, chunk) = chunk_layout(n);
    if !P::parallelize(n) || chunks <= 1 {
        data.iter().for_each(f);
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<BlockingTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            for item in &data[start..end] {
                f(item);
            }
        })
        .expect("moirai global executor: for_each_with");
}

/// Apply `f` to every element of `data` in place, scheduled by policy `P`.
pub fn for_each_mut_with<P, T, F>(data: &mut [T], f: F)
where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(&mut T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    let (chunks, chunk) = chunk_layout(n);
    if !P::parallelize(n) || chunks <= 1 {
        data.iter_mut().for_each(f);
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<BlockingTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: ranges `[start, end)` are pairwise disjoint across `ci`, so
            // the `&mut` sub-slices never alias; `data` is mutably borrowed for
            // the whole call and `for_each_indexed` joins every task before
            // returning, so `base` stays valid.
            let slice =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            for item in slice {
                f(item);
            }
        })
        .expect("moirai global executor: for_each_mut_with");
}

/// Apply `f(index, &element)` to every element of `data`, scheduled by policy `P`.
pub fn enumerate_with<P, T, F>(data: &[T], f: F)
where
    P: ExecutionPolicy,
    T: Sync,
    F: Fn(usize, &T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    let (chunks, chunk) = chunk_layout(n);
    if !P::parallelize(n) || chunks <= 1 {
        data.iter().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<BlockingTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            for (offset, item) in data[start..end].iter().enumerate() {
                f(start + offset, item);
            }
        })
        .expect("moirai global executor: enumerate_with");
}

/// Apply `f(index, &mut element)` to every element of `data` in place,
/// scheduled by policy `P`.
pub fn enumerate_mut_with<P, T, F>(data: &mut [T], f: F)
where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    let (chunks, chunk) = chunk_layout(n);
    if !P::parallelize(n) || chunks <= 1 {
        data.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<BlockingTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: disjoint ranges; see `for_each_mut_with`.
            let slice =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            for (offset, item) in slice.iter_mut().enumerate() {
                f(start + offset, item);
            }
        })
        .expect("moirai global executor: enumerate_mut_with");
}

/// Map each element of `data` with `f`, collecting into a `Vec<R>` in order,
/// scheduled by policy `P`.
pub fn map_collect_with<P, T, R, F>(data: &[T], f: F) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Send + Sync,
{
    let n = data.len();
    if !P::parallelize(n) {
        return data.iter().map(f).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: capacity is `n`; every slot is written exactly once below before
    // being read, and `MaybeUninit` makes `set_len` sound without initialization.
    unsafe {
        out.set_len(n);
    }
    enumerate_mut_with::<Parallel, _, _>(&mut out, |i, slot| {
        slot.write(f(&data[i]));
    });
    // SAFETY: every slot initialized above; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), n, out.capacity()) }
}

/// Map-reduce over `data`, scheduled by policy `P`.
///
/// `reduce` must be associative and `identity` its neutral element, since chunk
/// boundaries and combination order are unspecified.
pub fn map_reduce_with<P, T, R, M, Rd>(data: &[T], identity: R, map: M, reduce: Rd) -> R
where
    P: ExecutionPolicy,
    T: Sync,
    R: Send + Sync + Clone,
    M: Fn(&T) -> R + Send + Sync,
    Rd: Fn(R, R) -> R + Send + Sync,
{
    let n = data.len();
    let (chunks, chunk) = if n == 0 { (0, 0) } else { chunk_layout(n) };
    if n == 0 || !P::parallelize(n) || chunks <= 1 {
        let mut acc = identity;
        for item in data {
            acc = reduce(acc, map(item));
        }
        return acc;
    }
    let map = &map;
    let reduce = &reduce;
    let identity_for_map = identity.clone();
    global()
        .map_reduce_indexed::<BlockingTask, _, _, _>(
            chunks,
            identity,
            move |ci| {
                let start = ci * chunk;
                let mut acc = identity_for_map.clone();
                if start < n {
                    let end = (start + chunk).min(n);
                    for item in &data[start..end] {
                        acc = reduce(acc, map(item));
                    }
                }
                acc
            },
            move |a, b| reduce(a, b),
        )
        .expect("moirai global executor: map_reduce_with")
}

// ---------------------------------------------------------------------------
// Extension traits: trait-based, type-selected parallel views over slices
// ---------------------------------------------------------------------------

/// A read-only parallel view of a slice bound to execution policy `P`.
///
/// Construct via [`ParallelSlice::par`] (adaptive) or
/// [`ParallelSlice::par_with`] (`::<P>`). Zero-sized beyond the borrowed slice.
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
        ParRef { data: self, _policy: PhantomData }
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
        ParMut { data: self, _policy: PhantomData }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn for_each_visits_every_element_once() {
        let data: Vec<usize> = (0..10_000).collect();
        let counter = AtomicUsize::new(0);
        data.par().for_each(|&x| {
            counter.fetch_add(x, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), data.iter().sum());
    }

    #[test]
    fn for_each_mut_mutates_in_place() {
        let mut data: Vec<u64> = (0..10_000).collect();
        data.par_mut().for_each(|x| *x *= 2);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, (i as u64) * 2);
        }
    }

    #[test]
    fn enumerate_mut_uses_index() {
        let mut data = vec![0usize; 5_000];
        data.par_mut().enumerate(|i, x| *x = i * 3);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, i * 3);
        }
    }

    #[test]
    fn map_collect_preserves_order() {
        let data: Vec<u64> = (0..20_000).collect();
        let squared = data.par().map_collect(|&x| x * x);
        for (i, &v) in squared.iter().enumerate() {
            assert_eq!(v, (i as u64) * (i as u64));
        }
    }

    #[test]
    fn map_reduce_sums_correctly() {
        let data: Vec<u64> = (0..100_000).collect();
        assert_eq!(
            data.par().map_reduce(0u64, |&x| x, |a, b| a + b),
            data.iter().copied().sum::<u64>()
        );
        // free-function form with explicit policy
        assert_eq!(
            map_reduce_with::<Sequential, _, _, _, _>(&data, 0u64, |&x| x, |a, b| a + b),
            data.iter().copied().sum::<u64>()
        );
    }

    #[test]
    fn adaptive_view_and_explicit_policies_agree() {
        let data: Vec<u64> = (0..50_000).collect();
        let expected: u64 = data.iter().sum();
        // adaptive trait surface
        assert_eq!(data.par().map_reduce(0u64, |&x| x, |a, b| a + b), expected);
        // explicit policy overrides via the low-level free functions
        assert_eq!(
            map_reduce_with::<Parallel, _, _, _, _>(&data, 0u64, |&x| x, |a, b| a + b),
            expected
        );
        assert_eq!(
            map_reduce_with::<Sequential, _, _, _, _>(&data, 0u64, |&x| x, |a, b| a + b),
            expected
        );
        let doubled = data.par().map_collect(|&x| x * 2);
        assert_eq!(doubled, data.iter().map(|&x| x * 2).collect::<Vec<_>>());
    }

    #[test]
    fn mut_view_and_explicit_sequential_agree() {
        let mut data: Vec<u64> = (0..50_000).collect();
        data.par_mut().for_each(|x| *x += 1);
        // forced-sequential override produces the same result
        enumerate_mut_with::<Sequential, _, _>(&mut data, |i, x| *x += i as u64);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, i as u64 + 1 + i as u64);
        }
    }

    #[test]
    fn empty_and_single_inputs_are_handled() {
        let empty: Vec<i32> = Vec::new();
        empty.par().for_each(|_| panic!("must not run"));
        assert_eq!(empty.par().map_reduce(42i64, |&x| x as i64, |a, b| a + b), 42);
        let mut one = vec![7u64];
        one.par_mut().for_each(|x| *x += 1);
        assert_eq!(one, vec![8]);
    }
}
