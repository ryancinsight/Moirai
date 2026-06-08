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
use moirai_executor::{global, SyncTask};

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

// The executor's `for_each_indexed`/`map_reduce_indexed` already split the index
// domain `0..n` into worker-sized chunks and run them on the shared pool, so
// these wrappers pass the full element count `n` (one index per element) and let
// the scheduler chunk — pre-chunking here would defeat the reduce heuristic and
// serialize. `SyncTask` is the CPU-compute work class.

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
    if !P::parallelize(n) {
        data.iter().for_each(f);
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| f(&data[i]))
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
    if !P::parallelize(n) {
        data.iter_mut().for_each(f);
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| {
            // SAFETY: the scheduler visits each index in `0..n` exactly once
            // across disjoint chunks, so no two tasks alias element `i`; `data`
            // is borrowed mutably for the whole joined call.
            f(unsafe { base.get_mut(i) });
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
    if !P::parallelize(n) {
        data.iter().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| f(i, &data[i]))
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
    if !P::parallelize(n) {
        data.iter_mut().enumerate().for_each(|(i, x)| f(i, x));
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| {
            // SAFETY: each index in `0..n` is visited exactly once; see
            // `for_each_mut_with`.
            f(i, unsafe { base.get_mut(i) });
        })
        .expect("moirai global executor: enumerate_mut_with");
}

/// Apply `f` to every index in `0..len` in parallel, scheduled by policy `P`.
///
/// Synchronous equivalent of rayon's `(0..len).into_par_iter().for_each(f)`. Use
/// when the work is keyed by index and writes through external disjoint state
/// (atomics, per-index channels) rather than returning a value.
pub fn for_each_index_with<P, F>(len: usize, f: F)
where
    P: ExecutionPolicy,
    F: Fn(usize) + Send + Sync,
{
    if len == 0 {
        return;
    }
    if !P::parallelize(len) {
        (0..len).for_each(f);
        return;
    }
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(len, f)
        .expect("moirai global executor: for_each_index_with");
}

/// Apply `f` to each consecutive `chunk_size`-element mutable chunk of `data` in
/// parallel, scheduled by policy `P`. The final chunk may be shorter.
///
/// Synchronous equivalent of rayon's `data.par_chunks_mut(chunk_size).for_each(f)`
/// — the natural shape for batched/lane-wise transforms.
pub fn for_each_chunk_mut_with<P, T, F>(data: &mut [T], chunk_size: usize, f: F)
where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(&mut [T]) + Send + Sync,
{
    let n = data.len();
    if n == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = n.div_ceil(chunk_size);
    if !P::parallelize(n) || num_chunks <= 1 {
        data.chunks_mut(chunk_size).for_each(&f);
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            let start = c * chunk_size;
            if start >= n {
                return;
            }
            let end = (start + chunk_size).min(n);
            // SAFETY: the chunks `[start, end)` for distinct `c` are pairwise
            // disjoint and each is visited exactly once, so no two tasks alias.
            let chunk =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            f(chunk);
        })
        .expect("moirai global executor: for_each_chunk_mut_with");
}

/// Apply `f(index, a_chunk, b_chunk)` to paired `chunk_size`-element mutable
/// chunks of two **distinct** buffers in parallel, scheduled by policy `P`.
///
/// Synchronous equivalent of
/// `a.par_chunks_mut(n).zip(b.par_chunks_mut(n)).enumerate().for_each(f)`. The
/// number of chunks is derived from `a`; `b` is chunked identically, so callers
/// must ensure `b.len() >= a.len()` (typically equal). The two buffers must not
/// alias.
pub fn for_each_chunk_pair_mut_enumerated_with<P, A, B, F>(
    a: &mut [A],
    b: &mut [B],
    chunk_size: usize,
    f: F,
) where
    P: ExecutionPolicy,
    A: Send,
    B: Send,
    F: Fn(usize, &mut [A], &mut [B]) + Send + Sync,
{
    let na = a.len();
    let nb = b.len();
    if chunk_size == 0 || na == 0 {
        return;
    }
    let num_chunks = na.div_ceil(chunk_size);
    if !P::parallelize(na) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (ca, cb))| f(i, ca, cb));
        return;
    }
    let abase = DisjointMutPtr(a.as_mut_ptr());
    let bbase = DisjointMutPtr(b.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            let start = c * chunk_size;
            if start >= na || start >= nb {
                return;
            }
            let ea = (start + chunk_size).min(na);
            let eb = (start + chunk_size).min(nb);
            // SAFETY: chunks `[start, e*)` for distinct `c` are pairwise disjoint
            // within each buffer and each is visited once; `a` and `b` are
            // distinct, non-aliasing buffers, so the two references never alias.
            let ca =
                unsafe { core::slice::from_raw_parts_mut(abase.base().add(start), ea - start) };
            let cb =
                unsafe { core::slice::from_raw_parts_mut(bbase.base().add(start), eb - start) };
            f(c, ca, cb);
        })
        .expect("moirai global executor: for_each_chunk_pair_mut_enumerated_with");
}

/// Like [`for_each_chunk_mut_with`] but also passes the zero-based chunk index to
/// `f` (synchronous equivalent of
/// `data.par_chunks_mut(chunk_size).enumerate().for_each(f)`).
pub fn for_each_chunk_mut_enumerated_with<P, T, F>(data: &mut [T], chunk_size: usize, f: F)
where
    P: ExecutionPolicy,
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    let n = data.len();
    if n == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = n.div_ceil(chunk_size);
    if !P::parallelize(n) || num_chunks <= 1 {
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
        return;
    }
    let base = DisjointMutPtr(data.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |c| {
            let start = c * chunk_size;
            if start >= n {
                return;
            }
            let end = (start + chunk_size).min(n);
            // SAFETY: chunks `[start, end)` for distinct `c` are pairwise disjoint
            // and each visited exactly once, so no two tasks alias.
            let chunk =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            f(c, chunk);
        })
        .expect("moirai global executor: for_each_chunk_mut_enumerated_with");
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
    if n == 0 || !P::parallelize(n) {
        let mut acc = identity;
        for item in data {
            acc = reduce(acc, map(item));
        }
        return acc;
    }
    let map = &map;
    let reduce = &reduce;
    // The executor folds each worker chunk locally (seeded by `identity`) then
    // combines chunk results, so `map` is per-element and `reduce` per-pair.
    global()
        .map_reduce_indexed::<SyncTask, _, _, _>(n, identity, move |i| map(&data[i]), reduce)
        .expect("moirai global executor: map_reduce_with")
}

/// Parallel fold-reduce over the index domain `0..len`, scheduled by policy `P`.
///
/// Each worker chunk creates one accumulator with `init()`, folds its indices
/// into it with `fold`, and the per-chunk accumulators are combined with
/// `reduce`. Unlike [`reduce_index_with`], `fold` mutates a single accumulator
/// per chunk (no per-element temporary), which is the efficient shape for
/// accumulating into a collection — e.g. grouping entries into a `HashMap`.
/// `reduce` must be associative; `init()` must yield its neutral element.
pub fn fold_reduce_with<P, A, Init, Fold, Red>(len: usize, init: Init, fold: Fold, reduce: Red) -> A
where
    P: ExecutionPolicy,
    A: Send,
    Init: Fn() -> A + Send + Sync,
    Fold: Fn(A, usize) -> A + Send + Sync,
    Red: Fn(A, A) -> A,
{
    if len == 0 {
        return init();
    }
    if !P::parallelize(len) {
        let mut acc = init();
        for i in 0..len {
            acc = fold(acc, i);
        }
        return acc;
    }
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let chunks = workers.min(len).max(1);
    let chunk = len.div_ceil(chunks);
    let mut slots: Vec<Option<A>> = (0..chunks).map(|_| None).collect();
    let base = DisjointMutPtr(slots.as_mut_ptr());
    let init_ref = &init;
    let fold_ref = &fold;
    global()
        .for_each_indexed::<SyncTask, _>(chunks, move |ci| {
            let start = ci * chunk;
            if start >= len {
                return;
            }
            let end = (start + chunk).min(len);
            let mut acc = init_ref();
            for i in start..end {
                acc = fold_ref(acc, i);
            }
            // SAFETY: each `ci` writes its own slot exactly once; slots are
            // disjoint and `slots` outlives the joined call.
            unsafe {
                *base.get_mut(ci) = Some(acc);
            }
        })
        .expect("moirai global executor: fold_reduce_with");
    slots
        .into_iter()
        .flatten()
        .reduce(reduce)
        .unwrap_or_else(init)
}

/// Parallel map over the index domain `0..len`, collecting into a `Vec<R>` in
/// order, scheduled by policy `P`.
///
/// `map(i)` produces the element at index `i`. Use this for index-aligned maps
/// over multiple slices that [`map_collect_with`] cannot express — e.g. an
/// elementwise product `map_collect_index_with::<Adaptive>(n, |i| a[i] * b[i])`.
pub fn map_collect_index_with<P, R, Map>(len: usize, map: Map) -> Vec<R>
where
    P: ExecutionPolicy,
    R: Send,
    Map: Fn(usize) -> R + Send + Sync,
{
    if !P::parallelize(len) {
        return (0..len).map(map).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(len);
    // SAFETY: capacity is `len`; every slot is written exactly once below.
    unsafe {
        out.set_len(len);
    }
    enumerate_mut_with::<Parallel, _, _>(&mut out, |i, slot| {
        slot.write(map(i));
    });
    // SAFETY: every slot initialized; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), len, out.capacity()) }
}

/// Map each element of `data` in place with `f(index, &mut element)`, collecting
/// each returned value into a `Vec<R>` in order, scheduled by policy `P`.
///
/// The synchronous equivalent of rayon's
/// `data.par_iter_mut().enumerate().map(f).collect()`: each element is mutated
/// and produces a result. Use for parallel solve-in-place-and-collect loops.
pub fn map_collect_mut_with<P, T, R, F>(data: &mut [T], f: F) -> Vec<R>
where
    P: ExecutionPolicy,
    T: Send,
    R: Send,
    F: Fn(usize, &mut T) -> R + Send + Sync,
{
    let n = data.len();
    if !P::parallelize(n) {
        return data.iter_mut().enumerate().map(|(i, x)| f(i, x)).collect();
    }
    let mut out: Vec<core::mem::MaybeUninit<R>> = Vec::with_capacity(n);
    // SAFETY: capacity is `n`; every slot is written exactly once below.
    unsafe {
        out.set_len(n);
    }
    let data_ptr = DisjointMutPtr(data.as_mut_ptr());
    let out_ptr = DisjointMutPtr(out.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(n, move |i| {
            // SAFETY: each index in `0..n` is visited exactly once, so neither the
            // input element nor the output slot at `i` aliases another task's.
            let elem = unsafe { data_ptr.get_mut(i) };
            let result = f(i, elem);
            unsafe { out_ptr.get_mut(i).write(result) };
        })
        .expect("moirai global executor: map_collect_mut_with");
    // SAFETY: every slot initialized; `MaybeUninit<R>` shares `R`'s layout.
    let mut out = core::mem::ManuallyDrop::new(out);
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<R>(), n, out.capacity()) }
}

/// Parallel reduction over the index domain `0..len`, scheduled by policy `P`.
///
/// `map(i)` produces a value for index `i`; results are folded within and across
/// chunks with `reduce`, seeded by `identity` (which must be `reduce`'s neutral
/// element). Use this for index-aligned reductions over multiple slices that
/// [`map_reduce_with`] cannot express — e.g. a dot product
/// `reduce_index_with::<Adaptive>(n, T::zero(), |i| a[i] * b[i], |x, y| x + y)`.
pub fn reduce_index_with<P, R, Map, Red>(len: usize, identity: R, map: Map, reduce: Red) -> R
where
    P: ExecutionPolicy,
    R: Send + Sync + Clone,
    Map: Fn(usize) -> R + Send + Sync,
    Red: Fn(R, R) -> R + Send + Sync,
{
    if len == 0 || !P::parallelize(len) {
        let mut acc = identity;
        for i in 0..len {
            acc = reduce(acc, map(i));
        }
        return acc;
    }
    global()
        .map_reduce_indexed::<SyncTask, _, _, _>(len, identity, map, reduce)
        .expect("moirai global executor: reduce_index_with")
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
    fn for_each_chunk_mut_processes_lanes() {
        // 1000 lanes of 7 elements; set each lane's elements to the lane index.
        let lanes = 1000usize;
        let width = 7usize;
        let mut data: Vec<u64> = vec![0; lanes * width];
        for_each_chunk_mut_with::<Adaptive, _, _>(&mut data, width, |chunk| {
            let lane = chunk[0]; // 0 initially; use position via first write instead
            let _ = lane;
            for x in chunk.iter_mut() {
                *x += 1;
            }
        });
        assert!(data.iter().all(|&v| v == 1));
        // uneven final chunk
        let mut d2: Vec<u64> = (0..10).collect();
        for_each_chunk_mut_with::<Parallel, _, _>(&mut d2, 3, |c| {
            for x in c {
                *x *= 10;
            }
        });
        assert_eq!(d2, (0..10).map(|x| x * 10).collect::<Vec<_>>());
    }

    #[test]
    fn for_each_index_visits_every_index() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n = 10_000usize;
        let counter = AtomicUsize::new(0);
        let sum = (0..n).map(|i| i as u64).sum::<u64>();
        let acc = std::sync::atomic::AtomicU64::new(0);
        for_each_index_with::<Adaptive, _>(n, |i| {
            counter.fetch_add(1, Ordering::Relaxed);
            acc.fetch_add(i as u64, Ordering::Relaxed);
        });
        assert_eq!(
            counter.load(Ordering::Relaxed),
            n,
            "every index visited once"
        );
        assert_eq!(
            acc.load(Ordering::Relaxed),
            sum,
            "index values summed correctly"
        );
    }

    #[test]
    fn for_each_chunk_pair_mut_processes_paired_chunks() {
        let frames = 300usize;
        let width = 5usize;
        let mut a: Vec<u64> = vec![0; frames * width];
        let mut b: Vec<u64> = vec![0; frames * width];
        for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
            &mut a,
            &mut b,
            width,
            |i, ca, cb| {
                for x in ca.iter_mut() {
                    *x = i as u64;
                }
                for (j, y) in cb.iter_mut().enumerate() {
                    *y = (i * width + j) as u64;
                }
            },
        );
        for i in 0..frames {
            assert!(a[i * width..(i + 1) * width].iter().all(|&v| v == i as u64));
            for j in 0..width {
                assert_eq!(b[i * width + j], (i * width + j) as u64);
            }
        }
    }

    #[test]
    fn for_each_chunk_mut_enumerated_passes_index() {
        let lanes = 500usize;
        let width = 4usize;
        let mut data: Vec<u64> = vec![0; lanes * width];
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(&mut data, width, |i, chunk| {
            for x in chunk.iter_mut() {
                *x = i as u64;
            }
        });
        for i in 0..lanes {
            assert!(data[i * width..(i + 1) * width]
                .iter()
                .all(|&v| v == i as u64));
        }
    }

    #[test]
    fn map_collect_mut_mutates_and_collects() {
        let mut data: Vec<u64> = (0..10_000).collect();
        let doubled_indices = map_collect_mut_with::<Adaptive, _, _, _>(&mut data, |i, x| {
            *x += 1; // mutate in place
            i as u64 // collect the index
        });
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, i as u64 + 1);
        }
        assert_eq!(doubled_indices, (0..10_000u64).collect::<Vec<_>>());
    }

    #[test]
    fn fold_reduce_accumulates_into_collection() {
        use std::collections::HashMap;
        let n = 30_000usize;
        // group i -> sum of i over its (i % 8) bucket
        let map = fold_reduce_with::<Adaptive, HashMap<usize, u64>, _, _, _>(
            n,
            HashMap::new,
            |mut acc, i| {
                *acc.entry(i % 8).or_insert(0) += i as u64;
                acc
            },
            |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            },
        );
        let mut expected: HashMap<usize, u64> = HashMap::new();
        for i in 0..n {
            *expected.entry(i % 8).or_insert(0) += i as u64;
        }
        assert_eq!(map, expected);
    }

    #[test]
    fn map_collect_index_zips_two_slices() {
        let a: Vec<u64> = (0..20_000).collect();
        let b: Vec<u64> = (0..20_000).map(|x| x + 1).collect();
        let prod = map_collect_index_with::<Adaptive, _, _>(a.len(), |i| a[i] * b[i]);
        let expected: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
        assert_eq!(prod, expected);
    }

    #[test]
    fn reduce_index_computes_dot_product() {
        let a: Vec<u64> = (0..50_000).collect();
        let b: Vec<u64> = (0..50_000).map(|x| x * 2).collect();
        let dot =
            reduce_index_with::<Adaptive, _, _, _>(a.len(), 0u64, |i| a[i] * b[i], |x, y| x + y);
        let expected: u64 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
        assert_eq!(dot, expected);
        // sequential policy agrees
        assert_eq!(
            reduce_index_with::<Sequential, _, _, _>(a.len(), 0u64, |i| a[i] * b[i], |x, y| x + y),
            expected
        );
    }

    #[test]
    fn empty_and_single_inputs_are_handled() {
        let empty: Vec<i32> = Vec::new();
        empty.par().for_each(|_| panic!("must not run"));
        assert_eq!(
            empty.par().map_reduce(42i64, |&x| x as i64, |a, b| a + b),
            42
        );
        let mut one = vec![7u64];
        one.par_mut().for_each(|x| *x += 1);
        assert_eq!(one, vec![8]);
    }

    #[cfg(feature = "melinoe")]
    #[test]
    fn test_par_partition_melinoe() {
        use melinoe::{brand_scope, MelinoeCell};
        let data = vec![0usize; 16];
        brand_scope(|token| {
            let mut cells: Vec<MelinoeCell<'_, usize>> = data.into_iter().map(MelinoeCell::new).collect();
            super::melinoe_ext::par_partition_for_each(&mut cells, 4, |start, mut shard| {
                for (j, slot) in shard.iter_mut().enumerate() {
                    *slot = start + j;
                }
            });
            let snap = token.share();
            for (i, cell) in cells.iter().enumerate() {
                assert_eq!(*cell.borrow(snap), i);
            }
        });
    }
}
