//! Synchronous data-parallel operators over the unified scheduler.
//!
//! # Safety
//!
//! The mutable operators split a buffer across worker tasks through
//! `DisjointMutPtr`, which hands out `&mut` by raw pointer with no
//! borrow-checker aliasing proof. Two executor contracts make
//! every such access sound; each per-site `SAFETY` comment appeals to one of
//! them:
//!
//! - **Disjoint partition.** `global().for_each_indexed(count, f)` invokes `f`
//!   with each index in `0..count` exactly once. The operators map that index
//!   (or a `chunk_size`-strided range derived from it) to a *disjoint* slice of
//!   the buffer, so no two concurrent tasks ever form `&mut` to the same
//!   element. Multi-buffer operators additionally rely on the caller's distinct
//!   `&mut [_]` arguments being non-aliasing (guaranteed by the borrow checker
//!   at the call site).
//! - **All-or-error collect.** The `map_collect_*` helpers build a
//!   `Vec<MaybeUninit<R>>`, `set_len` it (sound — `MaybeUninit` needs no
//!   initialization), fill every slot through the disjoint-partition contract,
//!   then reinterpret it as `Vec<R>`. `for_each_indexed` returns `Ok` only after
//!   writing every index, so the reinterpretation is reached only when all slots
//!   are initialized; a task panic instead surfaces as `Err`, which `.expect()`
//!   turns into a propagating panic that unwinds with the buffer still typed
//!   `MaybeUninit<R>` (its contents are not dropped — a leak of the written
//!   values on panic, never a use of uninitialized memory).

use super::DisjointMutPtr;
use crate::policy::{ExecutionPolicy, Parallel};
use moirai_core::error::{ExecutorError, ExecutorResult};
use moirai_executor::{global, HybridExecutor, SchedulerScope, SyncTask};
use std::sync::Mutex;

mod chunks;
pub use chunks::{
    for_each_chunk_buffers_mut_enumerated_with, for_each_chunk_mut_enumerated_with,
    for_each_chunk_mut_with, for_each_chunk_mut_with_state,
    for_each_chunk_pair_mut_enumerated_with, for_each_chunk_quad_mut_enumerated_with,
    for_each_chunk_triple_mut_enumerated_with, ChunkBuffersError,
};

/// State of the scheduled branch of a join.
///
/// The scheduler can refuse a job — while shutting down, or when a worker's
/// bounded admission queue is full — and it drops the refused job before
/// returning the error. A branch owned by that job would go with it, so the
/// closure lives here instead and whichever lane reaches it first takes it.
/// The caller can therefore still run a branch the scheduler never did.
enum Branch<F, R> {
    /// Nobody has claimed this branch yet.
    Pending(F),
    /// A lane claimed the branch and has not published a result. Observing
    /// this once the scope has joined means that lane unwound.
    Claimed,
    /// Ran to completion.
    Done(R),
}

impl<F, R> Branch<F, R>
where
    F: FnOnce() -> R,
{
    /// Take the closure if this lane is the one that gets to run it.
    fn claim(&mut self) -> Option<F> {
        match std::mem::replace(self, Self::Claimed) {
            Self::Pending(branch) => Some(branch),
            other => {
                *self = other;
                None
            }
        }
    }

    fn complete(&mut self, result: R) {
        *self = Self::Done(result);
    }

    /// Run the branch on a lane that shares the slot, unless another lane
    /// already claimed it.
    ///
    /// The lock is released before the closure runs, so a branch never holds it
    /// across arbitrary caller code and a panicking branch cannot poison it.
    fn run_shared(slot: &Mutex<Self>) {
        let Some(branch) = lock(slot).claim() else {
            return;
        };
        let result = branch();
        lock(slot).complete(result);
    }

    /// Run a branch that never leaves this thread.
    fn run_here(&mut self) {
        let Some(branch) = self.claim() else {
            return;
        };
        let result = branch();
        self.complete(result);
    }

    /// Take the finished value.
    fn into_result(self) -> R {
        match self {
            Self::Done(result) => result,
            _ => panic!("invariant: a join branch neither ran nor reported failure"),
        }
    }
}

/// Lock without propagating poisoning: every path that touches a slot leaves
/// it in a consistent state, and [`Branch::run_shared`] never holds the lock
/// across the branch closure, so a poisoned flag carries no information here.
fn lock<T>(slot: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    slot.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reclaim a slot once every lane that shared it has finished.
fn lock_owned<T>(slot: Mutex<T>) -> T {
    slot.into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Run two closures to completion and return both results.
///
/// This is the synchronous Rayon-style `join` shape. The policy is selected at
/// compile time; [`Sequential`](crate::Sequential) runs both closures on the
/// caller, [`Parallel`] schedules the left closure on the unified scheduler and
/// runs the right closure on the caller lane, and [`crate::Adaptive`] currently
/// stays sequential for a fixed two-branch join.
///
/// A branch the scheduler refuses runs on the caller instead, so a shutting-down
/// or saturated executor makes the join sequential rather than losing a branch.
///
/// # Panics
///
/// Panics if a branch panicked, propagating the failure on the caller's thread
/// as rayon does.
pub fn join_with<P, A, B, RA, RB>(left: A, right: B) -> (RA, RB)
where
    P: ExecutionPolicy,
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB,
    RA: Send,
{
    if !P::parallelize_pair() {
        return (left(), right());
    }

    join_on(global(), left, right)
}

/// [`join_with`]'s parallel path against a named executor.
///
/// Separate from the public entry so the refusal path can be exercised against
/// a shut-down executor in tests.
pub(crate) fn join_on<A, B, RA, RB>(executor: &HybridExecutor, left: A, right: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB,
    RA: Send,
{
    // Only the left branch crosses a lane boundary, so only it needs a shared
    // slot. The right branch stays on this thread and is reborrowed by the
    // scope body, which leaves it runnable here if the body returns early.
    let left_slot = Mutex::new(Branch::Pending(left));
    let mut right_slot = Branch::Pending(right);

    let forked = executor.scope::<SyncTask, _>(|scope| {
        scope.spawn(|_| Branch::run_shared(&left_slot))?;
        // Enter the scheduler before the caller takes its own branch, so the
        // two overlap instead of running back to back.
        scope.flush()?;
        right_slot.run_here();
        Ok(())
    });

    match forked {
        Ok(()) => {}
        // The scheduler refused the job and dropped it unexecuted, so neither
        // branch is guaranteed to have run. Both claims are idempotent: a
        // branch that did run is no longer `Pending`.
        Err(ExecutorError::ShuttingDown | ExecutorError::ResourceExhausted(_)) => {
            Branch::run_shared(&left_slot);
            right_slot.run_here();
        }
        Err(error) => panic!("invariant: scheduled join branch failed ({error})"),
    }

    (
        lock_owned(left_slot).into_result(),
        right_slot.into_result(),
    )
}

/// Adaptive Rayon-style two-closure join.
///
/// Use [`join_with`] to force a specific execution policy.
pub fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB,
    RA: Send,
{
    join_with::<crate::Adaptive, _, _, _, _>(left, right)
}

/// Borrowing scope for spawning parallel sub-tasks that may capture non-`'static`
/// references.
///
/// Created by [`scope`]. Each [`Scope::spawn`] call registers a job on the unified
/// scheduler; the scope blocks until every spawned job has completed before the
/// body closure returns, so borrowed data cannot escape the scope.
///
/// This is the Rayon-style `scope` shape, adapted to Moirai's unified hybrid
/// scheduler. Unlike [`join`], which forks exactly two branches, `scope` allows
/// an arbitrary number of sub-tasks to be spawned and joined within a single
/// region.
pub struct Scope<'scope> {
    inner: &'scope SchedulerScope<'scope, SyncTask>,
}

impl<'scope> Scope<'scope> {
    /// Spawn a parallel sub-task within this scope.
    ///
    /// The task may borrow values that outlive the scope call. The scope waits
    /// for every spawned task before returning, so borrowed data cannot escape.
    /// A task the scheduler's admission queue turns away runs on the calling
    /// thread instead, so backpressure costs parallelism rather than the task.
    ///
    /// # Panics
    ///
    /// Panics if the underlying scheduler refuses to register the task, which
    /// registration itself does not do — the failure surfaces from [`scope`]
    /// when the scheduler is shutting down.
    #[inline]
    pub fn spawn<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        self.inner
            .spawn(move |_| task())
            .expect("moirai global executor: scope spawn");
    }
}

/// Create a borrowing scope for parallel sub-tasks.
///
/// Within the body closure, [`Scope::spawn`] registers jobs on the unified
/// scheduler. The scope blocks until every spawned job has completed before
/// returning, so tasks may borrow non-`'static` data from the enclosing
/// environment.
///
/// This is the Rayon-style `scope` shape, adapted to Moirai's unified hybrid
/// scheduler.
///
/// # Examples
///
/// ```
/// use moirai_parallel::scope;
///
/// let data: Vec<u64> = (0..1000).collect();
/// use std::sync::atomic::{AtomicU64, Ordering};
///
/// let sum = AtomicU64::new(0);
/// scope(|s| {
///     s.spawn(|| {
///         sum.fetch_add(data.iter().sum::<u64>(), Ordering::Relaxed);
///     });
///     s.spawn(|| {
///         sum.fetch_add(data.len() as u64, Ordering::Relaxed);
///     });
/// });
/// assert_eq!(sum.load(Ordering::Relaxed), data.iter().sum::<u64>() + 1000);
/// ```
#[inline]
pub fn scope<F, R>(body: F) -> R
where
    F: for<'scope> FnOnce(&Scope<'scope>) -> R,
    R: Send,
{
    let mut result = None;
    global()
        .scope::<SyncTask, _>(|inner| {
            let scope = Scope { inner };
            result = Some(body(&scope));
            ExecutorResult::Ok(())
        })
        .expect("moirai global executor: scope");

    result.expect("scoped body must complete")
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
    let workers = themis::CpuTopology::detect()
        .map(|topology| topology.logical_processors())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .max(1);
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
            // SAFETY: unique visit of index i under the indexed scheduler,
            // so this mutable view aliases nothing.
            let elem = unsafe { data_ptr.get_mut(i) };
            let result = f(i, elem);
            // SAFETY: slot i was lengthened into place above and is written
            // exactly once by this unique visit.
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
