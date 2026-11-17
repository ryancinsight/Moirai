//! Synchronous data-parallel operators over the unified scheduler.
//!
//! # Safety
//!
//! The mutable operators split a buffer across worker tasks through Melinoe's
//! [`WriterShard`](melinoe::region::WriterShard) /
//! [`ParChunks`](melinoe::region::ParChunks), which own the disjoint-partition
//! contract and the range math once; each operator brands the caller's slice in
//! place ([`MelinoeCell::from_mut_slice`](melinoe::MelinoeCell::from_mut_slice))
//! and vends one partition per task. One
//! executor contract makes every such access sound, and each per-site `SAFETY`
//! comment appeals to it:
//!
//! - **Disjoint partition.** `global().for_each_indexed(count, f)` invokes `f`
//!   with each index in `0..count` exactly once — the same guarantee
//!   `ParChunks::get_unchecked_chunk` requires of its caller. The operators map
//!   that index (or a `chunk_size`-strided range derived from it) to a
//!   *disjoint* slice of the buffer, so no two concurrent tasks ever form
//!   `&mut` to the same element. Multi-buffer operators additionally rely on the
//!   caller's distinct `&mut [_]` arguments being non-aliasing (guaranteed by
//!   the borrow checker at the call site).
//! - **All-or-error collect.** Indexed output uses the chunk owner in
//!   moirai-core. A panicking mapper drops its initialized prefix in its worker.
//!   The parent drops completed chunks after joining all jobs. Success takes
//!   the same initialized allocation in index order.

use crate::policy::ExecutionPolicy;
use melinoe::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_core::error::{ExecutorError, ExecutorResult};
use moirai_executor::{HybridExecutor, SchedulerScope, SyncTask, global};
use std::sync::Mutex;

mod chunks;
mod collect;
mod shards;
mod unit_tasks;
pub use chunks::{
    ChunkBuffersError, for_each_chunk_buffers_mut_enumerated_with,
    for_each_chunk_mut_enumerated_with, for_each_chunk_mut_with, for_each_chunk_mut_with_state,
    for_each_chunk_pair_mut_enumerated_with, for_each_chunk_quad_mut_enumerated_with,
    for_each_chunk_triple_mut_enumerated_with,
};
pub use collect::{map_collect_index_with, map_collect_mut_with, map_collect_with};
pub use unit_tasks::{
    UNIT_TASK_BYTES, for_each_unit_task_many_mut_with, for_each_unit_task_mut_with,
    for_each_unit_task_pair_mut_with, for_each_unit_task_range_with,
    for_each_unit_task_triple_mut_with, units_per_task,
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
/// caller, [`crate::Parallel`] schedules the left closure on the unified scheduler and
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
pub struct Scope<'scope, 'env: 'scope> {
    inner: &'scope SchedulerScope<'env, SyncTask>,
}

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
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
        F: FnOnce() + Send + 'env,
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
///
/// A task cannot borrow a value local to the body, which is dropped before the
/// scheduler runs the buffered task:
///
/// ```compile_fail,E0597
/// moirai_parallel::scope(|s| {
///     let local = vec![7_u8; 8];
///     let borrowed = &local;
///     s.spawn(move || assert_eq!(borrowed[0], 7));
/// });
/// ```
#[inline]
pub fn scope<'env, F, R>(body: F) -> R
where
    F: for<'scope> FnOnce(&Scope<'scope, 'env>) -> R,
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

/// Partition width for an indexed region: about one worker-sized chunk per task.
///
/// `for_each_indexed` already splits its domain into contiguous worker-sized
/// ranges, so matching the partition to that width keeps the same element
/// distribution while constructing one `WriterShard` per range instead of one
/// per element. A width of `1` would put a shard build (and its range math) on
/// every element's path for no benefit.
fn worker_chunk_size(len: usize) -> usize {
    let workers = moirai_core::executor::logical_parallelism().max(1);
    len.div_ceil(workers).max(1)
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
    let partitions =
        WriterShard::new(MelinoeCell::from_mut_slice(data)).par_chunks(worker_chunk_size(n));
    let tasks = partitions.len();
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |c| {
            // SAFETY: the scheduler visits each index in `0..tasks` exactly once,
            // which is the contract `get_unchecked_chunk` documents; distinct
            // partitions name disjoint element ranges, so no two tasks form a
            // `&mut` to the same element. `data` is branded for the whole
            // joined call, so the view cannot outlive it.
            let mut shard = unsafe { partitions.get_unchecked_chunk(c) };
            for element in shard.iter_mut() {
                f(element);
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
    let chunk_size = worker_chunk_size(n);
    let partitions = WriterShard::new(MelinoeCell::from_mut_slice(data)).par_chunks(chunk_size);
    let tasks = partitions.len();
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |c| {
            // Partition `c` covers the elements `[c * chunk_size, …)`.
            let start = c * chunk_size;
            // SAFETY: each index in `0..tasks` is visited exactly once; distinct
            // partitions are disjoint element ranges, so the element at absolute
            // index `start + offset` is touched by exactly one task. See
            // `for_each_mut_with`.
            let mut shard = unsafe { partitions.get_unchecked_chunk(c) };
            for (offset, element) in shard.iter_mut().enumerate() {
                f(start + offset, element);
            }
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
    let workers = moirai_core::executor::logical_parallelism().max(1);
    let chunks = workers.min(len).max(1);
    let chunk = len.div_ceil(chunks);
    let mut slots: Vec<Option<A>> = (0..chunks).map(|_| None).collect();
    let partitions =
        WriterShard::new(MelinoeCell::from_mut_slice(slots.as_mut_slice())).par_chunks(1);
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
            // SAFETY: each `ci` in `0..chunks` is visited exactly once, and
            // partition `ci` is exactly slot `ci` (one cell per partition), so
            // every slot is written exactly once and no two tasks alias.
            let mut shard = unsafe { partitions.get_unchecked_chunk(ci) };
            if let Some(slot) = shard.get_mut(0) {
                *slot = Some(acc);
            }
        })
        .expect("moirai global executor: fold_reduce_with");
    slots
        .into_iter()
        .flatten()
        .reduce(reduce)
        .unwrap_or_else(init)
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
