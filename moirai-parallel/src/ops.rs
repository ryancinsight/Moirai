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

/// Apply `f(state, chunk)` to each consecutive mutable chunk, creating one
/// reusable state value per scheduled worker shard.
///
/// This is the scratch-buffer form of [`for_each_chunk_mut_with`]. It matches
/// the allocation discipline of Rayon-style `for_each_init`/`for_each_with`
/// loops: a worker shard initializes `S` once, then reuses it for every logical
/// chunk assigned to that shard.
pub fn for_each_chunk_mut_with_state<P, T, S, Init, F>(
    data: &mut [T],
    chunk_size: usize,
    init: Init,
    f: F,
) where
    P: ExecutionPolicy,
    T: Send,
    S: Send,
    Init: Fn() -> S + Send + Sync,
    F: Fn(&mut S, &mut [T]) + Send + Sync,
{
    let n = data.len();
    if n == 0 || chunk_size == 0 {
        return;
    }
    let num_chunks = n.div_ceil(chunk_size);
    if !P::parallelize(n) || num_chunks <= 1 {
        let mut state = init();
        for chunk in data.chunks_mut(chunk_size) {
            f(&mut state, chunk);
        }
        return;
    }

    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(num_chunks)
        .max(1);
    let chunks_per_worker = num_chunks.div_ceil(workers);
    let base = DisjointMutPtr(data.as_mut_ptr());
    let init = &init;
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(workers, move |worker| {
            let first_chunk = worker * chunks_per_worker;
            let last_chunk = ((worker + 1) * chunks_per_worker).min(num_chunks);
            if first_chunk >= last_chunk {
                return;
            }
            let mut state = init();
            for chunk_index in first_chunk..last_chunk {
                let start = chunk_index * chunk_size;
                let end = (start + chunk_size).min(n);
                // SAFETY: logical chunks are assigned to exactly one worker and
                // are pairwise disjoint, so each mutable slice is exclusive.
                let chunk =
                    unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
                f(&mut state, chunk);
            }
        })
        .expect("moirai global executor: for_each_chunk_mut_with_state");
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

/// Apply `f(index, a_chunk, b_chunk, c_chunk, d_chunk)` to four **distinct**
/// mutable buffers chunked identically, scheduled by policy `P`.
///
/// This is the four-buffer counterpart to
/// [`for_each_chunk_pair_mut_enumerated_with`]. It is intended for fused
/// statistics and stencil bookkeeping kernels where one authoritative pass
/// updates several output arrays without allocating intermediate tuples.
pub fn for_each_chunk_quad_mut_enumerated_with<P, A, B, C, D, F>(
    a: &mut [A],
    b: &mut [B],
    c: &mut [C],
    d: &mut [D],
    chunk_size: usize,
    f: F,
) where
    P: ExecutionPolicy,
    A: Send,
    B: Send,
    C: Send,
    D: Send,
    F: Fn(usize, &mut [A], &mut [B], &mut [C], &mut [D]) + Send + Sync,
{
    let na = a.len();
    let nb = b.len();
    let nc = c.len();
    let nd = d.len();
    assert_eq!(na, nb, "quad chunk buffers must have equal lengths");
    assert_eq!(na, nc, "quad chunk buffers must have equal lengths");
    assert_eq!(na, nd, "quad chunk buffers must have equal lengths");
    if chunk_size == 0 || na == 0 {
        return;
    }
    let num_chunks = na.div_ceil(chunk_size);
    if !P::parallelize(na) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .zip(c.chunks_mut(chunk_size))
            .zip(d.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (((ca, cb), cc), cd))| f(i, ca, cb, cc, cd));
        return;
    }
    let abase = DisjointMutPtr(a.as_mut_ptr());
    let bbase = DisjointMutPtr(b.as_mut_ptr());
    let cbase = DisjointMutPtr(c.as_mut_ptr());
    let dbase = DisjointMutPtr(d.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |chunk_index| {
            let start = chunk_index * chunk_size;
            if start >= na || start >= nb || start >= nc || start >= nd {
                return;
            }
            let ea = (start + chunk_size).min(na);
            let eb = (start + chunk_size).min(nb);
            let ec = (start + chunk_size).min(nc);
            let ed = (start + chunk_size).min(nd);
            // SAFETY: chunks `[start, e*)` for distinct `chunk_index` values
            // are pairwise disjoint within each buffer and each is visited at
            // most once. The four input buffers are distinct non-aliasing
            // `&mut` slices, so the returned mutable chunk references cannot
            // alias each other.
            let ca =
                unsafe { core::slice::from_raw_parts_mut(abase.base().add(start), ea - start) };
            let cb =
                unsafe { core::slice::from_raw_parts_mut(bbase.base().add(start), eb - start) };
            let cc =
                unsafe { core::slice::from_raw_parts_mut(cbase.base().add(start), ec - start) };
            let cd =
                unsafe { core::slice::from_raw_parts_mut(dbase.base().add(start), ed - start) };
            f(chunk_index, ca, cb, cc, cd);
        })
        .expect("moirai global executor: for_each_chunk_quad_mut_enumerated_with");
}

/// Apply `f(index, a_chunk, b_chunk, c_chunk)` to three **distinct** mutable
/// buffers chunked identically, scheduled by policy `P`.
///
/// This is the three-buffer counterpart to
/// [`for_each_chunk_pair_mut_enumerated_with`].
pub fn for_each_chunk_triple_mut_enumerated_with<P, A, B, C, F>(
    a: &mut [A],
    b: &mut [B],
    c: &mut [C],
    chunk_size: usize,
    f: F,
) where
    P: ExecutionPolicy,
    A: Send,
    B: Send,
    C: Send,
    F: Fn(usize, &mut [A], &mut [B], &mut [C]) + Send + Sync,
{
    let na = a.len();
    let nb = b.len();
    let nc = c.len();
    assert_eq!(na, nb, "triple chunk buffers must have equal lengths");
    assert_eq!(na, nc, "triple chunk buffers must have equal lengths");
    if chunk_size == 0 || na == 0 {
        return;
    }
    let num_chunks = na.div_ceil(chunk_size);
    if !P::parallelize(na) || num_chunks <= 1 {
        a.chunks_mut(chunk_size)
            .zip(b.chunks_mut(chunk_size))
            .zip(c.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, ((ca, cb), cc))| f(i, ca, cb, cc));
        return;
    }
    let abase = DisjointMutPtr(a.as_mut_ptr());
    let bbase = DisjointMutPtr(b.as_mut_ptr());
    let cbase = DisjointMutPtr(c.as_mut_ptr());
    let f = &f;
    global()
        .for_each_indexed::<SyncTask, _>(num_chunks, move |chunk_index| {
            let start = chunk_index * chunk_size;
            if start >= na || start >= nb || start >= nc {
                return;
            }
            let ea = (start + chunk_size).min(na);
            let eb = (start + chunk_size).min(nb);
            let ec = (start + chunk_size).min(nc);
            // SAFETY: chunks `[start, e*)` for distinct `chunk_index` values
            // are pairwise disjoint within each buffer and each is visited at
            // most once. The three input buffers are distinct non-aliasing
            // `&mut` slices, so the returned mutable chunk references cannot
            // alias each other.
            let ca =
                unsafe { core::slice::from_raw_parts_mut(abase.base().add(start), ea - start) };
            let cb =
                unsafe { core::slice::from_raw_parts_mut(bbase.base().add(start), eb - start) };
            let cc =
                unsafe { core::slice::from_raw_parts_mut(cbase.base().add(start), ec - start) };
            f(chunk_index, ca, cb, cc);
        })
        .expect("moirai global executor: for_each_chunk_triple_mut_enumerated_with");
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
