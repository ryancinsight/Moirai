//! Whole-unit tasks sized by the bytes each unit moves (ADR 0059).
//!
//! A unit is a lane, row or matrix: a fixed run of elements that one closure
//! call transforms together. How many units a task carries, and whether the
//! pass runs in parallel at all, depend on the bytes a unit moves — including
//! any input read beside it — not on its element count.

use super::super::DisjointMutPtr;
use crate::policy::ExecutionPolicy;
use moirai_executor::{SyncTask, global};

#[cfg(test)]
mod tests;

/// Bytes of work one scheduled task carries.
///
/// One length-64 lane per task made apollo's 64³ transform pass 3.3x slower
/// than serial: moirai's per-task dispatch measures about 180 ns against a
/// 55–130 ns lane. Tasks of 64 KiB made the same pass faster than serial and
/// than quarter-megabyte tasks (apollo `dimension_3d::pass_attribution`,
/// 2026-09-09), and leto-ops' batched transpose settled on the same width. It
/// stays inside one core's L2.
pub const UNIT_TASK_BYTES: usize = 64 * 1024;

/// Units one task carries when each unit moves `unit_bytes`, never fewer than
/// one: a unit at or above [`UNIT_TASK_BYTES`] is a task on its own.
#[must_use]
pub const fn units_per_task(unit_bytes: usize) -> usize {
    let per_task = UNIT_TASK_BYTES / if unit_bytes == 0 { 1 } else { unit_bytes };
    if per_task == 0 { 1 } else { per_task }
}

/// Task layout of a unit-task pass over `len` elements.
#[derive(Clone, Copy)]
struct UnitTaskPlan {
    /// Units one task carries.
    per_task: usize,
    /// Elements one task carries: `per_task` whole units.
    task_len: usize,
    /// Tasks the pass splits into; the last may be shorter.
    tasks: usize,
    /// Whether the policy spreads the tasks over workers.
    parallel: bool,
}

/// Rejects data that does not divide into whole `unit_len`-element units.
#[track_caller]
fn assert_whole_units(len: usize, unit_len: usize) {
    assert!(
        unit_len > 0 && len.is_multiple_of(unit_len),
        "unit tasks need whole units: data length {len} is not a multiple of unit length {unit_len}",
    );
}

/// The task layout and policy decision for `len` elements of whole
/// `unit_len`-element units that each move `unit_bytes`, or `None` when there
/// is no unit to run. `P::parallelize_work` is consulted only when the pass
/// spans more than one task.
fn plan_unit_tasks<P: ExecutionPolicy>(
    len: usize,
    unit_len: usize,
    unit_bytes: usize,
) -> Option<UnitTaskPlan> {
    let units = len / unit_len;
    if units == 0 {
        return None;
    }
    let per_task = units_per_task(unit_bytes);
    let tasks = units.div_ceil(per_task);
    Some(UnitTaskPlan {
        per_task,
        task_len: per_task * unit_len,
        tasks,
        parallel: tasks > 1 && P::parallelize_work(len, tasks, units.saturating_mul(unit_bytes)),
    })
}

/// Apply `f(state, first_unit, units)` to consecutive runs of whole units of
/// `data`, each run sized to about [`UNIT_TASK_BYTES`] of work.
///
/// `data` holds whole units of `unit_len` elements. `unit_bytes` is the total a
/// unit moves: its own bytes plus any input the closure reads beside it, so a
/// pass whose output sits beside a wider input counts both. The pass runs in
/// parallel when `P::parallelize_work(data.len(), tasks, units * unit_bytes)`
/// says so; `init` builds one state per scheduled task (one in all when
/// serial), and `first_unit` is the index of the run's first unit, which lets
/// the closure read shared inputs by unit.
///
/// # Panics
///
/// Panics if `unit_len` is zero or `data.len()` is not a multiple of it, and
/// propagates a panic raised by `init` or `f`.
///
/// # Examples
///
/// ```
/// use moirai_parallel::{for_each_unit_task_mut_with, WorkBytes};
///
/// let source: Vec<u64> = (0..12).collect();
/// let mut rows = vec![0_u64; 12];
/// // Rows of 3, each moving its 24 output bytes and 24 input bytes.
/// for_each_unit_task_mut_with::<WorkBytes<{ 1 << 20 }>, _, _, _, _>(
///     &mut rows,
///     3,
///     48,
///     || (),
///     |(), first_unit, units| {
///         let start = first_unit * 3;
///         for (out, &input) in units.iter_mut().zip(&source[start..]) {
///             *out = 2 * input;
///         }
///     },
/// );
/// assert_eq!(rows, (0..12).map(|v| 2 * v).collect::<Vec<u64>>());
/// ```
#[track_caller]
pub fn for_each_unit_task_mut_with<P, T, S, Init, F>(
    data: &mut [T],
    unit_len: usize,
    unit_bytes: usize,
    init: Init,
    f: F,
) where
    P: ExecutionPolicy,
    T: Send,
    Init: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, &mut [T]) + Send + Sync,
{
    assert_whole_units(data.len(), unit_len);
    let Some(UnitTaskPlan {
        per_task,
        task_len,
        tasks,
        parallel,
    }) = plan_unit_tasks::<P>(data.len(), unit_len, unit_bytes)
    else {
        return;
    };
    if !parallel {
        let mut state = init();
        for (task, run) in data.chunks_mut(task_len).enumerate() {
            f(&mut state, task * per_task, run);
        }
        return;
    }
    let n = data.len();
    let base = DisjointMutPtr(data.as_mut_ptr());
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |task| {
            let start = task * task_len;
            let end = (start + task_len).min(n);
            // SAFETY: the runs `[task * task_len, end)` for distinct `task` are
            // pairwise disjoint and each is visited exactly once, so no two
            // tasks form `&mut` to one element.
            let run =
                unsafe { core::slice::from_raw_parts_mut(base.base().add(start), end - start) };
            let mut state = init();
            f(&mut state, task * per_task, run);
        })
        .expect("moirai global executor: for_each_unit_task_mut_with");
}

/// Apply `f(state, first_unit, a_units, b_units)` to aligned runs of whole
/// units of two mutable buffers, each run sized to about [`UNIT_TASK_BYTES`]
/// of work.
///
/// This is [`for_each_unit_task_mut_with`] for a pass that writes two fields
/// per element: `a` and `b` hold the same number of `unit_len`-element units,
/// and each task receives the same units of both. `unit_bytes` counts one unit
/// of `a`, one of `b` and any input read beside them.
///
/// # Panics
///
/// Panics if `unit_len` is zero, `a.len()` is not a multiple of it, or `b` is
/// not the length of `a`, and propagates a panic raised by `init` or `f`.
///
/// # Examples
///
/// ```
/// use moirai_parallel::{for_each_unit_task_pair_mut_with, WorkBytes};
///
/// let source: Vec<u64> = (0..12).collect();
/// let (mut sums, mut squares) = (vec![0_u64; 12], vec![0_u64; 12]);
/// for_each_unit_task_pair_mut_with::<WorkBytes<{ 1 << 20 }>, _, _, _, _, _>(
///     &mut sums,
///     &mut squares,
///     4,
///     72,
///     || (),
///     |(), first_unit, sums, squares| {
///         let start = first_unit * 4;
///         for ((sum, square), &value) in sums.iter_mut().zip(squares).zip(&source[start..]) {
///             *sum = value + 1;
///             *square = value * value;
///         }
///     },
/// );
/// assert_eq!(sums, (1..13).collect::<Vec<u64>>());
/// assert_eq!(squares, (0..12).map(|v| v * v).collect::<Vec<u64>>());
/// ```
#[track_caller]
pub fn for_each_unit_task_pair_mut_with<P, A, B, S, Init, F>(
    a: &mut [A],
    b: &mut [B],
    unit_len: usize,
    unit_bytes: usize,
    init: Init,
    f: F,
) where
    P: ExecutionPolicy,
    A: Send,
    B: Send,
    Init: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, &mut [A], &mut [B]) + Send + Sync,
{
    assert_whole_units(a.len(), unit_len);
    assert_eq!(
        a.len(),
        b.len(),
        "paired unit tasks need buffers of one length",
    );
    let Some(UnitTaskPlan {
        per_task,
        task_len,
        tasks,
        parallel,
    }) = plan_unit_tasks::<P>(a.len(), unit_len, unit_bytes)
    else {
        return;
    };
    if !parallel {
        let mut state = init();
        for (task, (run_a, run_b)) in a
            .chunks_mut(task_len)
            .zip(b.chunks_mut(task_len))
            .enumerate()
        {
            f(&mut state, task * per_task, run_a, run_b);
        }
        return;
    }
    let n = a.len();
    let base_a = DisjointMutPtr(a.as_mut_ptr());
    let base_b = DisjointMutPtr(b.as_mut_ptr());
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |task| {
            let start = task * task_len;
            let end = (start + task_len).min(n);
            // SAFETY: the runs `[task * task_len, end)` for distinct `task` are
            // pairwise disjoint within `a` and each is visited exactly once.
            let run_a =
                unsafe { core::slice::from_raw_parts_mut(base_a.base().add(start), end - start) };
            // SAFETY: the same disjoint runs within `b`, a distinct buffer of
            // the same length that the caller holds exclusively.
            let run_b =
                unsafe { core::slice::from_raw_parts_mut(base_b.base().add(start), end - start) };
            let mut state = init();
            f(&mut state, task * per_task, run_a, run_b);
        })
        .expect("moirai global executor: for_each_unit_task_pair_mut_with");
}

/// Apply `f(state, first_unit, a_units, b_units, c_units)` to aligned runs of
/// whole units of three mutable buffers, each run sized to about
/// [`UNIT_TASK_BYTES`] of work.
///
/// This is [`for_each_unit_task_pair_mut_with`] for a pass that writes three
/// fields per element: `a`, `b` and `c` hold the same number of
/// `unit_len`-element units, and each task receives the same units of all
/// three. `unit_bytes` counts one unit of each buffer and any input read beside
/// them.
///
/// # Panics
///
/// Panics if `unit_len` is zero, `a.len()` is not a multiple of it, or `b` or
/// `c` is not the length of `a`, and propagates a panic raised by `init` or `f`.
///
/// # Examples
///
/// ```
/// use moirai_parallel::{for_each_unit_task_triple_mut_with, WorkBytes};
///
/// let source: Vec<u64> = (0..12).collect();
/// let (mut xs, mut ys, mut zs) = (vec![0_u64; 12], vec![0_u64; 12], vec![0_u64; 12]);
/// // Rows of 4, each moving 32 bytes of each output and 32 of the source.
/// for_each_unit_task_triple_mut_with::<WorkBytes<{ 1 << 20 }>, _, _, _, _, _, _>(
///     &mut xs,
///     &mut ys,
///     &mut zs,
///     4,
///     128,
///     || (),
///     |(), first_unit, xs, ys, zs| {
///         let start = first_unit * 4;
///         let outputs = xs.iter_mut().zip(ys.iter_mut()).zip(zs.iter_mut());
///         for (((x, y), z), &value) in outputs.zip(&source[start..]) {
///             *x += value;
///             *y += value;
///             *z += value;
///         }
///     },
/// );
/// assert!(xs == source && ys == source && zs == source);
/// ```
#[track_caller]
pub fn for_each_unit_task_triple_mut_with<P, A, B, C, S, Init, F>(
    a: &mut [A],
    b: &mut [B],
    c: &mut [C],
    unit_len: usize,
    unit_bytes: usize,
    init: Init,
    f: F,
) where
    P: ExecutionPolicy,
    A: Send,
    B: Send,
    C: Send,
    Init: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, &mut [A], &mut [B], &mut [C]) + Send + Sync,
{
    assert_whole_units(a.len(), unit_len);
    assert_eq!(
        a.len(),
        b.len(),
        "triple unit tasks need buffers of one length",
    );
    assert_eq!(
        a.len(),
        c.len(),
        "triple unit tasks need buffers of one length",
    );
    let Some(UnitTaskPlan {
        per_task,
        task_len,
        tasks,
        parallel,
    }) = plan_unit_tasks::<P>(a.len(), unit_len, unit_bytes)
    else {
        return;
    };
    if !parallel {
        let mut state = init();
        let runs = a
            .chunks_mut(task_len)
            .zip(b.chunks_mut(task_len))
            .zip(c.chunks_mut(task_len));
        for (task, ((run_a, run_b), run_c)) in runs.enumerate() {
            f(&mut state, task * per_task, run_a, run_b, run_c);
        }
        return;
    }
    let n = a.len();
    let base_a = DisjointMutPtr(a.as_mut_ptr());
    let base_b = DisjointMutPtr(b.as_mut_ptr());
    let base_c = DisjointMutPtr(c.as_mut_ptr());
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |task| {
            let start = task * task_len;
            let end = (start + task_len).min(n);
            // SAFETY: the runs `[task * task_len, end)` for distinct `task` are
            // pairwise disjoint within `a` and each is visited exactly once.
            let run_a =
                unsafe { core::slice::from_raw_parts_mut(base_a.base().add(start), end - start) };
            // SAFETY: the same disjoint runs within `b`, a distinct buffer of
            // the same length that the caller holds exclusively.
            let run_b =
                unsafe { core::slice::from_raw_parts_mut(base_b.base().add(start), end - start) };
            // SAFETY: the same disjoint runs within `c`, a third distinct buffer
            // of the same length that the caller holds exclusively.
            let run_c =
                unsafe { core::slice::from_raw_parts_mut(base_c.base().add(start), end - start) };
            let mut state = init();
            f(&mut state, task * per_task, run_a, run_b, run_c);
        })
        .expect("moirai global executor: for_each_unit_task_triple_mut_with");
}
