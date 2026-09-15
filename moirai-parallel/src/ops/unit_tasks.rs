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
    assert!(
        unit_len > 0 && data.len().is_multiple_of(unit_len),
        "unit tasks need whole units: data length {} is not a multiple of unit length {unit_len}",
        data.len(),
    );
    let units = data.len() / unit_len;
    if units == 0 {
        return;
    }
    let per_task = units_per_task(unit_bytes);
    let task_len = per_task * unit_len;
    let tasks = units.div_ceil(per_task);
    if tasks <= 1 || !P::parallelize_work(data.len(), tasks, units.saturating_mul(unit_bytes)) {
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
    assert!(
        unit_len > 0 && a.len().is_multiple_of(unit_len),
        "unit tasks need whole units: data length {} is not a multiple of unit length {unit_len}",
        a.len(),
    );
    assert_eq!(
        a.len(),
        b.len(),
        "paired unit tasks need buffers of one length",
    );
    let units = a.len() / unit_len;
    if units == 0 {
        return;
    }
    let per_task = units_per_task(unit_bytes);
    let task_len = per_task * unit_len;
    let tasks = units.div_ceil(per_task);
    if tasks <= 1 || !P::parallelize_work(a.len(), tasks, units.saturating_mul(unit_bytes)) {
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
