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
