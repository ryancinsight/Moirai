//! Unit-task planning: the bytes-per-task arithmetic shared by the
//! [`for_each_unit_task_*`](self) operators (ADR 0059).

use crate::policy::ExecutionPolicy;

/// Bytes of work one scheduled task carries.
///
/// One length-64 lane per task made apollo's 64Â³ transform pass 3.3x slower
/// than serial: moirai's per-task dispatch measures about 180 ns against a
/// 55â€“130 ns lane. Tasks of 64 KiB made the same pass faster than serial and
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
pub(super) struct UnitTaskPlan {
    /// Units one task carries.
    pub(super) per_task: usize,
    /// Elements one task carries: `per_task` whole units.
    pub(super) task_len: usize,
    /// Tasks the pass splits into; the last may be shorter.
    pub(super) tasks: usize,
    /// Whether the policy spreads the tasks over workers.
    pub(super) parallel: bool,
}

/// Rejects data that does not divide into whole `unit_len`-element units.
#[track_caller]
pub(super) fn assert_whole_units(len: usize, unit_len: usize) {
    assert!(
        unit_len > 0 && len.is_multiple_of(unit_len),
        "unit tasks need whole units: data length {len} is not a multiple of unit length {unit_len}",
    );
}

/// The task layout and policy decision for `len` elements of whole
/// `unit_len`-element units that each move `unit_bytes`, or `None` when there
/// is no unit to run. `P::parallelize_work` is consulted only when the pass
/// spans more than one task.
pub(super) fn plan_unit_tasks<P: ExecutionPolicy>(
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
