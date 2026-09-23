//! Whole-unit tasks for passes that address their own data: the walk is a
//! range of unit indices, so only the task width comes from the runtime.

use super::layout::{UnitTaskPlan, plan_unit_tasks};
use crate::policy::ExecutionPolicy;
use moirai_executor::{SyncTask, global};

/// Apply `f(state, first_unit, units)` to consecutive runs of whole units of a
/// pass that addresses its own data, each run sized to about
/// [`crate::UNIT_TASK_BYTES`] of work.
///
/// This is [`crate::for_each_unit_task_mut_with`] for a pass whose units are not a
/// dense slice the runtime can split: a strided row walk, a tiled block pass,
/// a reduction that writes one output per axis index. The caller owns the
/// disjointness proof for whatever those indices address, exactly as it does
/// today; this owns the one decision the slice operators own â€” how many units
/// a task carries, from the bytes a unit moves, and whether the pass spreads
/// over workers at all.
///
/// `unit_bytes` is the total a unit moves, its own bytes plus any input read
/// beside it. `init` builds one state per scheduled task (one in all when
/// serial). Each unit index in `0..units` is passed to exactly one call, in
/// one run of consecutive indices.
///
/// # Panics
///
/// Propagates a panic raised by `init` or `f`.
///
/// # Examples
///
/// ```
/// use moirai_parallel::{for_each_unit_task_range_with, Parallel};
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// // Rows of a strided matrix: the walk addresses its own offsets, so only
/// // the task width comes from the runtime.
/// let (rows, row_bytes) = (1024, 8 * 64);
/// let visited: Vec<AtomicUsize> = (0..rows).map(|_| AtomicUsize::new(0)).collect();
/// for_each_unit_task_range_with::<Parallel, _, _, _>(
///     rows,
///     row_bytes,
///     || (),
///     |(), first_row, count| {
///         for row in first_row..first_row + count {
///             visited[row].fetch_add(1, Ordering::Relaxed);
///         }
///     },
/// );
/// assert!(visited.iter().all(|seen| seen.load(Ordering::Relaxed) == 1));
/// ```
#[track_caller]
pub fn for_each_unit_task_range_with<P, S, Init, F>(
    units: usize,
    unit_bytes: usize,
    init: Init,
    f: F,
) where
    P: ExecutionPolicy,
    Init: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, usize) + Send + Sync,
{
    let Some(UnitTaskPlan {
        per_task,
        tasks,
        parallel,
        ..
    }) = plan_unit_tasks::<P>(units, 1, unit_bytes)
    else {
        return;
    };
    let run_of = |task: usize| {
        let first = task * per_task;
        (first, per_task.min(units - first))
    };
    if !parallel {
        let mut state = init();
        for task in 0..tasks {
            let (first, count) = run_of(task);
            f(&mut state, first, count);
        }
        return;
    }
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |task| {
            let (first, count) = run_of(task);
            let mut state = init();
            f(&mut state, first, count);
        })
        .expect("moirai global executor: for_each_unit_task_range_with");
}
