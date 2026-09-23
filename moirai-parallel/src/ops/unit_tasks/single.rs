//! Whole-unit tasks over one mutable buffer, split into disjoint runs.

use super::layout::{UnitTaskPlan, assert_whole_units, plan_unit_tasks};
use crate::policy::ExecutionPolicy;
use melinoe::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_executor::{SyncTask, global};

/// Apply `f(state, first_unit, units)` to consecutive runs of whole units of
/// `data`, each run sized to about [`crate::UNIT_TASK_BYTES`] of work.
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
    let partitions = WriterShard::new(MelinoeCell::from_mut_slice(data)).par_chunks(task_len);
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |task| {
            // SAFETY: `for_each_indexed(tasks, _)` visits each task index exactly
            // once, and `partitions` holds exactly `tasks` runs, so `task` is in
            // bounds; distinct indices name disjoint element ranges, so no two
            // tasks form `&mut` to one element.
            let run = unsafe { partitions.get_unchecked_chunk(task) }.into_mut_slice();
            let mut state = init();
            f(&mut state, task * per_task, run);
        })
        .expect("moirai global executor: for_each_unit_task_mut_with");
}
