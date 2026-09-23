//! Whole-unit tasks over two equally-long mutable buffers, split in lockstep.

use super::layout::{UnitTaskPlan, assert_whole_units, plan_unit_tasks};
use crate::policy::ExecutionPolicy;
use melinoe::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_executor::{SyncTask, global};

/// Apply `f(state, first_unit, a_units, b_units)` to aligned runs of whole
/// units of two mutable buffers, each run sized to about [`crate::UNIT_TASK_BYTES`]
/// of work.
///
/// This is [`crate::for_each_unit_task_mut_with`] for a pass that writes two fields
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
    let a_partitions = WriterShard::new(MelinoeCell::from_mut_slice(a)).par_chunks(task_len);
    let b_partitions = WriterShard::new(MelinoeCell::from_mut_slice(b)).par_chunks(task_len);
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |task| {
            // SAFETY: each task index is visited exactly once and is in bounds for
            // both partition views (they hold the same number of runs, since `a`
            // and `b` have equal length); distinct indices name disjoint element
            // ranges in each buffer, so no two tasks alias.
            let run_a = unsafe { a_partitions.get_unchecked_chunk(task) }.into_mut_slice();
            let run_b = unsafe { b_partitions.get_unchecked_chunk(task) }.into_mut_slice();
            let mut state = init();
            f(&mut state, task * per_task, run_a, run_b);
        })
        .expect("moirai global executor: for_each_unit_task_pair_mut_with");
}
