//! Whole-unit tasks over three equally-long mutable buffers, split in lockstep.

use super::layout::{UnitTaskPlan, assert_whole_units, plan_unit_tasks};
use crate::policy::ExecutionPolicy;
use melinoe::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_executor::{SyncTask, global};

/// Apply `f(state, first_unit, a_units, b_units, c_units)` to aligned runs of
/// whole units of three mutable buffers, each run sized to about
/// [`crate::UNIT_TASK_BYTES`] of work.
///
/// This is [`crate::for_each_unit_task_pair_mut_with`] for a pass that writes three
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
    let a_partitions = WriterShard::new(MelinoeCell::from_mut_slice(a)).par_chunks(task_len);
    let b_partitions = WriterShard::new(MelinoeCell::from_mut_slice(b)).par_chunks(task_len);
    let c_partitions = WriterShard::new(MelinoeCell::from_mut_slice(c)).par_chunks(task_len);
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |task| {
            // SAFETY: each task index is visited exactly once and is in bounds for
            // every partition view (they hold the same number of runs, since all
            // three buffers have equal length); distinct indices name disjoint
            // element ranges in each buffer, so no two tasks alias.
            let run_a = unsafe { a_partitions.get_unchecked_chunk(task) }.into_mut_slice();
            let run_b = unsafe { b_partitions.get_unchecked_chunk(task) }.into_mut_slice();
            let run_c = unsafe { c_partitions.get_unchecked_chunk(task) }.into_mut_slice();
            let mut state = init();
            f(&mut state, task * per_task, run_a, run_b, run_c);
        })
        .expect("moirai global executor: for_each_unit_task_triple_mut_with");
}
