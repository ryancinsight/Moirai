//! Whole-unit tasks over `K` same-type buffers, split in lockstep: the fused
//! alternative to one pass per output.

use super::layout::{UnitTaskPlan, assert_whole_units, plan_unit_tasks};
use crate::policy::ExecutionPolicy;
use melinoe::MelinoeCell;
use melinoe::region::WriterShard;
use moirai_executor::{SyncTask, global};

/// [`crate::for_each_unit_task_mut_with`] over `K` buffers of one type, split in
/// lockstep.
///
/// The pair and triple forms take their buffers by separate type, which suits
/// two or three outputs of different types. A pass whose outputs share a type
/// and whose count is a property of the problem -- the six fields a
/// velocity-Verlet kick-and-drift writes, say -- takes them as one array
/// instead, and stays one pass rather than becoming `K` of them. Splitting a
/// fused pass into one call per output multiplies the parallel regions, and
/// each region's wake costs more than the traffic such a fusion saves.
///
/// Every buffer must have the same length, and that length must be whole
/// units. `f` receives each task's runs in the order the buffers were given.
///
/// # Examples
///
/// ```
/// use moirai_parallel::{for_each_unit_task_many_mut_with, WorkBytes};
///
/// let (mut xs, mut ys) = (vec![0_u64; 8], vec![1_u64; 8]);
/// for_each_unit_task_many_mut_with::<WorkBytes<{ 1 << 20 }>, _, _, _, _, 2>(
///     [&mut xs, &mut ys],
///     4,
///     64,
///     || (),
///     |(), _first_unit, [xs, ys]| {
///         for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
///             *x += 2;
///             *y += *x;
///         }
///     },
/// );
/// assert!(xs.iter().all(|&x| x == 2) && ys.iter().all(|&y| y == 3));
/// ```
#[track_caller]
pub fn for_each_unit_task_many_mut_with<P, T, S, Init, F, const K: usize>(
    outputs: [&mut [T]; K],
    unit_len: usize,
    unit_bytes: usize,
    init: Init,
    f: F,
) where
    P: ExecutionPolicy,
    T: Send,
    Init: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, [&mut [T]; K]) + Send + Sync,
{
    let Some(first) = outputs.first() else {
        return;
    };
    let len = first.len();
    assert_whole_units(len, unit_len);
    for buffer in &outputs {
        assert_eq!(
            buffer.len(),
            len,
            "many unit tasks need buffers of one length",
        );
    }
    let Some(UnitTaskPlan {
        per_task,
        task_len,
        tasks,
        parallel,
    }) = plan_unit_tasks::<P>(len, unit_len, unit_bytes)
    else {
        return;
    };
    if !parallel {
        let mut state = init();
        let mut runs = outputs.map(|buffer| buffer.chunks_mut(task_len));
        for task in 0..tasks {
            let chunk: [&mut [T]; K] = core::array::from_fn(|index| {
                runs[index]
                    .next()
                    .expect("invariant: every buffer holds the planned run count")
            });
            f(&mut state, task * per_task, chunk);
        }
        return;
    }
    let partitions = outputs
        .map(|buffer| WriterShard::new(MelinoeCell::from_mut_slice(buffer)).par_chunks(task_len));
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(tasks, move |task| {
            // SAFETY: each task index is visited exactly once and is in bounds
            // for every partition view -- they hold the same number of runs,
            // since the buffers are asserted equal in length. Distinct indices
            // name disjoint element ranges in each buffer, so no two tasks
            // alias, and within a task the `K` runs come from `K` distinct
            // buffers.
            let chunk: [&mut [T]; K] = core::array::from_fn(|index| unsafe {
                partitions[index].get_unchecked_chunk(task).into_mut_slice()
            });
            let mut state = init();
            f(&mut state, task * per_task, chunk);
        })
        .expect("moirai global executor: for_each_unit_task_many_mut_with");
}
