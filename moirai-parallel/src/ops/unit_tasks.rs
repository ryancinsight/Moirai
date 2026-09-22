//! Whole-unit tasks sized by the bytes each unit moves (ADR 0059).
//!
//! A unit is a lane, row or matrix: a fixed run of elements that one closure
//! call transforms together. How many units a task carries, and whether the
//! pass runs in parallel at all, depend on the bytes a unit moves — including
//! any input read beside it — not on its element count.

use crate::policy::ExecutionPolicy;
use melinoe::MelinoeCell;
use melinoe::region::WriterShard;
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

/// Apply `f(state, first_unit, units)` to consecutive runs of whole units of a
/// pass that addresses its own data, each run sized to about
/// [`UNIT_TASK_BYTES`] of work.
///
/// This is [`for_each_unit_task_mut_with`] for a pass whose units are not a
/// dense slice the runtime can split: a strided row walk, a tiled block pass,
/// a reduction that writes one output per axis index. The caller owns the
/// disjointness proof for whatever those indices address, exactly as it does
/// today; this owns the one decision the slice operators own — how many units
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

/// [`for_each_unit_task_mut_with`] over `K` buffers of one type, split in
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
