//! Whole-unit tasks over two equally-long mutable buffers, split in lockstep.

use super::driver::drive_unit_tasks;
use crate::ops::shards::chunk_shards;
use crate::policy::ExecutionPolicy;

chunk_shards! {
    /// The paired unit-task operator's buffer set.
    struct UnitPairShards { a: A, b: B }
}

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
    assert_eq!(
        a.len(),
        b.len(),
        "paired unit tasks need buffers of one length",
    );
    drive_unit_tasks::<P, _, _, _, _>(
        UnitPairShards { a, b },
        unit_len,
        unit_bytes,
        "moirai global executor: for_each_unit_task_pair_mut_with",
        init,
        |state, first_unit, (run_a, run_b)| f(state, first_unit, run_a, run_b),
    );
}
