//! Whole-unit tasks over three equally-long mutable buffers, split in lockstep.

use super::driver::drive_unit_tasks;
use crate::ops::shards::chunk_shards;
use crate::policy::ExecutionPolicy;

chunk_shards! {
    /// The triple unit-task operator's buffer set.
    struct UnitTripleShards { a: A, b: B, c: C }
}

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
    drive_unit_tasks::<P, _, _, _, _>(
        UnitTripleShards { a, b, c },
        unit_len,
        unit_bytes,
        "moirai global executor: for_each_unit_task_triple_mut_with",
        init,
        |state, first_unit, (run_a, run_b, run_c)| f(state, first_unit, run_a, run_b, run_c),
    );
}
