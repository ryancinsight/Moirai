//! The one driver the byte-sized unit-task operators share.
//!
//! [`drive_unit_tasks`] owns the skeleton — whole-unit assertion, byte-sized
//! plan, policy gate, sequential fallback, and the parallel dispatch that
//! reconstructs each task's disjoint run from its index — over any buffer set
//! implementing [`ChunkShards`]. It lives beside the plan it consumes rather
//! than in [`crate::ops::shards`] so the plan's crate-private internals stay
//! private to this module tree.

use super::layout::{assert_whole_units, plan_unit_tasks};
use crate::ops::shards::ChunkShards;
use crate::policy::ExecutionPolicy;
use moirai_executor::{SyncTask, global};

/// Drive a byte-sized unit-task pass over one buffer set.
///
/// `context` is appended to the executor error when the parallel branch cannot
/// be scheduled, so a failure names the entry point that raised it.
pub(super) fn drive_unit_tasks<P, B, S, Init, F>(
    buffers: B,
    unit_len: usize,
    unit_bytes: usize,
    context: &'static str,
    init: Init,
    f: F,
) where
    P: ExecutionPolicy,
    B: ChunkShards,
    Init: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, B::Chunk) + Send + Sync,
{
    let len = buffers.len();
    assert_whole_units(len, unit_len);
    let Some(plan) = plan_unit_tasks::<P>(len, unit_len, unit_bytes) else {
        return;
    };
    let views = buffers.split(plan.task_len);
    if !plan.parallel {
        let mut state = init();
        for task in 0..plan.tasks {
            // SAFETY: the plan divides the equal-length buffers, so every view
            // holds exactly `plan.tasks` runs and `task < plan.tasks` is in
            // bounds for each; each index is visited once by this loop, so no two
            // chunks alias.
            let chunk = unsafe { B::chunk(&views, task) };
            f(&mut state, task * plan.per_task, chunk);
        }
        return;
    }
    let (init, f) = (&init, &f);
    global()
        .for_each_indexed::<SyncTask, _>(plan.tasks, move |task| {
            // SAFETY: `for_each_indexed(plan.tasks, _)` visits each index in
            // `0..plan.tasks` exactly once — the contract `get_unchecked_chunk`
            // documents — and the plan divides the equal-length buffers into
            // exactly `plan.tasks` runs, so each index is in bounds for each
            // view. The buffers are distinct `&mut [_]` arguments that cannot
            // alias, and distinct indices name disjoint element ranges in each
            // buffer, so no two tasks form `&mut` to one element.
            let chunk = unsafe { B::chunk(&views, task) };
            let mut state = init();
            f(&mut state, task * plan.per_task, chunk);
        })
        .expect(context);
}
