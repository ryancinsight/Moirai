#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! Lifecycle tokens against a concurrent retention sweep.
//!
//! A scheduler-bounded token holds no owner of its block, so the release store
//! that clears `token_active` is the token's last access: the registry may free
//! the block the instant the store is visible. These tests retire the last slot
//! of a full block while another thread sweeps, which is the interleaving Miri
//! needs to see a reference to the state that outlives the store.

use std::{sync::Arc, thread, time::Duration};

use super::registry::TaskRegistry;
use super::state::TASK_STATE_BLOCK_SIZE;
use super::token::{SchedulerStateLease, TaskLifecycleToken};

/// How a token gives up its slot.
#[derive(Debug, Clone, Copy)]
enum Retire {
    /// Start, then complete: `RunningTaskToken` releases the lease.
    Complete,
    /// Drop before start: `TaskLifecycleToken::drop` releases the lease.
    DropUnstarted,
    /// Honor a pending cancel request: `TaskLifecycleToken::cancel`.
    Cancel,
}

impl Retire {
    fn apply(self, lifecycle: TaskLifecycleToken<SchedulerStateLease>) {
        match self {
            Self::Complete => {
                lifecycle.start(0).complete();
            }
            Self::DropUnstarted => drop(lifecycle),
            Self::Cancel => {
                assert!(lifecycle.start_unless_cancelled(0).is_none());
            }
        }
    }
}

/// Sweep attempts the racing thread makes before giving up on catching the
/// retirement itself; a final sweep after the join retires the block regardless.
const SWEEP_ATTEMPTS: usize = if cfg!(miri) { 64 } else { 20_000 };

/// Rounds per retirement path. Each round fills a fresh block, which Miri
/// interprets slowly.
const ROUNDS: usize = if cfg!(miri) { 1 } else { 100 };

/// Fill block 0 except its last slot, then retire that slot through a
/// scheduler-bounded token while another thread sweeps continuously.
fn retire_last_slot_during_sweep(retire: Retire) {
    let registry = Arc::new(TaskRegistry::new());
    // Id 0 is never issued, so ids `1..TASK_STATE_BLOCK_SIZE - 1` are the rest
    // of block 0 and the next id is its last slot.
    for _ in 1..TASK_STATE_BLOCK_SIZE - 1 {
        let (_, lifecycle) = registry.register_next_task();
        lifecycle.start(0).complete();
    }
    // SAFETY: `registry` outlives the token: the token moves into the scoped
    // thread below, which joins before `registry` drops.
    let (last_id, last) = unsafe { registry.register_next_scheduled_task() };
    assert_eq!(last_id, (TASK_STATE_BLOCK_SIZE - 1) as u64);
    if matches!(retire, Retire::Cancel) {
        registry.request_cancel(last_id).unwrap();
    }

    let swept_while_racing = thread::scope(|scope| {
        let retiring = scope.spawn(move || retire.apply(last));
        let sweeping = scope.spawn(|| {
            let mut retired = 0;
            for _ in 0..SWEEP_ATTEMPTS {
                retired += registry.cleanup_completed(Duration::ZERO);
                if retired > 0 {
                    break;
                }
                thread::yield_now();
            }
            retired
        });
        retiring.join().unwrap();
        sweeping.join().unwrap()
    });

    let swept_after = registry.cleanup_completed(Duration::ZERO);
    assert_eq!(
        swept_while_racing + swept_after,
        1,
        "the settled block retires exactly once, whichever sweep sees it"
    );
    assert_eq!(registry.blocks.read().unwrap().resident(), 0);
    for id in 1..=last_id {
        assert!(registry.is_completed(id), "id {id}");
    }
}

#[test]
fn completing_the_last_scheduled_token_races_a_sweep_soundly() {
    for _ in 0..ROUNDS {
        retire_last_slot_during_sweep(Retire::Complete);
    }
}

#[test]
fn dropping_the_last_scheduled_token_races_a_sweep_soundly() {
    for _ in 0..ROUNDS {
        retire_last_slot_during_sweep(Retire::DropUnstarted);
    }
}

#[test]
fn cancelling_the_last_scheduled_token_races_a_sweep_soundly() {
    for _ in 0..ROUNDS {
        retire_last_slot_during_sweep(Retire::Cancel);
    }
}
