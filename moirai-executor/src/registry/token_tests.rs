#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! Lifecycle tokens against a concurrent retention sweep.
//!
//! A scheduler-bounded token holds no owner of its block, so the release store
//! that clears `token_active` is the token's last access: the registry may free
//! the block the instant the store is visible. These tests retire the last slot
//! of a full block while another thread sweeps, which is the interleaving Miri
//! needs to see a reference to the state that outlives the store.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Wake, Waker},
    thread,
    time::Duration,
};

use super::registry::TaskRegistry;
use super::state::TASK_STATE_BLOCK_SIZE;
use super::token::{SchedulerStateLease, StateLease, TaskLifecycleToken};
use crate::schedule::job::ScheduledJob;

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
                lifecycle.start(0).complete().notify();
            }
            Self::DropUnstarted => drop(lifecycle),
            Self::Cancel => {
                let Err(completion) = lifecycle.start_unless_cancelled(0) else {
                    panic!("cancelled lifecycle must not yield a running token");
                };
                completion.notify();
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

#[derive(Debug, Clone, Copy)]
enum Terminal {
    Complete,
    Cancel,
}

struct PanickingWake {
    calls: AtomicUsize,
}

impl PanickingWake {
    fn notify(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        panic!("waiter callback panic");
    }
}

impl Wake for PanickingWake {
    fn wake(self: Arc<Self>) {
        self.notify();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
    }
}

fn terminal_receipt_survives_retirement<L, F>(register: F, terminal: Terminal)
where
    L: StateLease,
    F: Fn(&TaskRegistry) -> (u64, TaskLifecycleToken<L>),
{
    let registry = TaskRegistry::new();
    for _ in 1..TASK_STATE_BLOCK_SIZE - 1 {
        let (_, lifecycle) = register(&registry);
        lifecycle.start(0).complete().notify();
    }
    let (task_id, lifecycle) = register(&registry);
    let wake = Arc::new(PanickingWake {
        calls: AtomicUsize::new(0),
    });
    let _waiter = registry
        .register_waker(task_id, &Waker::from(Arc::clone(&wake)))
        .expect("invariant: terminal test task is pending");

    let completion = match terminal {
        Terminal::Complete => lifecycle.start(0).complete(),
        Terminal::Cancel => {
            assert_eq!(
                registry.request_cancel(task_id),
                Some(super::registry::CancelOutcome::Requested)
            );
            lifecycle
                .start_unless_cancelled(0)
                .expect_err("the requested cancellation must skip execution")
        }
    };
    let metadata = registry.get_metadata(task_id).unwrap();
    assert_eq!(metadata.cancelled, matches!(terminal, Terminal::Cancel));
    assert_eq!(
        registry.with_state(task_id, super::state::TaskState::token_active),
        Some(false)
    );
    assert_eq!(wake.calls.load(Ordering::Relaxed), 0);
    assert_eq!(registry.cleanup_completed(Duration::ZERO), 1);
    assert!(registry.is_completed(task_id));
    assert!(registry.get_metadata(task_id).is_none());

    let panic = catch_unwind(AssertUnwindSafe(|| completion.notify()))
        .expect_err("the registered waiter must panic");
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"waiter callback panic"));
    assert_eq!(wake.calls.load(Ordering::Relaxed), 1);
}

fn drop_through_job<T: Send + 'static>(token: T) {
    drop(ScheduledJob::new(move |_| drop(token)));
}

/// Fill block 0 except its last slot, then retire that slot through a
/// scheduler-bounded token while another thread sweeps continuously.
fn retire_last_slot_during_sweep(retire: Retire) {
    let registry = Arc::new(TaskRegistry::new());
    // Id 0 is never issued, so ids `1..TASK_STATE_BLOCK_SIZE - 1` are the rest
    // of block 0 and the next id is its last slot.
    for _ in 1..TASK_STATE_BLOCK_SIZE - 1 {
        let (_, lifecycle) = registry.register_next_task();
        lifecycle.start(0).complete().notify();
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

#[test]
fn terminal_receipts_outlive_owned_and_scheduled_state() {
    for terminal in [Terminal::Complete, Terminal::Cancel] {
        terminal_receipt_survives_retirement(|registry| registry.register_next_task(), terminal);
        terminal_receipt_survives_retirement(
            |registry| {
                // SAFETY: the registry remains live until after the token and
                // detached completion receipt are consumed.
                unsafe { registry.register_next_scheduled_task() }
            },
            terminal,
        );
    }
}

#[test]
fn job_cleanup_contains_drop_notification_panics() {
    for started in [false, true] {
        let registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();
        let wake = Arc::new(PanickingWake {
            calls: AtomicUsize::new(0),
        });
        let _waiter = registry
            .register_waker(task_id, &Waker::from(Arc::clone(&wake)))
            .expect("invariant: drop test task is pending");

        if started {
            drop_through_job(lifecycle.start(0));
        } else {
            drop_through_job(lifecycle);
        }

        let metadata = registry.get_metadata(task_id).unwrap();
        assert!(metadata.cancelled);
        assert!(metadata.completed_at >= Some(metadata.created_at));
        assert_eq!(
            registry.with_state(task_id, super::state::TaskState::token_active),
            Some(false)
        );
        assert_eq!(wake.calls.load(Ordering::Relaxed), 1);
    }
}
