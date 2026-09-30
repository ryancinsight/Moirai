//! Completed-task retention: which settled blocks the registry releases, and
//! the bounded sweep that releases them.

use std::{sync::Arc, time::Duration, time::Instant};

use moirai_core::executor::CleanupConfig;

use super::directory::SweepWindow;
use super::registry::TaskRegistry;
use super::state::{Retirement, TASK_STATE_BLOCK_SIZE, TaskStateBlock};

/// Queued blocks one sweep step examines.
///
/// A step runs once per block the registry creates, i.e. once per
/// [`TASK_STATE_BLOCK_SIZE`] registrations, so the examination cost per spawn is
/// this budget over that block size. The budget exceeds one so the sweep laps
/// the resident blocks faster than allocation adds to them.
pub(super) const SWEEP_WINDOW: usize = 8;

/// How long, and how many, completed tasks the registry keeps observable.
///
/// Retention is block-granular: a block of 1,024 tasks is released as one unit
/// once **every** task in it has completed and released its lifecycle token,
/// and either the block's newest completion is older than `max_age` or more
/// than `max_completed_tasks` worth of blocks are resident. The cap holds
/// regardless of age, and it counts resident blocks, so a block pinned by a
/// running task and the block being filled count toward it.
///
/// A single long-running task keeps its own block resident (about 73 KiB)
/// however many later tasks complete, and no other. The sweep visits resident
/// blocks in a queue, eight per created block, so a settled block waits at
/// most one lap of that queue. With `k` the larger of the cap in blocks and the
/// pinned blocks plus the one being filled, the resident count stays at most
/// `M + ceil(M / 8)` for the least `M` with `M >= k + ceil(M / 8)`, about 9/7
/// of `k`, however many blocks are created.
///
/// After release, [`TaskRegistry::is_completed`] stays `true` and a cancel
/// request still reports "already completed"; only per-task metadata and
/// statistics are gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Completed-task metadata older than this may be released.
    pub max_age: Duration,
    /// Most completed tasks kept regardless of age, rounded up to whole blocks.
    pub max_completed_tasks: usize,
}

impl RetentionPolicy {
    /// The policy an executor derives from its cleanup configuration, or
    /// `None` when automatic cleanup is disabled.
    #[must_use]
    pub fn from_cleanup(config: &CleanupConfig) -> Option<Self> {
        config.enable_automatic_cleanup.then_some(Self {
            max_age: config.task_retention_duration,
            max_completed_tasks: config.max_retained_tasks,
        })
    }

    /// Resident blocks above which settled blocks retire regardless of age: the
    /// retained completed tasks in whole blocks, plus the block being filled.
    pub(super) fn max_resident_blocks(&self) -> usize {
        self.max_completed_tasks
            .div_ceil(TASK_STATE_BLOCK_SIZE)
            .saturating_add(1)
    }
}

/// What a sweep decided about one examined block, before the directory lock is
/// retaken.
#[derive(Clone, Copy)]
enum Verdict {
    /// The block is pinned, partly unregistered, or younger than the window
    /// while the resident cap is not exceeded.
    Keep,
    /// Every task completed before the age cutoff.
    Expired,
    /// The block is settled but young: it retires only while more blocks than
    /// the cap are resident when the sweep commits.
    OverCap,
}

impl Verdict {
    /// `forceable` is whether the resident cap can retire a settled block.
    fn of(
        block: &TaskStateBlock,
        first_slot_unissued: bool,
        cutoff: Option<Instant>,
        forceable: bool,
    ) -> Self {
        let expired = || {
            cutoff.is_some_and(|cutoff| {
                block.is_settled(first_slot_unissued, Retirement::CompletedBefore(cutoff))
            })
        };
        if forceable && !block.is_settled(first_slot_unissued, Retirement::Forced) {
            Self::Keep
        } else if expired() {
            Self::Expired
        } else if forceable {
            Self::OverCap
        } else {
            Self::Keep
        }
    }
}

impl TaskRegistry {
    /// Advance the sweep by one bounded window.
    ///
    /// Called from the registration slow path, once per created block, so the
    /// reclamation work is proportional to allocation: memory can grow only as
    /// fast as tasks register, and each new block pays for examining a few
    /// queued ones. That is at most [`SWEEP_WINDOW`] blocks, each up to two
    /// settledness scans of at most [`TASK_STATE_BLOCK_SIZE`] slots (the cap
    /// test, then the age test), plus two brief directory write locks, per
    /// [`TASK_STATE_BLOCK_SIZE`] registrations; the other registrations do none
    /// of it. A registry without a retention policy never sweeps.
    pub(super) fn sweep_step(&self) {
        let Some(policy) = self.retention else {
            return;
        };
        let cutoff = Instant::now().checked_sub(policy.max_age);
        self.sweep_window(cutoff, policy.max_resident_blocks());
    }

    /// Release every block whose tasks all completed at least `older_than` ago,
    /// returning how many blocks were released.
    ///
    /// Every resident block is examined, including any that a concurrent
    /// automatic sweep has checked out; that sweep finds such a block already
    /// retired and moves on. The directory lock is held only to list the
    /// resident blocks and to retire the settled ones, never across a scan, so
    /// registrations proceed throughout. Automatic retention makes calling this
    /// unnecessary for an executor; it serves callers that manage a registry
    /// directly.
    pub fn cleanup_completed(&self, older_than: Duration) -> usize {
        // `Instant - Duration` panics when the result predates the platform's
        // clock origin, which a caller-supplied retention window longer than the
        // process uptime reaches. No recorded completion can be older than a
        // cutoff before the clock started, so that case is an empty sweep.
        let Some(cutoff) = Instant::now().checked_sub(older_than) else {
            return 0;
        };
        let resident: Vec<_> = self
            .blocks
            .read()
            .expect("task registry block directory is never poisoned")
            .resident_indexed()
            .map(|(index, block)| (index, Arc::clone(block)))
            .collect();
        let settled: Vec<usize> = resident
            .iter()
            .filter(|(index, block)| {
                block.is_settled(*index == 0, Retirement::CompletedBefore(cutoff))
            })
            .map(|&(index, _)| index)
            .collect();
        // The retired blocks drop with `resident`, after the directory lock.
        let released = self
            .blocks
            .write()
            .expect("task registry block directory is never poisoned")
            .cleanup(&settled);
        released.len()
    }

    /// Check out the next window of queued blocks, retire the settled ones, and
    /// requeue the rest.
    ///
    /// `cutoff` bounds completion age (`None`: no block is old enough), and
    /// settled blocks retire regardless of age while more than `resident_cap`
    /// blocks are resident. Whether the cap is exceeded is decided under the
    /// directory lock against the live count, so concurrent sweeps cannot each
    /// retire down from a stale copy of it; only the settledness scans, which
    /// cost up to a block's slots each, run outside the lock.
    fn sweep_window(&self, cutoff: Option<Instant>, resident_cap: usize) {
        let SweepWindow { blocks, resident } = self
            .blocks
            .write()
            .expect("task registry block directory is never poisoned")
            .check_out::<SWEEP_WINDOW>();
        let forceable = resident > resident_cap;
        let judged = blocks.map(|slot| {
            slot.map(|(index, block)| {
                let verdict = Verdict::of(&block, index == 0, cutoff, forceable);
                (index, block, verdict)
            })
        });

        // Released blocks drop with `judged`, after the directory lock.
        let mut released = [const { None }; SWEEP_WINDOW];
        let mut directory = self
            .blocks
            .write()
            .expect("task registry block directory is never poisoned");
        for (slot, (index, _, verdict)) in released.iter_mut().zip(judged.iter().flatten()) {
            let retire = match verdict {
                Verdict::Expired => true,
                Verdict::OverCap => directory.resident() > resident_cap,
                Verdict::Keep => false,
            };
            if retire {
                *slot = directory.retire(*index);
            } else {
                directory.requeue(*index);
            }
        }
        drop(directory);
    }
}
