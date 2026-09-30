//! Completed-task retention: which settled blocks the registry releases, and
//! the bounded sweep that releases them.

use std::{sync::atomic::Ordering, time::Duration, time::Instant};

use moirai_core::executor::CleanupConfig;

use super::directory::SweepWindow;
use super::registry::TaskRegistry;
use super::state::{Retirement, TASK_STATE_BLOCK_SIZE};

/// Directory entries one sweep step examines.
///
/// A step runs once per block the registry creates, i.e. once per
/// [`TASK_STATE_BLOCK_SIZE`] registrations, so the examination cost per spawn is
/// this budget over that block size. The budget exceeds one so the sweep keeps
/// pace with allocation even while it steps over already retired entries.
pub(super) const SWEEP_WINDOW: usize = 8;

/// How long, and how many, completed tasks the registry keeps observable.
///
/// Retention is block-granular: a block of 1,024 tasks is
/// released as one unit once **every** task in it has completed and released
/// its lifecycle token, and the block's newest completion is older than
/// `max_age` — or immediately, oldest first, while more than
/// `max_completed_tasks` worth of blocks are resident. A single long-running
/// task therefore keeps its block resident (about 73 KiB) however many later
/// tasks complete.
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

impl TaskRegistry {
    /// Advance the sweep by one bounded window.
    ///
    /// Called from the registration slow path, once per created block, so the
    /// reclamation work is proportional to allocation: memory can grow only as
    /// fast as tasks register, and each new block pays for examining a few old
    /// ones. A registry without a retention policy never sweeps.
    pub(super) fn sweep_step(&self) {
        let Some(policy) = self.retention else {
            return;
        };
        let cutoff = Instant::now().checked_sub(policy.max_age);
        let cursor = self.sweep_cursor.load(Ordering::Relaxed);
        let (next, _wrapped, _retired) =
            self.sweep_window(cursor, cutoff, Some(policy.max_resident_blocks()));
        self.sweep_cursor.store(next, Ordering::Relaxed);
    }

    /// Release every block whose tasks all completed at least `older_than` ago,
    /// returning how many blocks were released.
    ///
    /// The directory lock is taken per released block, never across the scan, so
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
        let mut cursor = 0;
        let mut retired = 0;
        loop {
            let (next, wrapped, released) = self.sweep_window(cursor, Some(cutoff), None);
            retired += released;
            if wrapped {
                return retired;
            }
            cursor = next;
        }
    }

    /// Examine one window of directory entries from `cursor`, retiring the
    /// settled ones.
    ///
    /// `cutoff` bounds completion age (`None`: no block is old enough), and
    /// `resident_cap` retires settled blocks oldest first regardless of age
    /// while more blocks than that are resident. Returns the cursor for the
    /// next window, whether this window reached the end of the directory, and
    /// the number of blocks retired.
    fn sweep_window(
        &self,
        cursor: usize,
        cutoff: Option<Instant>,
        resident_cap: Option<usize>,
    ) -> (usize, bool, usize) {
        let SweepWindow {
            next,
            wrapped,
            blocks,
            mut resident,
        } = self
            .blocks
            .read()
            .expect("task registry block directory is never poisoned")
            .window::<SWEEP_WINDOW>(cursor);

        let mut retired = 0;
        for (index, block) in blocks.into_iter().flatten() {
            let retirement = match (resident_cap, cutoff) {
                (Some(cap), _) if resident > cap => Retirement::Forced,
                (_, Some(cutoff)) => Retirement::CompletedBefore(cutoff),
                _ => continue,
            };
            if !block.is_settled(index == 0, retirement) {
                continue;
            }
            let released = self
                .blocks
                .write()
                .expect("task registry block directory is never poisoned")
                .retire(index, &block);
            // The block's storage is released here, after the directory lock.
            if released.is_some() {
                resident -= 1;
                retired += 1;
            }
        }
        (next, wrapped, retired)
    }
}
