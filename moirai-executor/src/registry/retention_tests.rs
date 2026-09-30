#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! Block retirement: what a released block still answers, what pins a block,
//! and that the automatic sweep keeps resident storage bounded.

use std::{sync::Arc, time::Duration};

use super::directory::BlockLookup;
use super::registry::{CancelOutcome, TaskRegistry};
use super::retention::{RetentionPolicy, SWEEP_WINDOW};
use super::state::{TASK_STATE_BLOCK_SIZE, task_location};
use super::token::RunningTaskToken;

/// Register `count` tasks and run each to completion, returning their ids.
fn complete_tasks(registry: &TaskRegistry, count: usize) -> Vec<u64> {
    (0..count)
        .map(|_| {
            let (id, lifecycle) = registry.register_next_task();
            lifecycle.start(0).complete();
            id
        })
        .collect()
}

/// Ids `1..=2 * TASK_STATE_BLOCK_SIZE - 1` fill block 0 (id 0 is never issued)
/// and block 1 completely.
const TWO_BLOCKS: usize = 2 * TASK_STATE_BLOCK_SIZE - 1;

#[test]
fn settled_blocks_release_state_but_still_answer_completed() {
    let registry = TaskRegistry::new();
    let ids = complete_tasks(&registry, TWO_BLOCKS);
    assert_eq!(registry.completed_count(), TWO_BLOCKS);

    let retired = registry.cleanup_completed(Duration::ZERO);

    assert_eq!(retired, 2);
    assert_eq!(registry.completed_count(), 0);
    for id in [ids[0], ids[TASK_STATE_BLOCK_SIZE], ids[TWO_BLOCKS - 1]] {
        assert!(registry.get_metadata(id).is_none(), "id {id}");
        assert!(registry.is_completed(id), "id {id}");
        assert_eq!(
            registry.request_cancel(id),
            Some(CancelOutcome::AlreadyCompleted),
            "id {id}"
        );
    }
    let directory = registry.blocks.read().unwrap();
    assert_eq!(directory.resident(), 0);
    assert_eq!(
        directory.span(),
        0,
        "retiring the oldest blocks collapses the directory into its watermark"
    );
}

#[test]
fn completion_tells_unknown_running_and_released_apart_in_one_read() {
    let registry = TaskRegistry::new();
    let (running_id, running) = registry.register_next_task();
    let running = running.start(0);
    let released = complete_tasks(&registry, TWO_BLOCKS - 1);
    let beyond = (2 * TASK_STATE_BLOCK_SIZE + 5) as u64;

    assert_eq!(registry.completion(running_id), Some(false));
    assert_eq!(registry.completion(released[0]), Some(true));
    assert_eq!(registry.completion(beyond), None);

    registry.cleanup_completed(Duration::ZERO);

    assert!(registry.get_metadata(released[TWO_BLOCKS - 2]).is_none());
    assert_eq!(registry.completion(released[TWO_BLOCKS - 2]), Some(true));
    assert_eq!(registry.completion(running_id), Some(false));
    assert_eq!(registry.completion(beyond), None);
    running.complete();
}

#[test]
fn unregistered_ids_stay_unknown_after_retirement() {
    let registry = TaskRegistry::new();
    complete_tasks(&registry, TWO_BLOCKS);
    registry.cleanup_completed(Duration::ZERO);

    let beyond = (2 * TASK_STATE_BLOCK_SIZE + 5) as u64;
    assert!(!registry.is_completed(beyond));
    assert_eq!(registry.request_cancel(beyond), None);
}

#[test]
fn running_task_pins_its_block_and_only_its_block() {
    let registry = TaskRegistry::new();
    let (pinned_id, pinned) = registry.register_next_task();
    let pinned = pinned.start(0);
    // The rest of block 0 and all of block 1 complete around the pinned task.
    complete_tasks(&registry, TWO_BLOCKS - 1);

    assert_eq!(registry.cleanup_completed(Duration::ZERO), 1);
    assert!(registry.get_metadata(pinned_id).is_some());
    assert!(!registry.is_completed(pinned_id));
    {
        let directory = registry.blocks.read().unwrap();
        assert_eq!(directory.resident(), 1);
        assert_eq!(
            directory.span(),
            2,
            "block 1 retired above the pinned block 0, so it costs one entry"
        );
        assert!(matches!(directory.lookup(1), BlockLookup::Retired));
    }

    pinned.complete();
    assert_eq!(registry.cleanup_completed(Duration::ZERO), 1);
    assert_eq!(registry.blocks.read().unwrap().span(), 0);
    assert!(registry.is_completed(pinned_id));
}

#[test]
fn issued_id_without_registration_keeps_a_finished_block() {
    let registry = TaskRegistry::new();
    let mut unregistered = None;
    for position in 1..TASK_STATE_BLOCK_SIZE {
        let id = registry.issue_id();
        if position == 512 {
            unregistered = Some(id);
        } else {
            registry.register_owned(id).start(0).complete();
        }
    }

    assert_eq!(
        registry.cleanup_completed(Duration::ZERO),
        0,
        "an issued id whose registration has not run must not be retired under"
    );

    registry
        .register_owned(unregistered.unwrap())
        .start(0)
        .complete();
    assert_eq!(registry.cleanup_completed(Duration::ZERO), 1);
}

#[test]
fn blocks_younger_than_the_window_are_kept() {
    let registry = TaskRegistry::new();
    complete_tasks(&registry, TWO_BLOCKS);

    assert_eq!(registry.cleanup_completed(Duration::from_secs(3600)), 0);
    assert_eq!(registry.completed_count(), TWO_BLOCKS);
    // A window longer than the process has run cannot have elapsed for any task.
    assert_eq!(registry.cleanup_completed(Duration::MAX), 0);
}

/// With a cap of zero completed tasks, every settled block retires at the
/// first sweep that reaches it, so resident storage tracks only what the
/// sweep has not yet visited.
#[test]
fn automatic_sweep_bounds_resident_blocks_under_a_task_cap() {
    let registry = TaskRegistry::with_retention(RetentionPolicy {
        max_age: Duration::from_secs(3600),
        max_completed_tasks: 0,
    });

    let blocks = 40;
    let ids = complete_tasks(&registry, blocks * TASK_STATE_BLOCK_SIZE);

    let directory = registry.blocks.read().unwrap();
    assert!(
        directory.resident() <= 3,
        "the cap keeps one settled block beside the block being filled; resident {}",
        directory.resident()
    );
    drop(directory);
    for id in ids {
        assert!(registry.is_completed(id), "id {id}");
    }
}

#[test]
fn automatic_sweep_honors_the_age_window() {
    let registry = TaskRegistry::with_retention(RetentionPolicy {
        max_age: Duration::from_secs(3600),
        max_completed_tasks: usize::MAX,
    });

    complete_tasks(&registry, 10 * TASK_STATE_BLOCK_SIZE);

    assert_eq!(
        registry.blocks.read().unwrap().resident(),
        11,
        "nothing completed an hour ago, so no block retires"
    );
}

#[test]
fn concurrent_producers_and_sweeps_keep_every_task_observable_as_completed() {
    const PRODUCERS: usize = 4;
    const PER_PRODUCER: usize = 6 * TASK_STATE_BLOCK_SIZE;

    let registry = Arc::new(TaskRegistry::with_retention(RetentionPolicy {
        max_age: Duration::ZERO,
        max_completed_tasks: 0,
    }));

    let mut ids = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..PRODUCERS)
            .map(|_| {
                let registry = Arc::clone(&registry);
                scope.spawn(move || complete_tasks(&registry, PER_PRODUCER))
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });

    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), PRODUCERS * PER_PRODUCER);
    for id in &ids {
        assert!(registry.is_completed(*id), "id {id}");
    }

    // Every block but the partially filled last one settles, and one final pass
    // retires whatever the concurrent sweeps had not reached.
    registry.cleanup_completed(Duration::ZERO);
    assert_eq!(registry.blocks.read().unwrap().resident(), 1);
}

/// Least resident-block bound `M` satisfying `M >= kept + ceil(M / W)`, plus one
/// lap of growth: `M + ceil(M / W)`.
///
/// `W` is [`SWEEP_WINDOW`]. The sweep examines `W` queued blocks per created
/// block, so a lap over `R` queued blocks takes at most `ceil(R / W)` creations.
/// At the end of a lap either every settled block that existed at its start was
/// examined while over the cap and retired, leaving the `pinned` blocks that
/// cannot retire plus the blocks created during the lap, or the resident count
/// fell to the cap `cap_blocks` at some point, from which it grew by at most the
/// lap's creations. So a lap-boundary count `R <= M` gives a next boundary
/// count `<= max(cap_blocks, pinned) + ceil(M / W)`, hence `M` is a fixed point,
/// and the count inside a lap exceeds a boundary by at most `ceil(M / W)`.
/// Neither the number of blocks created nor how long a block stays pinned
/// appears.
fn resident_bound(cap_blocks: usize, pinned: usize) -> usize {
    let kept = cap_blocks.max(pinned);
    let mut lap = kept;
    while lap < kept + lap.div_ceil(SWEEP_WINDOW) {
        lap += 1;
    }
    lap + lap.div_ceil(SWEEP_WINDOW)
}

/// Complete every task of `total_blocks` blocks except the first task of each of
/// the first `pinned_blocks` blocks, which stays running. Returns the running
/// tasks and the most blocks resident after any block was created.
fn churn_around_pinned_blocks(
    registry: &TaskRegistry,
    pinned_blocks: usize,
    total_blocks: usize,
) -> (Vec<(u64, RunningTaskToken)>, usize) {
    let mut held = Vec::new();
    let mut peak = 0;
    for _ in 1..total_blocks * TASK_STATE_BLOCK_SIZE {
        let (id, lifecycle) = registry.register_next_task();
        let running = lifecycle.start(0);
        let (block, slot) = task_location(id);
        if block < pinned_blocks && slot == usize::from(block == 0) {
            held.push((id, running));
        } else {
            running.complete();
        }
        if slot == TASK_STATE_BLOCK_SIZE - 1 {
            peak = peak.max(registry.blocks.read().unwrap().resident());
        }
    }
    (held, peak)
}

/// Blocks created by the pinned-sweep test. Before the sweep queued only
/// resident blocks, 1,000 blocks around one pin left 102 resident with a cap of
/// one block, growing linearly with the block count.
const BLOCKS_CREATED: usize = 1000;

/// Blocks pinned by long-running tasks cost their own storage and a lag that
/// depends on the cap and the pin count, never on how many blocks were created.
#[test]
fn pinned_blocks_do_not_stretch_the_sweep_lap() {
    // (retained-task cap, pinned blocks): one pin at the directory base, the
    // case where retired entries above it lengthen the directory span, and
    // several pins, where pinned entries crowd the sweep window.
    for (max_completed_tasks, pinned_blocks) in [(0, 1), (TASK_STATE_BLOCK_SIZE, 1), (0, 12)] {
        let policy = RetentionPolicy {
            max_age: Duration::from_secs(3600),
            max_completed_tasks,
        };
        let registry = TaskRegistry::with_retention(policy);

        let (held, peak) = churn_around_pinned_blocks(&registry, pinned_blocks, BLOCKS_CREATED);

        // The block being filled cannot retire either.
        let bound = resident_bound(policy.max_resident_blocks(), pinned_blocks + 1);
        assert!(
            peak <= bound,
            "cap {max_completed_tasks}, {pinned_blocks} pinned: peak resident blocks {peak} exceeds {bound}"
        );
        for (id, _) in &held {
            assert!(registry.get_metadata(*id).is_some(), "pinned id {id}");
            assert!(!registry.is_completed(*id), "pinned id {id}");
        }
        assert_eq!(held.len(), pinned_blocks);

        for (_, running) in held {
            running.complete();
        }
        registry.cleanup_completed(Duration::ZERO);
        let directory = registry.blocks.read().unwrap();
        assert_eq!(directory.resident(), 0, "cap {max_completed_tasks}");
        assert_eq!(directory.span(), 0, "cap {max_completed_tasks}");
    }
}
