//! Evidence that a chain containing a converted adapter keeps the source's shards.
//!
//! A value-equality test alone cannot distinguish the two shapes this change
//! separates: the superseded `drive` collected the whole logical stream into one
//! vector and re-split *that*, which returns the same values as driving the
//! source directly. What differs is whether the source's shards survive the
//! adapter, so these tests observe the shards themselves.
//!
//! Two independent signals, because each has a limitation the other does not:
//!
//! - **Distinct worker threads.** The direct meaning of "this chain shards": the
//!   adapter's own closure records which thread ran it. This is scheduler-visible
//!   evidence, and `drive_split` runs a branch on the caller when admission is
//!   refused, so a shutdown or a saturated queue can legitimately collapse it to
//!   one thread. The suite warms the executor before measuring.
//! - **Allocation count.** A property of the code shape rather than of a run, on
//!   the model established by `parallel_terminal_allocations`. The collect shape
//!   allocated the whole logical stream ahead of any work; driving the base
//!   allocates per shard.
//!
//! Every case runs above `PARALLEL_DRIVE_THRESHOLD` (1024). Below it a source
//! never splits at all, so a chain measured there exercises the sequential
//! fallback and proves nothing about the converted path — the coverage gap the
//! terminal rework found in the pre-existing tests.

use moirai_iter::parallel::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::ThreadId;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the layout it was
// given, so the allocator contract is exactly the system allocator's. The
// counter is a `Relaxed` atomic add on a separate static and imposes no ordering
// requirement of its own.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Sixty-four times the 1024-element dispatch threshold, so the drive splits to
/// many shards rather than to the one or two a threshold-adjacent length gives.
const LEN: usize = 65_536;

/// Allocations permitted per driven chain.
///
/// One eighth of the element count, the bound
/// `parallel_terminal_allocations` derives: the collect shape allocated at
/// least once per element for the materialized stream, so it exceeds this by
/// more than an order of magnitude, while driving the base allocates per shard
/// — `LEN / 1024` of them. The gap is wide enough that the bound measures the
/// code shape rather than the allocator's exact behaviour on a given run.
const ALLOCATION_BUDGET: usize = LEN / 8;

fn source() -> Vec<u64> {
    (0..LEN as u64).collect()
}

/// Threads that ran a recorded closure.
#[derive(Default)]
struct Workers {
    seen: Mutex<HashSet<ThreadId>>,
}

impl Workers {
    fn record(&self) {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(std::thread::current().id());
    }

    fn count(&self) -> usize {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

/// Assert that `chain` ran its recorded closure on more than one thread.
///
/// `chain` is run once to warm the executor's workers and their queues before
/// the observed run, on the model of `parallel_terminal_allocations`.
fn assert_shards<T>(adapter: &str, chain: impl Fn(&Workers) -> T) {
    let warm = Workers::default();
    let _ = chain(&warm);

    let observed = Workers::default();
    let _ = chain(&observed);

    let workers = observed.count();
    assert!(
        workers > 1,
        "a chain containing `{adapter}` over {LEN} elements ran on {workers} thread(s); \
         the source's shards did not survive the adapter"
    );
}

/// Run `operation` once warm, then count the allocations of a second call.
fn allocations_of<T>(operation: impl Fn() -> T) -> (T, usize) {
    let _warm = operation();

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let value = operation();
    let after = ALLOCATIONS.load(Ordering::Relaxed);

    (value, after.saturating_sub(before))
}

fn assert_within_budget(adapter: &str, allocations: usize) {
    assert!(
        allocations <= ALLOCATION_BUDGET,
        "a chain containing `{adapter}` over {LEN} elements made {allocations} allocations, \
         above the {ALLOCATION_BUDGET} budget; the logical stream is being materialized \
         before the split"
    );
}

#[test]
fn filter_map_chain_shards_and_preserves_values() {
    let data = source();
    let expected: u64 = data
        .iter()
        .filter_map(|value| (value % 3 == 0).then_some(value * 2))
        .sum();

    assert_shards("filter_map", |workers| {
        data.par_iter()
            .filter_map(|value| {
                workers.record();
                (value % 3 == 0).then_some(value * 2)
            })
            .sum::<u64>()
    });

    let (total, allocations) = allocations_of(|| {
        data.par_iter()
            .filter_map(|value| (value % 3 == 0).then_some(value * 2))
            .sum::<u64>()
    });

    assert_eq!(total, expected);
    assert_within_budget("filter_map", allocations);
}

#[test]
fn filter_map_chain_preserves_logical_order_across_shards() {
    let data = source();
    let expected: Vec<u64> = data
        .iter()
        .filter_map(|value| (value % 3 == 0).then_some(value * 2))
        .collect();

    let collected: Vec<u64> = data
        .par_iter()
        .filter_map(|value| (value % 3 == 0).then_some(value * 2))
        .collect();

    assert_eq!(collected, expected);
}

#[test]
fn flat_map_chain_shards_and_preserves_values() {
    let data = source();
    let expected: u64 = data.iter().flat_map(|value| [value % 5, value % 7]).sum();

    assert_shards("flat_map", |workers| {
        data.par_iter()
            .flat_map(|value| {
                workers.record();
                [value % 5, value % 7]
            })
            .sum::<u64>()
    });

    let (total, allocations) = allocations_of(|| {
        data.par_iter()
            .flat_map(|value| [value % 5, value % 7])
            .sum::<u64>()
    });

    assert_eq!(total, expected);
    assert_within_budget("flat_map", allocations);
}

#[test]
fn flat_map_chain_preserves_flattened_order_across_shards() {
    let data = source();
    let expected: Vec<u64> = data
        .iter()
        .flat_map(|value| [value % 5, value % 7])
        .collect();

    let collected: Vec<u64> = data
        .par_iter()
        .flat_map(|value| [value % 5, value % 7])
        .collect();

    assert_eq!(collected, expected);
}

/// Groups in the nested source.
///
/// Above the 1024-element dispatch threshold, because `Flatten`'s base is the
/// *nested* stream: it is the group count, not the flattened length, that
/// decides whether the source splits at all. A few long groups would flatten to
/// `LEN` items and still never split.
const GROUPS: usize = 2_048;

/// Items per group, chosen so the flattened length is `LEN`.
const GROUP_LEN: usize = LEN / GROUPS;

/// Nested source of `GROUPS` groups whose flattened length is `LEN`.
fn nested_source() -> Vec<Vec<u64>> {
    (0..GROUPS as u64)
        .map(|group| {
            (0..GROUP_LEN as u64)
                .map(|offset| group * GROUP_LEN as u64 + offset)
                .collect()
        })
        .collect()
}

#[test]
fn flatten_chain_shards_and_preserves_values() {
    let nested = nested_source();
    let expected: u64 = nested.iter().flatten().sum();

    assert_shards("flatten", |workers| {
        nested
            .clone()
            .into_par_iter()
            .flatten()
            .map(|value| {
                workers.record();
                value
            })
            .sum::<u64>()
    });

    let (total, allocations) =
        allocations_of(|| nested.clone().into_par_iter().flatten().sum::<u64>());

    assert_eq!(total, expected);
    // The clone inside the measured closure is one allocation per group plus
    // the outer vector, so `GROUPS + 1`; the flattening itself adds none beyond
    // the per-shard cost. Both together stay inside the budget, which the
    // collect shape's one-allocation-per-flattened-item could not.
    assert_within_budget("flatten", allocations);
}

#[test]
fn flatten_chain_preserves_nested_order_across_shards() {
    let nested = nested_source();
    let expected: Vec<u64> = nested.iter().flatten().copied().collect();

    let collected: Vec<u64> = nested.clone().into_par_iter().flatten().collect();

    assert_eq!(collected, expected);
}

#[test]
fn update_chain_shards_and_preserves_values() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value.wrapping_mul(3)).sum();

    assert_shards("update", |workers| {
        data.clone()
            .into_par_iter()
            .update(|value| {
                workers.record();
                *value = value.wrapping_mul(3);
            })
            .sum::<u64>()
    });

    let (total, allocations) = allocations_of(|| {
        data.clone()
            .into_par_iter()
            .update(|value| *value = value.wrapping_mul(3))
            .sum::<u64>()
    });

    assert_eq!(total, expected);
    assert_within_budget("update", allocations);
}

#[test]
fn update_chain_preserves_logical_order_across_shards() {
    let data = source();
    let expected: Vec<u64> = data.iter().map(|value| value.wrapping_mul(3)).collect();

    let collected: Vec<u64> = data
        .clone()
        .into_par_iter()
        .update(|value| *value = value.wrapping_mul(3))
        .collect();

    assert_eq!(collected, expected);
}

#[test]
fn exponential_blocks_chain_shards_and_preserves_values() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value % 11).sum();

    assert_shards("by_exponential_blocks", |workers| {
        data.clone()
            .into_par_iter()
            .by_exponential_blocks()
            .map(|value| {
                workers.record();
                value % 11
            })
            .sum::<u64>()
    });

    let (total, allocations) = allocations_of(|| {
        data.clone()
            .into_par_iter()
            .by_exponential_blocks()
            .map(|value| value % 11)
            .sum::<u64>()
    });

    assert_eq!(total, expected);
    assert_within_budget("by_exponential_blocks", allocations);
}

#[test]
fn uniform_blocks_chain_shards_and_preserves_values() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value % 11).sum();

    assert_shards("by_uniform_blocks", |workers| {
        data.clone()
            .into_par_iter()
            .by_uniform_blocks(512)
            .map(|value| {
                workers.record();
                value % 11
            })
            .sum::<u64>()
    });

    let (total, allocations) = allocations_of(|| {
        data.clone()
            .into_par_iter()
            .by_uniform_blocks(512)
            .map(|value| value % 11)
            .sum::<u64>()
    });

    assert_eq!(total, expected);
    assert_within_budget("by_uniform_blocks", allocations);
}

/// A chain whose adapters are all converted must shard end to end, not only at
/// its first adapter: each pushes into the consumer the next one wraps.
#[test]
fn stacked_converted_adapters_shard_and_preserve_values() {
    let data = source();
    let expected: u64 = data
        .iter()
        .filter_map(|value| (value % 3 == 0).then_some(*value))
        .flat_map(|value| [value % 5, value % 7])
        .sum();

    assert_shards("filter_map + flat_map", |workers| {
        data.par_iter()
            .filter_map(|value| (value % 3 == 0).then_some(*value))
            .flat_map(|value| {
                workers.record();
                [value % 5, value % 7]
            })
            .sum::<u64>()
    });

    let (total, allocations) = allocations_of(|| {
        data.par_iter()
            .filter_map(|value| (value % 3 == 0).then_some(*value))
            .flat_map(|value| [value % 5, value % 7])
            .sum::<u64>()
    });

    assert_eq!(total, expected);
    assert_within_budget("filter_map + flat_map", allocations);
}

/// The adapters recorded as staying sequential must still return the values a
/// standard sequential pass does at a length that would split.
///
/// This is the guard on the non-conversions: an adapter whose reason is a
/// cross-shard dependency returns wrong values if it is later pushed into a
/// consumer without supplying what the reason says is missing.
#[test]
fn unconverted_adapters_preserve_values_above_the_threshold() {
    let data = source();

    let indexed: Vec<(usize, u64)> = data.clone().into_par_iter().enumerate().collect();
    assert_eq!(
        indexed,
        data.iter().copied().enumerate().collect::<Vec<_>>()
    );

    let stepped: Vec<u64> = data.clone().into_par_iter().step_by(3).collect();
    assert_eq!(stepped, data.iter().copied().step_by(3).collect::<Vec<_>>());

    let taken: Vec<u64> = data.clone().into_par_iter().take(LEN / 3).collect();
    assert_eq!(taken, data[..LEN / 3].to_vec());

    let skipped: Vec<u64> = data.clone().into_par_iter().skip(LEN / 3).collect();
    assert_eq!(skipped, data[LEN / 3..].to_vec());

    let reversed: Vec<u64> = data.clone().into_par_iter().rev().collect();
    assert_eq!(reversed, data.iter().rev().copied().collect::<Vec<_>>());

    // `while_some` stops at the first `None` in the whole stream, not per shard.
    let mut optional: Vec<Option<u64>> = data.iter().copied().map(Some).collect();
    optional[LEN / 4] = None;
    let unwrapped: Vec<u64> = optional.into_par_iter().while_some().collect();
    assert_eq!(unwrapped, data[..LEN / 4].to_vec());
}
