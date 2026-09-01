//! Evidence that a chain containing a converted adapter keeps the source's shards.
//!
//! A value-equality test alone cannot distinguish the two shapes this change
//! separates: the superseded `drive` collected the whole logical stream into one
//! vector and re-split *that*, which returns the same values as driving the
//! source directly. What differs is whether the source's shards survive the
//! adapter, so these tests measure that directly.
//!
//! # The signal
//!
//! Allocated bytes, which is a property of the code shape rather than of a run.
//! A thread-identity signal was tried first and removed: `drive_split` runs a
//! branch on the caller when scheduler admission is refused, so a chain that
//! shards perfectly well collapses to one thread under a saturated machine.
//! Asserting on it passed when this file ran alone and failed when the
//! workspace suite ran every test binary in parallel — flakiness authored into
//! the suite rather than evidence. Allocated bytes have no such dependence.
//!
//! The two source kinds leave opposite signatures, and both are checked:
//!
//! - **Borrowed sources** split a slice by index range and copy nothing, so a
//!   converted chain allocates essentially nothing (measured: 512 bytes) where
//!   the collect shape allocated the whole logical stream (524288 bytes).
//! - **Owned sources** have no safe zero-copy split, so splitting copies each
//!   half down to the dispatch threshold: a chain that splits allocates
//!   measurably more than the source itself (measured: 786944 against a 524288
//!   source). The collect shape's tell is the opposite — it handed the whole
//!   vector to one `consume` call and never split, allocating the source and
//!   nothing beyond it.
//!
//! Every case runs above `PARALLEL_DRIVE_THRESHOLD` (1024). Below it a source
//! never splits at all, so a chain measured there exercises the sequential
//! fallback and proves nothing about the converted path — the coverage gap the
//! terminal rework found in the pre-existing tests.

use moirai_iter::parallel::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the layout it was
// given, so the allocator contract is exactly the system allocator's. The
// counter is a `Relaxed` atomic add on a separate static and imposes no
// ordering requirement of its own.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Sixty-four times the 1024-element dispatch threshold, so the source is well
/// past the point where its drive splits.
const LEN: usize = 65_536;

/// Bytes the logical stream occupies once materialized.
const MATERIALIZED: usize = LEN * std::mem::size_of::<u64>();

/// Run `operation` once warm, then total the bytes a second call allocates.
///
/// The warm call pays for the executor's workers and their queues, which are
/// allocated on first use and are not part of the chain's own cost.
fn allocated_bytes<T>(operation: impl Fn() -> T) -> (T, usize) {
    let _warm = operation();

    let before = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let value = operation();
    let after = ALLOCATED_BYTES.load(Ordering::Relaxed);

    (value, after.saturating_sub(before))
}

/// A chain over a borrowed source must not materialize its logical stream.
///
/// The bound sits an order of magnitude below one full copy of the stream, so
/// it separates the two shapes without pinning the allocator's exact
/// behaviour.
fn assert_no_materialization(adapter: &str, bytes: usize) {
    let budget = MATERIALIZED / 16;
    assert!(
        bytes <= budget,
        "a chain containing `{adapter}` over {LEN} borrowed elements allocated {bytes} bytes,          above the {budget} budget; the logical stream is being materialized before the split          ({MATERIALIZED} bytes is one full copy of it)"
    );
}

/// A chain over an owned source must actually split it.
///
/// Splitting an owned source copies each half, so a drive that splits allocates
/// measurably more than the source. A drive that consumed the whole source in
/// one call allocates exactly the source and stops there.
fn assert_split_copies(adapter: &str, bytes: usize, source_bytes: usize) {
    let floor = source_bytes + source_bytes / 4;
    assert!(
        bytes >= floor,
        "a chain containing `{adapter}` over {LEN} owned elements allocated {bytes} bytes,          below the {floor} bytes that splitting a {source_bytes}-byte source copies; the          drive consumed the whole source without splitting it"
    );
}

fn source() -> Vec<u64> {
    (0..LEN as u64).collect()
}

#[test]
fn filter_map_chain_keeps_borrowed_shards() {
    let data = source();
    let expected: u64 = data
        .iter()
        .filter_map(|value| (value % 3 == 0).then_some(value * 2))
        .sum();

    let (total, bytes) = allocated_bytes(|| {
        data.par_iter()
            .filter_map(|value| (value % 3 == 0).then_some(value * 2))
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    assert_no_materialization("filter_map", bytes);
}

#[test]
fn flat_map_chain_keeps_borrowed_shards() {
    let data = source();
    let expected: u64 = data.iter().flat_map(|value| [value % 5, value % 7]).sum();

    let (total, bytes) = allocated_bytes(|| {
        data.par_iter()
            .flat_map(|value| [value % 5, value % 7])
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    assert_no_materialization("flat_map", bytes);
}

/// A chain whose adapters are all converted must shard end to end, not only at
/// its first adapter: each pushes into the consumer the next one wraps.
#[test]
fn stacked_converted_adapters_keep_borrowed_shards() {
    let data = source();
    let expected: u64 = data
        .iter()
        .filter_map(|value| (value % 3 == 0).then_some(*value))
        .flat_map(|value| [value % 5, value % 7])
        .sum();

    let (total, bytes) = allocated_bytes(|| {
        data.par_iter()
            .filter_map(|value| (value % 3 == 0).then_some(*value))
            .flat_map(|value| [value % 5, value % 7])
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    assert_no_materialization("filter_map + flat_map", bytes);
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
fn flatten_chain_keeps_its_nested_source_shards() {
    let nested = nested_source();
    let expected: u64 = nested.iter().flatten().sum();

    let (total, bytes) = allocated_bytes(|| {
        nested
            .clone()
            .into_par_iter()
            .flatten()
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    {
        // The clone inside the measured closure is the floor. The collect shape
        // added a full copy of the flattened stream on top of it; the push adds
        // only the nested source's own split.
        let clone_bytes = GROUPS * GROUP_LEN * std::mem::size_of::<u64>()
            + GROUPS * std::mem::size_of::<Vec<u64>>();
        let budget = clone_bytes + MATERIALIZED / 4;
        assert!(
            bytes <= budget,
            "a chain containing `flatten` allocated {bytes} bytes against a              {clone_bytes}-byte source clone, above the {budget} budget; the flattened              stream is being materialized"
        );
    }
}

#[test]
fn update_chain_splits_its_owned_source() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value.wrapping_mul(3)).sum();

    let (total, bytes) = allocated_bytes(|| {
        data.clone()
            .into_par_iter()
            .update(|value| *value = value.wrapping_mul(3))
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    assert_split_copies("update", bytes, MATERIALIZED);
}

#[test]
fn exponential_blocks_chain_splits_its_owned_source() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value % 11).sum();

    let (total, bytes) = allocated_bytes(|| {
        data.clone()
            .into_par_iter()
            .by_exponential_blocks()
            .map(|value| value % 11)
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    assert_split_copies("by_exponential_blocks", bytes, MATERIALIZED);
}

#[test]
fn uniform_blocks_chain_splits_its_owned_source() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value % 11).sum();

    let (total, bytes) = allocated_bytes(|| {
        data.clone()
            .into_par_iter()
            .by_uniform_blocks(512)
            .map(|value| value % 11)
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    assert_split_copies("by_uniform_blocks", bytes, MATERIALIZED);
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

#[test]
fn flatten_chain_preserves_nested_order_across_shards() {
    let nested = nested_source();
    let expected: Vec<u64> = nested.iter().flatten().copied().collect();

    let collected: Vec<u64> = nested.clone().into_par_iter().flatten().collect();

    assert_eq!(collected, expected);
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
