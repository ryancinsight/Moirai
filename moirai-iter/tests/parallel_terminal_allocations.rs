//! Allocation probe for the parallel-iterator terminal path.
//!
//! Wall-clock benchmarks for this path are machine-dependent: the host is a
//! hybrid-core part whose scheduler can move a worker between performance and
//! efficiency cores mid-measurement. Allocation count is not — it is a property
//! of the code shape alone, and it is the property the terminal rework changed.
//!
//! The contract these tests pin is sub-linear allocation: a terminal must not
//! allocate per element. The superseded shape drove every terminal through
//! `seq_items()`, which collected each shard into a `Vec` and appended those
//! vectors on the way back up the merge tree, so allocation count grew with
//! element count. The folding consumers allocate only per shard, and shard
//! count is the source length divided by the dispatch threshold.
//!
//! This binary installs a counting global allocator, which is why it is its own
//! test target rather than a module of the unit-test suite.

use moirai_iter::parallel::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the layout it was
// given, so the allocator contract is exactly the system allocator's. The
// counter is a `Relaxed` atomic add on a separate static and imposes no
// ordering requirement of its own.
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

/// Well above the 1024-element dispatch threshold, so the drive splits.
const LEN: usize = 65_536;

/// Allocations permitted per terminal call.
///
/// One eighth of the element count. The superseded collect path allocated at
/// least once per element for the shard vectors alone, so it exceeds this by
/// more than an order of magnitude; the folding path allocates only the
/// scheduler jobs for the `LEN / 1024` shards, so it sits far below. The gap
/// is wide enough that the bound measures the code shape rather than the
/// allocator's or the scheduler's exact behaviour on a given run.
const ALLOCATION_BUDGET: usize = LEN / 8;

fn source() -> Vec<u64> {
    (0..LEN as u64).collect()
}

/// Run `operation` once to warm the executor, then count the allocations of a
/// second call. Worker threads and their queues are allocated on first use and
/// are not part of the terminal's own cost.
fn allocations_of<T>(operation: impl Fn() -> T) -> (T, usize) {
    let _warm = operation();

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let value = operation();
    let after = ALLOCATIONS.load(Ordering::Relaxed);

    (value, after.saturating_sub(before))
}

#[test]
fn borrowed_map_reassociated_sum_allocates_sublinearly() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value % 7).sum();

    let (total, allocations) = allocations_of(|| {
        data.par_iter()
            .map(|value| value % 7)
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    assert!(
        allocations <= ALLOCATION_BUDGET,
        "borrowed map/sum over {LEN} elements made {allocations} allocations, \
         above the {ALLOCATION_BUDGET} budget"
    );
}

#[test]
fn owned_map_reassociated_sum_allocates_sublinearly() {
    let data = source();
    let expected: u64 = data.iter().map(|value| value % 7).sum();

    let (total, allocations) = allocations_of(|| {
        data.clone()
            .into_par_iter()
            .map(|value| value % 7)
            .sum_reassociated::<u64>()
    });

    assert_eq!(total, expected);
    // The owned drive additionally clones the source and allocates one vector
    // per split, which is `len / threshold` allocations, not one per element.
    assert!(
        allocations <= ALLOCATION_BUDGET,
        "owned map/sum over {LEN} elements made {allocations} allocations, \
         above the {ALLOCATION_BUDGET} budget"
    );
}

#[test]
fn count_and_extrema_allocate_sublinearly() {
    let data = source();

    let (count, count_allocations) = allocations_of(|| data.par_iter().count());
    assert_eq!(count, LEN);
    assert!(
        count_allocations <= ALLOCATION_BUDGET,
        "count over {LEN} elements made {count_allocations} allocations, \
         above the {ALLOCATION_BUDGET} budget"
    );

    let (max, max_allocations) = allocations_of(|| data.par_iter().max());
    assert_eq!(max, data.iter().max());
    assert!(
        max_allocations <= ALLOCATION_BUDGET,
        "max over {LEN} elements made {max_allocations} allocations, \
         above the {ALLOCATION_BUDGET} budget"
    );
}

#[test]
fn find_any_allocates_sublinearly_and_short_circuits() {
    let mut data = source();
    let target = u64::MAX;
    data[LEN / 8] = target;

    let (found, allocations) =
        allocations_of(|| data.par_iter().find_any(|value| **value == target));

    assert_eq!(found, Some(&target));
    assert!(
        allocations <= ALLOCATION_BUDGET,
        "find_any over {LEN} elements made {allocations} allocations, \
         above the {ALLOCATION_BUDGET} budget"
    );
}
