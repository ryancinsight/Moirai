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

use moirai_iter::{
    cache::CacheIterExt,
    iter_ops::ParallelIter,
    parallel::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    AsyncContext, ExecutionContext, MoiraiIterator,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::future::Future;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::Poll;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the layout it was
// given, so the allocator contract is exactly the system allocator's. The
// counter is a `Relaxed` atomic add on a separate static and imposes no
// ordering requirement of its own.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Well above the 1024-element dispatch threshold, so the drive splits.
const LEN: usize = 65_536;

/// Large enough that the public host-derived chunk planner reaches fan-out on
/// the supported CI and measurement hosts.
const MAP_LEN: usize = 131_072;
const MAP_OUTPUT_BYTES: usize = MAP_LEN * size_of::<u64>();

/// Small enough to retain the zero-copy iterator's sequential map route.
const ZERO_COPY_MAP_LEN: usize = 1_024;
const ZERO_COPY_MAP_OUTPUT_BYTES: usize = ZERO_COPY_MAP_LEN * size_of::<u64>();

const CONTEXT_MAP_LEN: usize = 1_024;
const CONTEXT_MAP_OUTPUT_BYTES: usize = CONTEXT_MAP_LEN * size_of::<u64>();
// Retained future and output slabs scale with the concurrency bound rather than
// the item count. These budgets include the output vector and fixed executor
// metadata while rejecting a return to one task-node allocation per item.
const CONTEXT_FIXED_ALLOCATION_BUDGET: usize = 16;
const CONTEXT_FOR_EACH_FIXED_ALLOCATION_BUDGET: usize = 8;
const CONTEXT_SLOT_BYTE_BUDGET: usize = 64;
const CONTEXT_MAP_FIXED_BYTE_BUDGET: usize = 12_288;
const CONTEXT_FOR_EACH_FIXED_BYTE_BUDGET: usize = 2_048;

fn context_slot_count() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

fn context_map_allocation_budget() -> usize {
    context_slot_count() + CONTEXT_FIXED_ALLOCATION_BUDGET
}

fn context_map_byte_budget() -> usize {
    CONTEXT_MAP_OUTPUT_BYTES
        + context_slot_count() * CONTEXT_SLOT_BYTE_BUDGET
        + CONTEXT_MAP_FIXED_BYTE_BUDGET
}

fn context_for_each_allocation_budget() -> usize {
    context_slot_count() + CONTEXT_FOR_EACH_FIXED_ALLOCATION_BUDGET
}

fn context_for_each_byte_budget() -> usize {
    context_slot_count() * CONTEXT_SLOT_BYTE_BUDGET + CONTEXT_FOR_EACH_FIXED_BYTE_BUDGET
}

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

fn map_source() -> Vec<u64> {
    (0..MAP_LEN as u64)
        .map(|value| value.wrapping_mul(31).wrapping_add(7))
        .collect()
}

fn map_values(data: Vec<u64>) -> Vec<u64> {
    ParallelIter::new(data).map(|value| value.wrapping_mul(3).wrapping_add(1))
}

fn context_map_values(data: Vec<u64>) -> Vec<u64> {
    futures::executor::block_on(async {
        MoiraiIterator::parallel(data)
            .map_async(|value| async move { value.wrapping_mul(5).wrapping_add(3) })
            .await
            .collect()
            .await
    })
}

fn pending_once<T>(value: T) -> impl Future<Output = T> {
    let mut value = Some(value);
    let mut first_poll = true;
    futures::future::poll_fn(move |context| {
        if first_poll {
            first_poll = false;
            context.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(
                value
                    .take()
                    .expect("pending-once future polled after completion"),
            )
        }
    })
}

fn context_pending_map_values(data: Vec<u64>) -> Vec<u64> {
    futures::executor::block_on(async {
        MoiraiIterator::parallel(data)
            .map_async(|value| pending_once(value.wrapping_mul(5).wrapping_add(3)))
            .await
            .collect()
            .await
    })
}

fn context_large_limit_map_values(data: Vec<u64>) -> Vec<u64> {
    let context = ExecutionContext::Async(AsyncContext::new().with_max_concurrent(usize::MAX));
    futures::executor::block_on(async {
        MoiraiIterator::new(data, context)
            .map_async(|value| pending_once(value.wrapping_mul(5).wrapping_add(3)))
            .await
            .collect()
            .await
    })
}

fn context_pending_for_each(data: Vec<usize>, visits: Arc<Vec<AtomicUsize>>) {
    futures::executor::block_on(MoiraiIterator::parallel(data).for_each_async(move |index| {
        let visits = Arc::clone(&visits);
        async move {
            pending_once(()).await;
            visits[index].fetch_add(1, Ordering::SeqCst);
        }
    }));
}

fn warmed_allocation_ledger<T, R>(
    warm_input: T,
    measured_input: T,
    mut operation: impl FnMut(T) -> R,
) -> (R, usize, usize) {
    let _warm = operation(warm_input);
    let before_allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let before_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let result = operation(measured_input);
    let allocations = ALLOCATIONS
        .load(Ordering::Relaxed)
        .saturating_sub(before_allocations);
    let allocated_bytes = ALLOCATED_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(before_bytes);
    (result, allocations, allocated_bytes)
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
fn borrowed_copied_map_filter_standard_sum_allocates_nothing() {
    let data = source();
    let expected: u64 = data
        .iter()
        .copied()
        .map(|value| value.wrapping_mul(19).wrapping_add(23))
        .filter(|value| value & 7 != 0)
        .sum();

    let (total, allocations) = allocations_of(|| {
        data.par_iter()
            .copied()
            .map(|value| value.wrapping_mul(19).wrapping_add(23))
            .filter(|value| value & 7 != 0)
            .sum::<u64>()
    });

    assert_eq!(total, expected);
    assert_eq!(
        allocations, 0,
        "warmed borrowed copied/map/filter standard sum over {LEN} elements must not allocate"
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

#[test]
fn parallel_iter_map_records_output_allocation_ledger() {
    let warm = map_values(map_source());
    assert_eq!(warm.len(), MAP_LEN);

    let input = map_source();
    let before_allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let before_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let mapped = map_values(input);
    let allocations = ALLOCATIONS
        .load(Ordering::Relaxed)
        .saturating_sub(before_allocations);
    let allocated_bytes = ALLOCATED_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(before_bytes);

    assert_eq!(mapped.len(), MAP_LEN);
    assert!(
        mapped.iter().enumerate().all(|(index, value)| {
            let input = (index as u64).wrapping_mul(31).wrapping_add(7);
            *value == input.wrapping_mul(3).wrapping_add(1)
        }),
        "parallel map must preserve every ordered output value"
    );
    assert_eq!(
        allocations, 3,
        "warmed map must allocate only input chunk views, completion ranges, and final output"
    );
    assert!(
        (MAP_OUTPUT_BYTES..=MAP_OUTPUT_BYTES + MAP_OUTPUT_BYTES / 16).contains(&allocated_bytes),
        "map made {allocations} allocations totalling {allocated_bytes} gross bytes for a \
         {MAP_OUTPUT_BYTES}-byte output; \
         metadata must stay below one-sixteenth of the final output"
    );
}

#[test]
fn zero_copy_map_separates_constructor_and_output_allocations() {
    let data: Vec<u64> = (0..ZERO_COPY_MAP_LEN as u64)
        .map(|value| value.wrapping_mul(17).wrapping_add(11))
        .collect();

    black_box(data.zero_copy_par_iter());

    let before_constructor_allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let before_constructor_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let iter = data.zero_copy_par_iter();
    let constructor_allocations = ALLOCATIONS
        .load(Ordering::Relaxed)
        .saturating_sub(before_constructor_allocations);
    let constructor_bytes = ALLOCATED_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(before_constructor_bytes);

    let before_map_allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let before_map_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let mapped = iter.map(|value| value.wrapping_mul(3).wrapping_add(1));
    let map_allocations = ALLOCATIONS
        .load(Ordering::Relaxed)
        .saturating_sub(before_map_allocations);
    let map_bytes = ALLOCATED_BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(before_map_bytes);

    assert!(
        mapped.iter().enumerate().all(|(index, value)| {
            let input = (index as u64).wrapping_mul(17).wrapping_add(11);
            *value == input.wrapping_mul(3).wrapping_add(1)
        }),
        "zero-copy map must preserve every ordered output value"
    );
    assert_eq!(
        (
            constructor_allocations,
            constructor_bytes,
            map_allocations,
            map_bytes,
        ),
        (0, 0, 1, ZERO_COPY_MAP_OUTPUT_BYTES),
        "iterator construction must not allocate and its sequential map must allocate only output"
    );
}

#[test]
fn parallel_context_async_map_excludes_topology_allocations() {
    let source = || {
        (0..CONTEXT_MAP_LEN as u64)
            .map(|value| value.wrapping_mul(19).wrapping_add(7))
            .collect::<Vec<_>>()
    };
    let (mapped, allocations, allocated_bytes) =
        warmed_allocation_ledger(source(), source(), context_map_values);
    let allocation_budget = context_map_allocation_budget();
    let byte_budget = context_map_byte_budget();

    assert!(
        mapped.iter().enumerate().all(|(index, value)| {
            let input = (index as u64).wrapping_mul(19).wrapping_add(7);
            *value == input.wrapping_mul(5).wrapping_add(3)
        }),
        "parallel-context async map must preserve every ordered output value"
    );
    assert!(
        allocations <= allocation_budget,
        "warmed parallel-context async map made {allocations} allocations; its buffered futures \
         and output must fit the {allocation_budget}-allocation budget"
    );
    assert!(
        allocated_bytes <= byte_budget,
        "warmed parallel-context async map allocated {allocated_bytes} gross bytes; its buffered \
        futures, output, and fixed metadata must fit the {byte_budget}-byte budget"
    );
}

#[test]
fn large_context_limit_does_not_reserve_for_empty_or_single_input() {
    const FIXED_ALLOCATION_BUDGET: usize = 20;
    const FIXED_BYTE_BUDGET: usize = 8_192;

    let (empty, empty_allocations, empty_bytes) =
        warmed_allocation_ledger(Vec::new(), Vec::new(), context_large_limit_map_values);
    let (single, single_allocations, single_bytes) =
        warmed_allocation_ledger(vec![5], vec![5], context_large_limit_map_values);

    assert!(empty.is_empty());
    assert_eq!(single, [28]);
    assert!(
        empty_allocations <= FIXED_ALLOCATION_BUDGET,
        "empty map with a usize::MAX ceiling made {empty_allocations} allocations"
    );
    assert!(
        single_allocations <= FIXED_ALLOCATION_BUDGET,
        "single-item map with a usize::MAX ceiling made {single_allocations} allocations"
    );
    assert!(
        empty_bytes <= FIXED_BYTE_BUDGET,
        "empty map with a usize::MAX ceiling allocated {empty_bytes} gross bytes"
    );
    assert!(
        single_bytes <= FIXED_BYTE_BUDGET,
        "single-item map with a usize::MAX ceiling allocated {single_bytes} gross bytes"
    );
}

#[test]
fn parallel_context_pending_map_records_entry_allocation_ledger() {
    let source = || {
        (0..CONTEXT_MAP_LEN as u64)
            .map(|value| value.wrapping_mul(19).wrapping_add(7))
            .collect::<Vec<_>>()
    };
    let (mapped, allocations, allocated_bytes) =
        warmed_allocation_ledger(source(), source(), context_pending_map_values);
    let allocation_budget = context_map_allocation_budget();
    let byte_budget = context_map_byte_budget();

    assert!(
        mapped.iter().enumerate().all(|(index, value)| {
            let input = (index as u64).wrapping_mul(19).wrapping_add(7);
            *value == input.wrapping_mul(5).wrapping_add(3)
        }),
        "pending parallel-context map must preserve every ordered output value"
    );
    assert!(
        allocations <= allocation_budget,
        "pending ordered map made {allocations} allocations; entry futures and output must fit \
         the {allocation_budget}-allocation budget"
    );
    assert!(
        allocated_bytes <= byte_budget,
        "pending ordered map allocated {allocated_bytes} gross bytes; entry futures and output \
         must fit the {byte_budget}-byte budget"
    );
}

#[test]
fn parallel_context_pending_for_each_records_entry_allocation_ledger() {
    let visits = || {
        Arc::new(
            (0..CONTEXT_MAP_LEN)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>(),
        )
    };
    let input = || (0..CONTEXT_MAP_LEN).collect::<Vec<_>>();
    let warm_visits = visits();
    let measured_visits = visits();
    let ((), allocations, allocated_bytes) = warmed_allocation_ledger(
        (input(), warm_visits),
        (input(), Arc::clone(&measured_visits)),
        |(data, visits)| context_pending_for_each(data, visits),
    );
    let allocation_budget = context_for_each_allocation_budget();
    let byte_budget = context_for_each_byte_budget();

    assert!(
        measured_visits
            .iter()
            .all(|count| count.load(Ordering::SeqCst) == 1),
        "pending parallel-context for-each must visit every item exactly once"
    );
    assert!(
        allocations <= allocation_budget,
        "pending for-each made {allocations} allocations; entry futures must fit the \
         {allocation_budget}-allocation budget"
    );
    assert!(
        allocated_bytes <= byte_budget,
        "pending for-each allocated {allocated_bytes} gross bytes; entry futures must fit the \
         {byte_budget}-byte budget"
    );
}
