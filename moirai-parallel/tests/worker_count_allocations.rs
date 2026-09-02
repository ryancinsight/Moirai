//! Warm allocation contract for the entry points that derive a worker count.
//!
//! `for_each_chunk_mut_with_state` and `fold_reduce_with` used to call
//! `themis::CpuTopology::detect()` on every invocation to read one number.
//! That materializes the whole NUMA and cache-level description: 77
//! allocations totalling 16,480 bytes per call on a 24-processor host. The
//! count is a process constant, so it is now derived once by
//! `moirai_core::executor::logical_parallelism`.

use core::sync::atomic::{AtomicUsize, Ordering};
use moirai_parallel::{fold_reduce_with, for_each_chunk_mut_with_state, Parallel};
use std::alloc::{GlobalAlloc, Layout, System};

struct CountingAllocator;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation delegates unchanged pointers and layouts to the
// system allocator; the counter observes calls without altering allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` came from this delegated allocator.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the arguments are forwarded unchanged to the system
        // allocator that created `pointer`.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn traverse_chunks(data: &mut [u64]) {
    for_each_chunk_mut_with_state::<Parallel, _, _, _, _>(
        data,
        256,
        || 0u64,
        |state, chunk| {
            for value in chunk.iter_mut() {
                *value += 1;
            }
            *state += chunk.len() as u64;
        },
    );
}

fn reduce_indices(len: usize) -> u64 {
    fold_reduce_with::<Parallel, u64, _, _, _>(
        len,
        || 0u64,
        |accumulator, index| accumulator + index as u64,
        |left, right| left + right,
    )
}

#[test]
fn warmed_chunk_state_traversal_allocates_nothing() {
    let mut data = vec![0u64; 65_536];
    traverse_chunks(&mut data);

    ALLOCATIONS.store(0, Ordering::Relaxed);
    traverse_chunks(&mut data);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);

    assert_eq!(
        allocations, 0,
        "warmed traversal must allocate nothing; a per-call          CpuTopology::detect cost 77 allocations here"
    );
    assert!(
        data.iter().all(|&value| value == 2),
        "both traversals must visit every element exactly once"
    );
}

#[test]
fn warmed_fold_reduce_allocates_only_its_result_slots() {
    // `fold_reduce_with` owns one `Vec<Option<A>>` of per-worker slots, which
    // is its result storage, not worker-count derivation. Pinning a small
    // bound distinguishes the two: deriving the count from a topology probe
    // added 77 allocations on top of this.
    const LEN: usize = 65_536;
    let warm = reduce_indices(LEN);

    ALLOCATIONS.store(0, Ordering::Relaxed);
    let repeated = reduce_indices(LEN);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);

    assert_eq!(repeated, warm, "the reduction is unchanged by warming");
    assert_eq!(
        repeated,
        (0..LEN as u64).sum::<u64>(),
        "the reduction sums every index exactly once"
    );
    assert!(
        allocations <= 4,
        "warmed fold-reduce allocated {allocations} times; only its result          slots may allocate, so anything larger is worker-count derivation          creeping back"
    );
}
