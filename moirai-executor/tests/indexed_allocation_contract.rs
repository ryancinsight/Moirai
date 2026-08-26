//! Allocation contract for warmed indexed operations.

use moirai_core::Priority;
use moirai_executor::{SyncTask, ThreadScheduler};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

// SAFETY: every operation delegates to `System` with identical arguments; the
// relaxed counters observe calls and do not affect allocation semantics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: this method preserves the caller's allocation contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: this method forwards the pointer and layout unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: this method forwards the allocation arguments unchanged.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn warmed_indexed_operations_limit_allocations() {
    const ITEMS: usize = 4_096;
    const REPETITIONS: usize = 8;
    const EXPECTED_SUM: usize = ITEMS * (ITEMS + 1) / 2;

    let scheduler = ThreadScheduler::new(2, "indexed-allocation-contract")
        .expect("the test scheduler must initialize");
    let visits: [AtomicUsize; ITEMS] = std::array::from_fn(|_| AtomicUsize::new(0));

    scheduler
        .for_each_indexed::<SyncTask, _>(Priority::Normal, None, ITEMS, |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        })
        .expect("the warm-up fan-out must complete");

    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..REPETITIONS {
        scheduler
            .for_each_indexed::<SyncTask, _>(Priority::Normal, None, ITEMS, |index| {
                visits[index].fetch_add(1, Ordering::Relaxed);
            })
            .expect("the measured fan-out must complete");
    }
    COUNTING.store(false, Ordering::Relaxed);

    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        0,
        "completion-only indexed fan-out must retain no per-call heap state"
    );
    assert!(visits
        .iter()
        .all(|count| { count.load(Ordering::Relaxed) == REPETITIONS + 1 }));

    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let mut reduced = 0;
    for _ in 0..REPETITIONS {
        reduced = scheduler
            .map_reduce_indexed::<SyncTask, _, _, _>(
                Priority::Normal,
                None,
                ITEMS,
                0,
                |index| index + 1,
                |left, right| left + right,
            )
            .expect("the measured map/reduce must complete");
    }
    COUNTING.store(false, Ordering::Relaxed);

    assert_eq!(reduced, EXPECTED_SUM);
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        REPETITIONS,
        "indexed map/reduce must allocate only its result-slot buffer"
    );

    scheduler.shutdown();
}
