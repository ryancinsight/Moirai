//! Allocation contract for warmed indexed operations.

use moirai_core::Priority;
use moirai_executor::{SyncTask, ThreadScheduler};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(miri))]
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::AtomicBool,
};

#[cfg(not(miri))]
static COUNTING: AtomicBool = AtomicBool::new(false);
#[cfg(not(miri))]
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(miri))]
struct CountingAllocator;

// SAFETY: every operation delegates to `System` with identical arguments; the
// relaxed counters observe calls and do not affect allocation semantics.
#[cfg(not(miri))]
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

#[cfg(not(miri))]
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct AllocationCount {
    #[cfg(not(miri))]
    value: usize,
}

#[cfg(not(miri))]
impl AllocationCount {
    fn get(&self) -> usize {
        self.value
    }
}

#[cfg(not(miri))]
struct AllocationWindow;

#[cfg(not(miri))]
impl AllocationWindow {
    fn start() -> Self {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        COUNTING.store(true, Ordering::Relaxed);
        Self
    }
}

#[cfg(not(miri))]
impl Drop for AllocationWindow {
    fn drop(&mut self) {
        COUNTING.store(false, Ordering::Relaxed);
    }
}

fn measure_allocations<T>(operation: impl FnOnce() -> T) -> (T, AllocationCount) {
    #[cfg(not(miri))]
    let window = AllocationWindow::start();

    let output = operation();

    #[cfg(not(miri))]
    {
        drop(window);
        let count = AllocationCount {
            value: ALLOCATIONS.load(Ordering::Relaxed),
        };
        (output, count)
    }
    #[cfg(miri)]
    {
        (output, AllocationCount {})
    }
}

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

    let ((), fanout_allocations) = measure_allocations(|| {
        for _ in 0..REPETITIONS {
            scheduler
                .for_each_indexed::<SyncTask, _>(Priority::Normal, None, ITEMS, |index| {
                    visits[index].fetch_add(1, Ordering::Relaxed);
                })
                .expect("the measured fan-out must complete");
        }
    });

    #[cfg(not(miri))]
    assert_eq!(
        fanout_allocations.get(),
        0,
        "completion-only indexed fan-out must retain no per-call heap state"
    );
    #[cfg(miri)]
    let AllocationCount {} = fanout_allocations;
    assert!(visits
        .iter()
        .all(|count| { count.load(Ordering::Relaxed) == REPETITIONS + 1 }));

    let (reduced, reduction_allocations) = measure_allocations(|| {
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
        reduced
    });

    assert_eq!(reduced, EXPECTED_SUM);
    #[cfg(not(miri))]
    assert_eq!(
        reduction_allocations.get(),
        REPETITIONS,
        "indexed map/reduce must allocate only its result-slot buffer"
    );
    #[cfg(miri)]
    let AllocationCount {} = reduction_allocations;

    scheduler.shutdown();
}
