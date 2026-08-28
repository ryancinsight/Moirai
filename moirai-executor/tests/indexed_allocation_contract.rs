//! Allocation contract for warmed indexed operations.

use moirai_core::Priority;
use moirai_executor::{SyncTask, ThreadScheduler};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(miri))]
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::AtomicBool,
};

/// Serializes the two tests: both drive the same global counting state, and
/// the plain `cargo test` harness runs tests as threads of one process.
static HARNESS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(not(miri))]
static COUNTING: AtomicBool = AtomicBool::new(false);
#[cfg(not(miri))]
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
/// Live-bytes balance inside a window; signed for frees of pre-window blocks.
#[cfg(not(miri))]
static LIVE: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);
/// Ledger floor: allocations at or above this many bytes are listed by size.
#[cfg(not(miri))]
static LEDGER_FLOOR: AtomicUsize = AtomicUsize::new(usize::MAX);
#[cfg(not(miri))]
const LEDGER_SLOTS: usize = 512;
#[cfg(not(miri))]
static LEDGER_SIZE: [AtomicUsize; LEDGER_SLOTS] = [const { AtomicUsize::new(0) }; LEDGER_SLOTS];
#[cfg(not(miri))]
static LEDGER_LIVE: [AtomicBool; LEDGER_SLOTS] = [const { AtomicBool::new(false) }; LEDGER_SLOTS];
#[cfg(not(miri))]
static LEDGER_NEXT: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(miri))]
fn ledger_push(size: usize) {
    let slot = LEDGER_NEXT.fetch_add(1, Ordering::Relaxed);
    if slot < LEDGER_SLOTS {
        LEDGER_SIZE[slot].store(size, Ordering::Relaxed);
        LEDGER_LIVE[slot].store(true, Ordering::Relaxed);
    }
}

#[cfg(not(miri))]
fn ledger_free(size: usize) {
    let filled = LEDGER_NEXT.load(Ordering::Relaxed).min(LEDGER_SLOTS);
    for slot in 0..filled {
        if LEDGER_SIZE[slot].load(Ordering::Relaxed) == size
            && LEDGER_LIVE[slot]
                .compare_exchange(true, false, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            return;
        }
    }
}

#[cfg(not(miri))]
fn track(delta: isize, size: usize, freed: bool) {
    LIVE.fetch_add(delta, Ordering::Relaxed);
    if size >= LEDGER_FLOOR.load(Ordering::Relaxed) {
        if freed {
            ledger_free(size);
        } else {
            ledger_push(size);
        }
    }
}

#[cfg(not(miri))]
struct CountingAllocator;

// SAFETY: every operation delegates to `System` with identical arguments; the
// relaxed counters observe calls and do not affect allocation semantics.
#[cfg(not(miri))]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            track(
                isize::try_from(layout.size()).expect("layout size fits isize"),
                layout.size(),
                false,
            );
        }
        // SAFETY: this method preserves the caller's allocation contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            track(
                -isize::try_from(layout.size()).expect("layout size fits isize"),
                layout.size(),
                true,
            );
        }
        // SAFETY: this method forwards the pointer and layout unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            track(
                isize::try_from(new_size).expect("layout size fits isize")
                    - isize::try_from(layout.size()).expect("layout size fits isize"),
                layout.size(),
                true,
            );
            if new_size >= LEDGER_FLOOR.load(Ordering::Relaxed) {
                ledger_push(new_size);
            }
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

/// Runs `f` in a fresh ledger window and prints its allocation summary and
/// the surviving blocks at or above `floor` bytes, as `size x count`.
#[cfg(not(miri))]
fn footprint_window<R>(label: &str, floor: usize, f: impl FnOnce() -> R) -> (R, usize, isize) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    LEDGER_NEXT.store(0, Ordering::Relaxed);
    LEDGER_FLOOR.store(floor, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let result = f();
    COUNTING.store(false, Ordering::Relaxed);

    let filled = LEDGER_NEXT.load(Ordering::Relaxed).min(LEDGER_SLOTS);
    let mut survivors: Vec<(usize, usize)> = Vec::new();
    for slot in 0..filled {
        if LEDGER_LIVE[slot].load(Ordering::Relaxed) {
            let size = LEDGER_SIZE[slot].load(Ordering::Relaxed);
            match survivors.iter_mut().find(|(s, _)| *s == size) {
                Some((_, count)) => *count += 1,
                None => survivors.push((size, 1)),
            }
        }
    }
    survivors.sort_unstable_by(|a, b| b.cmp(a));
    let blocks = survivors
        .iter()
        .map(|(size, count)| format!("{size}x{count}"))
        .collect::<Vec<_>>()
        .join(" ");
    let allocs = ALLOCATIONS.load(Ordering::Relaxed);
    let retained = LIVE.load(Ordering::Relaxed);
    println!("  {label:<28} allocs {allocs:>5}  retained {retained:>10}  blocks: {blocks}");
    (result, allocs, retained)
}

/// Retained-footprint instrument for the pool itself
/// (`MOIRAI-POOL-RETAINED-FOOTPRINT-2026-08-27`, delivered by PR #184).
///
/// The item's consumer probe lived in apollo; this is the provider-side
/// instrument, so the next retention change measures here. Windows per
/// stage: scheduler construction, the first trivial fan-out, the first
/// fan-out whose closure exceeds the inline job words, a repeat of it, and
/// shutdown. Post-#184 baselines on a 24-worker host: construction retains
/// one 36,864-byte buffer per worker — 256 queue slots at the de-aligned
/// 18-word slot exactly — and every fan-out window allocates and retains
/// zero: indexed fan-out shares one closure by reference, so no per-job
/// storage exists on any path. Asserts the zero-allocation fan-out contract;
/// prints the rest. Run with `--ignored --nocapture`.
#[cfg(not(miri))]
#[test]
fn pool_retained_footprint_attribution() {
    let _serial = HARNESS.lock().expect("harness lock");
    const JOBS: usize = 256;
    let workers = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    println!("workers = {workers}, jobs per fan-out = {JOBS} (ledger floor 4096)");

    let (scheduler, _, _) = footprint_window("construction", 4096, || {
        ThreadScheduler::new(workers, "pool-footprint").expect("the probe scheduler must start")
    });

    let hits: Vec<AtomicUsize> = (0..JOBS).map(|_| AtomicUsize::new(0)).collect();
    let ((), trivial_allocs, trivial_retained) =
        footprint_window("first trivial fan-out", 4096, || {
            scheduler
                .for_each_indexed::<SyncTask, _>(Priority::Normal, None, JOBS, |index| {
                    hits[index].fetch_add(1, Ordering::Relaxed);
                })
                .expect("the trivial fan-out must complete");
        });
    assert_eq!(
        (trivial_allocs, trivial_retained),
        (0, 0),
        "the first indexed fan-out must allocate nothing: it shares one closure by reference"
    );

    // 32 words of capture: past the inline-job words, the shape closest to a
    // capture-heavy consumer.
    let payload: [usize; 32] = std::array::from_fn(|i| i);
    let ((), wide_allocs, wide_retained) =
        footprint_window("first wide-capture fan-out", 4096, || {
            scheduler
                .for_each_indexed::<SyncTask, _>(Priority::Normal, None, JOBS, move |index| {
                    std::hint::black_box(payload[index % 32]);
                })
                .expect("the wide-capture fan-out must complete");
        });
    assert_eq!(
        (wide_allocs, wide_retained),
        (0, 0),
        "a capture past the inline-job words changes nothing: indexed fan-out has no per-job storage"
    );
    let ((), repeat_allocs, repeat_retained) =
        footprint_window("repeat wide-capture fan-out", 4096, || {
            scheduler
                .for_each_indexed::<SyncTask, _>(Priority::Normal, None, JOBS, move |index| {
                    std::hint::black_box(payload[index % 32]);
                })
                .expect("the repeat fan-out must complete");
        });
    assert_eq!((repeat_allocs, repeat_retained), (0, 0));

    footprint_window("shutdown", 4096, || scheduler.shutdown());
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
}

#[test]
fn warmed_indexed_operations_limit_allocations() {
    let _serial = HARNESS.lock().expect("harness lock");
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
