//! Allocation contract for warmed indexed operations.

use moirai_core::Priority;
use moirai_executor::{SyncTask, ThreadScheduler};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(miri))]
#[path = "indexed_allocation_contract/allocation_ledger.rs"]
mod allocation_ledger;

/// Serializes the two tests: both drive the same global counting state, and
/// the plain `cargo test` harness runs tests as threads of one process.
static HARNESS: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

fn measure_allocations<T>(operation: impl FnOnce() -> T) -> (T, AllocationCount) {
    #[cfg(not(miri))]
    {
        let (output, snapshot) = allocation_ledger::measure(operation);
        let count = AllocationCount {
            value: snapshot.global.allocations(),
        };
        (output, count)
    }
    #[cfg(miri)]
    {
        let output = operation();
        (output, AllocationCount {})
    }
}

/// Retained-footprint instrument for the pool itself
/// (`MOI-LOCAL-QUEUE-FOOTPRINT-2026-08-28`).
///
/// Pointer-identity ledgers cover both the installed global allocator and
/// direct Mnemosyne allocation hooks. Windows isolate scheduler construction,
/// first and repeated fan-out, and shutdown. The construction oracle requires
/// one partitioned global injector and four local deque planes per worker;
/// fan-out remains allocation-free after initialization. Run this test alone
/// with `--run-ignored ignored-only --no-capture` so unrelated test-process
/// activity cannot enter its process-wide window.
#[cfg(all(not(miri), feature = "mnemosyne"))]
#[test]
#[ignore = "measurement probe for retained scheduler footprint"]
fn pool_retained_footprint_attribution() {
    let _serial = HARNESS.lock().expect("harness lock");
    let _hooks = allocation_ledger::MnemosyneHooks::install();
    const JOBS: usize = 256;
    let workers = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
    println!("workers = {workers}, jobs per fan-out = {JOBS}");

    let (scheduler, construction) = allocation_ledger::footprint_window("construction", || {
        ThreadScheduler::new(workers, "pool-footprint").expect("the probe scheduler must start")
    });

    // Each global injector stores one sequence word beside the 17-word queued
    // `(Priority, ScheduledJob)` value. Construction may retain other smaller
    // global blocks, so this oracle filters by the derived injector size.
    const SLOT_WORDS: usize = 18;
    let partition_slots =
        1usize << (moirai_core::executor::config::DEFAULT_GLOBAL_QUEUE_CAPACITY / workers).ilog2();
    let queue_buffer_bytes = partition_slots * SLOT_WORDS * core::mem::size_of::<usize>();
    assert_eq!(
        construction.global.block_count(queue_buffer_bytes),
        workers,
        "construction must retain one partition-sized global injector per worker"
    );

    // `ScheduledJob` is independently pinned to 16 words by its crate-local
    // layout test. Each worker owns four local Chase-Lev planes.
    const SCHEDULED_JOB_WORDS: usize = 16;
    const LOCAL_PLANES: usize = 4;
    let local_buffer_bytes = moirai_core::executor::config::DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY
        * SCHEDULED_JOB_WORDS
        * core::mem::size_of::<usize>();
    assert_eq!(
        construction.direct.total_blocks(),
        workers * LOCAL_PLANES,
        "construction must retain four direct local-queue buffers per worker"
    );
    assert_eq!(
        construction.direct.block_count(local_buffer_bytes),
        workers * LOCAL_PLANES,
        "each direct local-queue buffer must match the configured initial capacity"
    );

    let hits: Vec<AtomicUsize> = (0..JOBS).map(|_| AtomicUsize::new(0)).collect();
    let ((), trivial) = allocation_ledger::footprint_window("first trivial fan-out", || {
        scheduler
            .for_each_indexed::<SyncTask, _>(Priority::Normal, None, JOBS, |index| {
                hits[index].fetch_add(1, Ordering::Relaxed);
            })
            .expect("the trivial fan-out must complete");
    });
    assert_eq!(
        (
            trivial.global.allocations(),
            trivial.global.retained(),
            trivial.direct.allocations(),
            trivial.direct.retained(),
        ),
        (0, 0, 0, 0),
        "the first indexed fan-out must allocate nothing: it shares one closure by reference"
    );

    // 32 words of capture: past the inline-job words, the shape closest to a
    // capture-heavy consumer.
    let payload: [usize; 32] = std::array::from_fn(|i| i);
    let ((), wide) = allocation_ledger::footprint_window("first wide-capture fan-out", || {
        scheduler
            .for_each_indexed::<SyncTask, _>(Priority::Normal, None, JOBS, move |index| {
                std::hint::black_box(payload[index % 32]);
            })
            .expect("the wide-capture fan-out must complete");
    });
    assert_eq!(
        (
            wide.global.allocations(),
            wide.global.retained(),
            wide.direct.allocations(),
            wide.direct.retained(),
        ),
        (0, 0, 0, 0),
        "a capture past the inline-job words changes nothing: indexed fan-out has no per-job storage"
    );
    let ((), repeat) = allocation_ledger::footprint_window("repeat wide-capture fan-out", || {
        scheduler
            .for_each_indexed::<SyncTask, _>(Priority::Normal, None, JOBS, move |index| {
                std::hint::black_box(payload[index % 32]);
            })
            .expect("the repeat fan-out must complete");
    });
    assert_eq!(
        (
            repeat.global.allocations(),
            repeat.global.retained(),
            repeat.direct.allocations(),
            repeat.direct.retained(),
        ),
        (0, 0, 0, 0)
    );

    allocation_ledger::footprint_window("shutdown", || scheduler.shutdown());
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
