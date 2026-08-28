//! Allocation contract for warmed indexed operations.

use moirai_core::Priority;
use moirai_executor::{SyncTask, ThreadScheduler};
#[cfg(all(not(miri), feature = "mnemosyne"))]
use moirai_scheduler::{ChaseLevDeque, DequeCapacity};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(miri))]
#[path = "indexed_allocation_contract/allocation_ledger.rs"]
mod allocation_ledger;

/// Serializes the two tests: both drive the same global counting state, and
/// the plain `cargo test` harness runs tests as threads of one process.
static HARNESS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(not(miri), feature = "mnemosyne"))]
const LOCAL_QUEUE_CAPACITIES: &[usize] = &[16, 32, 64, 128, 256];
#[cfg(all(not(miri), feature = "mnemosyne"))]
const LOCAL_QUEUE_GROWTH_ITEMS: usize = 257;
#[cfg(all(not(miri), feature = "mnemosyne"))]
const LOCAL_QUEUE_PLANES: usize = 4;
#[cfg(all(not(miri), feature = "mnemosyne"))]
const SCHEDULED_JOB_WORDS: usize = 16;

#[cfg(all(not(miri), feature = "mnemosyne"))]
#[repr(transparent)]
struct LocalQueueProbe([usize; SCHEDULED_JOB_WORDS]);

#[cfg(all(not(miri), feature = "mnemosyne"))]
const _: () = assert!(
    core::mem::size_of::<LocalQueueProbe>() == SCHEDULED_JOB_WORDS * core::mem::size_of::<usize>()
);

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

#[cfg(all(not(miri), feature = "mnemosyne"))]
fn assert_no_window_allocations(snapshot: &allocation_ledger::FootprintSnapshot, operation: &str) {
    assert_eq!(
        (
            snapshot.global.allocations(),
            snapshot.global.retained(),
            snapshot.direct.allocations(),
            snapshot.direct.retained(),
        ),
        (0, 0, 0, 0),
        "{operation} must allocate nothing"
    );
}

#[cfg(all(not(miri), feature = "mnemosyne"))]
fn probe_scheduler_capacity(
    workers: usize,
    capacity: usize,
    hits: &[AtomicUsize],
    payload: [usize; 32],
) {
    let construction_label = format!("construction capacity {capacity}");
    let (scheduler, construction) =
        allocation_ledger::footprint_window(&construction_label, || {
            ThreadScheduler::<256>::new_with_local_queue_initial_capacity(
                workers,
                "pool-footprint",
                capacity,
            )
            .expect("the probe scheduler must start")
        });

    const SLOT_WORDS: usize = 18;
    let partition_slots =
        1usize << (moirai_core::executor::config::DEFAULT_GLOBAL_QUEUE_CAPACITY / workers).ilog2();
    let queue_buffer_bytes = partition_slots * SLOT_WORDS * core::mem::size_of::<usize>();
    assert_eq!(
        construction.global.block_count(queue_buffer_bytes),
        workers,
        "construction must retain one partition-sized global injector per worker"
    );

    let local_buffer_bytes = capacity * core::mem::size_of::<LocalQueueProbe>();
    assert_eq!(
        construction.direct.total_blocks(),
        workers * LOCAL_QUEUE_PLANES,
        "construction must retain four direct local-queue buffers per worker"
    );
    assert_eq!(
        construction.direct.block_count(local_buffer_bytes),
        workers * LOCAL_QUEUE_PLANES,
        "each direct local-queue buffer must match the candidate capacity"
    );

    let first_label = format!("first fan-out capacity {capacity}");
    let ((), first) = allocation_ledger::footprint_window(&first_label, || {
        scheduler
            .for_each_indexed::<SyncTask, _>(Priority::Normal, None, hits.len(), |index| {
                hits[index].fetch_add(1, Ordering::Relaxed);
            })
            .expect("the first fan-out must complete");
    });
    assert_no_window_allocations(&first, "the first indexed fan-out");

    let warm_label = format!("warm fan-out capacity {capacity}");
    let ((), warm) = allocation_ledger::footprint_window(&warm_label, || {
        scheduler
            .for_each_indexed::<SyncTask, _>(Priority::Normal, None, hits.len(), move |index| {
                std::hint::black_box(payload[index % payload.len()]);
            })
            .expect("the warm fan-out must complete");
    });
    assert_no_window_allocations(&warm, "the warm wide-capture fan-out");

    let shutdown_label = format!("shutdown capacity {capacity}");
    let ((), shutdown) =
        allocation_ledger::footprint_window(&shutdown_label, || scheduler.shutdown());
    assert_no_window_allocations(&shutdown, "scheduler shutdown");
}

#[cfg(all(not(miri), feature = "mnemosyne"))]
fn local_queue_probe_sum(deque: &mut ChaseLevDeque<LocalQueueProbe>, count: usize) -> usize {
    for value in 0..count {
        deque.push(LocalQueueProbe(
            [value.wrapping_add(1); SCHEDULED_JOB_WORDS],
        ));
    }

    let mut sum = 0usize;
    while let Some(payload) = deque.pop() {
        sum = sum.wrapping_add(payload.0[0]);
    }
    sum
}

#[cfg(all(not(miri), feature = "mnemosyne"))]
fn expected_growth_capacities(initial: usize, items: usize) -> Vec<usize> {
    let final_capacity = items
        .checked_add(1)
        .expect("probe item count must fit usize")
        .next_power_of_two();
    core::iter::successors(Some(initial * 2), |capacity| Some(capacity * 2))
        .take_while(|capacity| *capacity <= final_capacity)
        .collect()
}

#[cfg(all(not(miri), feature = "mnemosyne"))]
fn probe_first_growth(capacity: usize) {
    let deque_capacity =
        DequeCapacity::try_from(capacity).expect("candidate capacity must be representable");
    let mut deque = ChaseLevDeque::new(deque_capacity);
    let growth_capacities = expected_growth_capacities(capacity, LOCAL_QUEUE_GROWTH_ITEMS);
    let growth_label = format!("first growth capacity {capacity}");
    let (sum, growth) = allocation_ledger::footprint_window(&growth_label, || {
        local_queue_probe_sum(&mut deque, LOCAL_QUEUE_GROWTH_ITEMS)
    });
    assert_eq!(
        sum,
        LOCAL_QUEUE_GROWTH_ITEMS * (LOCAL_QUEUE_GROWTH_ITEMS + 1) / 2
    );
    assert_eq!(growth.direct.allocations(), growth_capacities.len());
    let expected_retained = growth_capacities.iter().fold(0usize, |total, slots| {
        total + slots * core::mem::size_of::<LocalQueueProbe>()
    });
    assert_eq!(
        growth.direct.retained(),
        isize::try_from(expected_retained).expect("growth footprint must fit isize")
    );
    for grown_capacity in &growth_capacities {
        assert_eq!(
            growth
                .direct
                .block_count(grown_capacity * core::mem::size_of::<LocalQueueProbe>()),
            1,
            "each doubling step must retain one direct queue buffer"
        );
    }

    let warm_label = format!("warm queue capacity {capacity}");
    let (sum, warm) = allocation_ledger::footprint_window(&warm_label, || {
        local_queue_probe_sum(&mut deque, LOCAL_QUEUE_GROWTH_ITEMS)
    });
    assert_eq!(
        sum,
        LOCAL_QUEUE_GROWTH_ITEMS * (LOCAL_QUEUE_GROWTH_ITEMS + 1) / 2
    );
    assert_no_window_allocations(&warm, "the warmed local queue");

    drop(deque);
    let lifecycle_label = format!("full lifecycle capacity {capacity}");
    let (sum, lifecycle) = allocation_ledger::footprint_window(&lifecycle_label, || {
        let deque_capacity =
            DequeCapacity::try_from(capacity).expect("candidate capacity must be representable");
        let mut lifecycle_deque = ChaseLevDeque::new(deque_capacity);
        local_queue_probe_sum(&mut lifecycle_deque, LOCAL_QUEUE_GROWTH_ITEMS)
    });
    assert_eq!(
        sum,
        LOCAL_QUEUE_GROWTH_ITEMS * (LOCAL_QUEUE_GROWTH_ITEMS + 1) / 2
    );
    assert_eq!(
        lifecycle.direct.allocations(),
        growth_capacities.len() + 1,
        "full lifecycle must allocate the initial and each grown direct buffer"
    );
    assert_eq!(
        lifecycle.direct.retained(),
        0,
        "dropping the queue must release every direct buffer"
    );
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
    // The contract under test is per-worker structure, so a bounded worker
    // set witnesses it on any host; more workers add allocations, not
    // information. The bound keeps the probe inside the ledger everywhere:
    // construction records ~30 allocations per worker (684 measured at 24),
    // so 64 workers sit within the 2,048-slot ledger with 3x headroom, while
    // an unclamped 256-core host would trip the ledger's overflow assert and
    // a host past 4,096 would fail queue partitioning outright.
    const PROBE_WORKERS_MAX: usize = 64;
    let workers = std::thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .min(PROBE_WORKERS_MAX);
    println!("workers = {workers}, jobs per fan-out = {JOBS}");

    let hits: Vec<AtomicUsize> = (0..JOBS).map(|_| AtomicUsize::new(0)).collect();
    let payload: [usize; 32] = std::array::from_fn(|i| i);
    for &capacity in LOCAL_QUEUE_CAPACITIES {
        probe_scheduler_capacity(workers, capacity, &hits, payload);
        probe_first_growth(capacity);
    }
    assert!(hits
        .iter()
        .all(|h| { h.load(Ordering::Relaxed) == LOCAL_QUEUE_CAPACITIES.len() }));
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
