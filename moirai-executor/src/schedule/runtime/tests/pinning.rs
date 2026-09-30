//! Worker pinning: planned assignments, fail-closed construction, and the
//! processor each pinned worker actually runs on.

use super::*;
use crate::schedule::runtime::scheduler::placement::{PinProbe, TestBinder};
use moirai_core::{error::PlacementFailure, executor::WorkerPlacement};
use themis::BindError;

type BindLog = Arc<Mutex<Vec<(usize, u32)>>>;

/// A bind function that logs `(worker, processor)` and fails the workers for
/// which `refusal` returns an error.
fn logging_binder(
    log: &BindLog,
    refusal: impl Fn(usize, u32) -> Option<BindError> + Send + Sync + 'static,
) -> TestBinder {
    let log = Arc::clone(log);
    Arc::new(move |worker, processor| {
        log.lock().unwrap().push((worker, processor));
        refusal(worker, processor).map_or(Ok(()), Err)
    })
}

fn sorted_log(log: &BindLog) -> Vec<(usize, u32)> {
    let mut recorded = log.lock().unwrap().clone();
    recorded.sort_unstable();
    recorded
}

fn probe(pairs: Vec<(u32, usize)>, binder: TestBinder, drops: &Arc<AtomicUsize>) -> PinProbe {
    PinProbe {
        pairs,
        binder,
        lifetime_owner: Box::new(DropProbe {
            drops: Arc::clone(drops),
        }),
    }
}

#[test]
fn bind_failure_names_the_lowest_failing_worker_and_no_worker_survives() {
    let log = BindLog::default();
    let drops = Arc::new(AtomicUsize::new(0));
    // Workers 1 and 3 are refused; the pairs arrive unsorted, so the plan
    // sorts before assigning: worker `i` receives the `i`-th smallest id.
    let binder = logging_binder(&log, |worker, _| match worker {
        1 => Some(BindError::Os { code: 87 }),
        3 => Some(BindError::Unsupported),
        _ => None,
    });
    let pairs = vec![(13, 1), (10, 0), (12, 1), (11, 0)];

    let result = ThreadScheduler::<256>::with_pin_probe(4, probe(pairs, binder, &drops));

    assert_eq!(
        result.err(),
        Some(ExecutorError::WorkerPlacementFailed {
            worker: 1,
            processor: 11,
            cause: PlacementFailure::Os { code: 87 },
        })
    );
    assert_eq!(
        drops.load(Ordering::Acquire),
        1,
        "the failed construction released its retained state, so every worker exited"
    );
    assert_eq!(
        sorted_log(&log),
        [(0, 10), (1, 11), (2, 12), (3, 13)],
        "every worker attempted its own planned processor"
    );
}

#[test]
fn out_of_range_refusal_reports_its_worker_and_processor() {
    let log = BindLog::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let binder = logging_binder(&log, |worker, processor| {
        (worker == 2).then_some(BindError::OutOfRange { processor })
    });

    let result = ThreadScheduler::<256>::with_pin_probe(
        3,
        probe(vec![(4, 0), (5, 0), (40_000, 1)], binder, &drops),
    );

    assert_eq!(
        result.err(),
        Some(ExecutorError::WorkerPlacementFailed {
            worker: 2,
            processor: 40_000,
            cause: PlacementFailure::OutOfRange,
        })
    );
    assert_eq!(drops.load(Ordering::Acquire), 1);
}

#[test]
fn bound_workers_publish_the_planned_nodes_and_enter_their_loops() {
    let log = BindLog::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let binder = logging_binder(&log, |_, _| None);

    let scheduler = ThreadScheduler::<256>::with_pin_probe(
        4,
        probe(vec![(10, 0), (11, 0), (12, 1), (13, 1)], binder, &drops),
    )
    .unwrap();

    assert_eq!(sorted_log(&log), [(0, 10), (1, 11), (2, 12), (3, 13)]);
    assert_eq!(
        &*scheduler.inner.worker_numa_nodes,
        &[Some(0), Some(0), Some(1), Some(1)]
    );
    assert!(
        scheduler
            .inner
            .workers
            .iter()
            .all(|worker| worker.placement.get() == Some(&Ok(())))
    );
    let (_, release) = occupy_compute_worker(&scheduler, 0);
    release.send(()).unwrap();
    scheduler.shutdown();
    assert_eq!(
        drops.load(Ordering::Acquire),
        0,
        "the scheduler still owns its state"
    );
    drop(scheduler);
    assert_eq!(drops.load(Ordering::Acquire), 1);
}

#[test]
fn workers_beyond_the_processor_count_share_processors_and_a_single_node_publishes_none() {
    let log = BindLog::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let binder = logging_binder(&log, |_, _| None);

    let scheduler =
        ThreadScheduler::<256>::with_pin_probe(3, probe(vec![(2, 0), (5, 0)], binder, &drops))
            .unwrap();

    assert_eq!(sorted_log(&log), [(0, 2), (1, 5), (2, 2)]);
    assert!(
        scheduler
            .inner
            .worker_numa_nodes
            .iter()
            .all(Option::is_none),
        "one represented node gives the locality pass nothing to distinguish"
    );
    scheduler.shutdown();
}

#[test]
fn a_topology_with_no_processor_is_an_invalid_configuration() {
    let log = BindLog::default();
    let drops = Arc::new(AtomicUsize::new(0));
    let binder = logging_binder(&log, |_, _| None);

    let result = ThreadScheduler::<256>::with_pin_probe(2, probe(Vec::new(), binder, &drops));

    assert_eq!(result.err(), Some(ExecutorError::InvalidConfiguration));
    assert!(log.lock().unwrap().is_empty(), "no worker was started");
}

#[test]
fn unbound_workers_never_bind_and_publish_no_node() {
    let scheduler = scheduler_with_queue_config::<256>(
        2,
        "unbound-default",
        ExecutorConfig::default().max_global_queue_size,
        DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
    )
    .unwrap();

    assert_eq!(
        ExecutorConfig::default().worker_placement,
        WorkerPlacement::Unbound
    );
    assert!(
        scheduler
            .inner
            .worker_numa_nodes
            .iter()
            .all(Option::is_none)
    );
    assert!(
        scheduler
            .inner
            .workers
            .iter()
            .all(|worker| worker.placement.get().is_none()),
        "an unbound worker publishes no binding outcome, so it never called bind"
    );
    scheduler.shutdown();
}

/// The logical processor executing the calling thread, in the flattened
/// numbering `themis` binds with.
#[cfg(any(windows, target_os = "linux"))]
mod os_processor {
    #[cfg(windows)]
    pub(super) fn current() -> u32 {
        #[repr(C)]
        struct ProcessorNumber {
            group: u16,
            number: u8,
            _reserved: u8,
        }

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcessorNumberEx(processor: *mut ProcessorNumber);
        }

        let mut processor = ProcessorNumber {
            group: 0,
            number: 0,
            _reserved: 0,
        };
        // SAFETY: the pointer is valid, aligned, and exclusively borrowed for
        // the call; the function only writes the calling thread's processor.
        unsafe { GetCurrentProcessorNumberEx(&raw mut processor) };
        u32::from(processor.group) * 64 + u32::from(processor.number)
    }

    #[cfg(target_os = "linux")]
    pub(super) fn current() -> u32 {
        unsafe extern "C" {
            fn sched_getcpu() -> i32;
        }

        // SAFETY: `sched_getcpu` takes no arguments and reads only the
        // calling thread's scheduling state.
        let cpu = unsafe { sched_getcpu() };
        u32::try_from(cpu).expect("invariant: sched_getcpu succeeds on a running thread")
    }
}

/// Each pinned worker runs on the processor the plan assigned it.
///
/// The operating-system query exists on Windows and Linux only; no other
/// target has a binding backend to verify.
#[cfg(any(windows, target_os = "linux"))]
#[test]
fn pinned_workers_run_on_their_planned_processors() {
    const WORKERS: usize = 2;

    let mut processors: Vec<u32> = themis::CpuTopology::detect()
        .expect("invariant: the test host reports its topology")
        .processor_node_pairs()
        .map(|(processor, _)| processor)
        .collect();
    processors.sort_unstable();

    let scheduler = ThreadScheduler::<256>::from_executor_config(&ExecutorConfig {
        worker_threads: WORKERS,
        thread_name_prefix: "pinned-os".into(),
        worker_placement: WorkerPlacement::Pinned,
        ..ExecutorConfig::default()
    })
    .expect("a pinned scheduler starts on a host that permits its planned processors");

    let (report_sender, report_receiver) = mpsc::sync_channel(0);
    let mut releases = Vec::new();
    for locality_hint in 0..WORKERS {
        let (release_sender, release_receiver) = mpsc::sync_channel(0);
        let report_sender = report_sender.clone();
        scheduler
            .schedule::<SyncTask, _>(Priority::Critical, Some(locality_hint), move |worker_id| {
                report_sender
                    .send((worker_id, os_processor::current()))
                    .expect("test observer remains connected");
                release_receiver
                    .recv()
                    .expect("test controller releases the occupied worker");
            })
            .expect("gate job must be admitted");
        releases.push(release_sender);
    }

    let mut reported = [None; WORKERS];
    for _ in 0..WORKERS {
        let (worker_id, processor) = report_receiver
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("each gate job must report before the deadline");
        assert!(
            reported[worker_id].replace(processor).is_none(),
            "each gate job must occupy a distinct worker"
        );
    }
    for release in releases {
        release.send(()).unwrap();
    }

    for (worker_id, processor) in reported.into_iter().enumerate() {
        assert_eq!(
            processor,
            Some(processors[worker_id % processors.len()]),
            "worker {worker_id} runs on the processor it was planned onto"
        );
    }
    scheduler.shutdown();
}
