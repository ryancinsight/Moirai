//! Unit tests for the thread scheduler runtime.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc, Barrier, Condvar, Mutex,
};
use std::time::Duration;

use super::types::{get_current_worker_id, ThreadScheduler};
use crate::schedule::{AsyncTask, BlockingTask, SyncTask};
use moirai_core::{
    error::{ExecutorError, TaskError},
    executor::{config::DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY, ExecutorConfig},
    Priority,
};

const TEST_ADMISSION_CAPACITY: usize = 8;
const TEST_EVENT_DEADLINE: Duration = Duration::from_secs(5);

struct DropProbe {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct DropSignal {
    state: Arc<(Mutex<bool>, Condvar)>,
}

impl Drop for DropSignal {
    fn drop(&mut self) {
        let (lock, signal) = &*self.state;
        let mut dropped = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *dropped = true;
        signal.notify_all();
    }
}

fn scheduler_with_queue_config<const BLOCKING_QUEUE_CAPACITY: usize>(
    worker_count: usize,
    name: &str,
    max_global_queue_size: usize,
    local_queue_initial_capacity: usize,
) -> Result<ThreadScheduler<BLOCKING_QUEUE_CAPACITY>, ExecutorError> {
    ThreadScheduler::<BLOCKING_QUEUE_CAPACITY>::from_executor_config(&ExecutorConfig {
        worker_threads: worker_count,
        max_global_queue_size,
        local_queue_initial_capacity,
        thread_name_prefix: name.into(),
        ..ExecutorConfig::default()
    })
}

fn scheduler_with_bounded_admission(name: &str) -> ThreadScheduler<256> {
    scheduler_with_queue_config::<256>(
        1,
        name,
        TEST_ADMISSION_CAPACITY,
        DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
    )
    .unwrap()
}

fn occupy_compute_worker(
    scheduler: &ThreadScheduler,
    locality_hint: usize,
) -> (usize, mpsc::SyncSender<()>) {
    let (started_sender, started_receiver) = mpsc::sync_channel(0);
    let (release_sender, release_receiver) = mpsc::sync_channel(0);
    scheduler
        .schedule::<SyncTask, _>(Priority::Critical, Some(locality_hint), move |worker_id| {
            started_sender
                .send(worker_id)
                .expect("test observer remains connected");
            release_receiver
                .recv()
                .expect("test controller releases the occupied worker");
        })
        .expect("gate job must be admitted");
    let worker_id = started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("gate job must start before the test deadline");
    (worker_id, release_sender)
}

#[test]
fn scheduler_runs_all_work_classes_through_one_facade() {
    let scheduler = ThreadScheduler::new(2, "test-scheduler").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();

    {
        let completed = Arc::clone(&completed);
        let sender = sender.clone();
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    {
        let completed = Arc::clone(&completed);
        let sender = sender.clone();
        scheduler
            .schedule::<AsyncTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    {
        let completed = Arc::clone(&completed);
        scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    for _ in 0..3 {
        receiver.recv().unwrap();
    }

    scheduler.shutdown();
    let metrics = scheduler.metrics();

    assert_eq!(completed.load(Ordering::Acquire), 3);
    assert_eq!(metrics.worker_count, 2);
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.completed_tasks, 3);
    assert_eq!(metrics.failed_tasks, 0);
}

#[test]
fn final_external_handle_drop_drains_and_releases_workers() {
    const COMPUTE_JOBS: usize = 32;
    const BLOCKING_VALUE: usize = 1_024;
    let scheduler = ThreadScheduler::new(1, "test-final-scheduler-drop").unwrap();
    let inner = Arc::downgrade(&scheduler.inner);
    let shutdown_started = Arc::new(Barrier::new(2));
    assert!(
        scheduler
            .inner
            .shutdown_started_barrier
            .set(Arc::clone(&shutdown_started))
            .is_ok(),
        "test installs one shutdown rendezvous"
    );
    let owner_drops = Arc::new(AtomicUsize::new(0));
    scheduler.retain_lifetime_owner(DropProbe {
        drops: Arc::clone(&owner_drops),
    });

    let surviving_handle = scheduler.clone();
    drop(scheduler);

    let (usable_sender, usable_receiver) = mpsc::sync_channel(1);
    surviving_handle
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            usable_sender.send(7).unwrap();
        })
        .unwrap();
    assert_eq!(
        usable_receiver.recv_timeout(TEST_EVENT_DEADLINE).unwrap(),
        7,
        "dropping a non-final handle must not stop the shared scheduler"
    );

    let (gate_started_sender, gate_started_receiver) = mpsc::sync_channel(2);
    let (compute_release_sender, compute_release_receiver) = mpsc::sync_channel(0);
    surviving_handle
        .schedule::<SyncTask, _>(Priority::Critical, Some(0), move |_| {
            gate_started_sender.send(()).unwrap();
            compute_release_receiver.recv().unwrap();
        })
        .unwrap();
    let (blocking_started_sender, blocking_started_receiver) = mpsc::sync_channel(1);
    let (blocking_release_sender, blocking_release_receiver) = mpsc::sync_channel(0);
    surviving_handle
        .schedule::<BlockingTask, _>(Priority::Critical, Some(0), move |_| {
            blocking_started_sender.send(()).unwrap();
            blocking_release_receiver.recv().unwrap();
        })
        .unwrap();
    gate_started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("compute gate must occupy the sole worker");
    blocking_started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("blocking gate must occupy the sole blocking worker");

    let completed = Arc::new(AtomicUsize::new(0));
    let capture_drops = Arc::new(AtomicUsize::new(0));

    for value in 1..=COMPUTE_JOBS {
        let completed = Arc::clone(&completed);
        let capture = DropProbe {
            drops: Arc::clone(&capture_drops),
        };
        surviving_handle
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(value, Ordering::Relaxed);
                drop(capture);
            })
            .unwrap();
    }

    let blocking_completed = Arc::clone(&completed);
    let blocking_capture = DropProbe {
        drops: Arc::clone(&capture_drops),
    };
    surviving_handle
        .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
            blocking_completed.fetch_add(BLOCKING_VALUE, Ordering::Relaxed);
            drop(blocking_capture);
        })
        .unwrap();

    let (drop_sender, drop_receiver) = mpsc::sync_channel(1);
    let drop_thread = std::thread::spawn(move || {
        drop(surviving_handle);
        drop_sender.send(()).unwrap();
    });
    shutdown_started.wait();
    let live_inner = inner
        .upgrade()
        .expect("shutdown rendezvous retains scheduler state");
    assert_eq!(
        live_inner.pending_tasks.load(Ordering::Acquire),
        COMPUTE_JOBS,
        "all compute values must still be queued when final drop publishes shutdown"
    );
    assert_eq!(
        live_inner.blocking_pending_tasks.load(Ordering::Acquire),
        1,
        "the blocking value must still be queued when final drop publishes shutdown"
    );
    drop(live_inner);
    compute_release_sender.send(()).unwrap();
    blocking_release_sender.send(()).unwrap();
    drop_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("final scheduler drop must synchronously release its workers");
    drop_thread.join().unwrap();

    assert_eq!(
        completed.load(Ordering::Relaxed),
        COMPUTE_JOBS * (COMPUTE_JOBS + 1) / 2 + BLOCKING_VALUE
    );
    assert_eq!(
        capture_drops.load(Ordering::Relaxed),
        COMPUTE_JOBS + 1,
        "every admitted capture must be released exactly once"
    );
    assert_eq!(
        owner_drops.load(Ordering::Relaxed),
        1,
        "scheduler-owned facade state must be released exactly once"
    );
    assert!(
        inner.upgrade().is_none(),
        "the final external drop must join workers and release scheduler state"
    );
}

#[test]
fn concurrent_compute_and_blocking_shutdown_cannot_cross_join() {
    let scheduler = ThreadScheduler::new(1, "test-concurrent-shutdown").unwrap();
    let callers_ready = Arc::new(Barrier::new(3));
    let (completed_sender, completed_receiver) = mpsc::sync_channel(2);

    let compute_scheduler = scheduler.clone();
    let compute_ready = Arc::clone(&callers_ready);
    let compute_completed = completed_sender.clone();
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, Some(0), move |_| {
            compute_ready.wait();
            compute_scheduler.shutdown();
            compute_completed.send(1).unwrap();
        })
        .unwrap();

    let blocking_scheduler = scheduler.clone();
    let blocking_ready = Arc::clone(&callers_ready);
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, Some(0), move |_| {
            blocking_ready.wait();
            blocking_scheduler.shutdown();
            completed_sender.send(2).unwrap();
        })
        .unwrap();

    callers_ready.wait();
    let mut completions = [
        completed_receiver
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("one shutdown caller must complete"),
        completed_receiver
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("both shutdown callers must complete"),
    ];
    completions.sort_unstable();
    assert_eq!(completions, [1, 2]);
    scheduler.shutdown();
}

#[test]
fn final_compute_owner_returns_before_blocking_dependency() {
    let scheduler = ThreadScheduler::new(1, "test-worker-final-owner").unwrap();
    let inner = Arc::downgrade(&scheduler.inner);
    let released = Arc::new((Mutex::new(false), Condvar::new()));
    scheduler.retain_lifetime_owner(DropSignal {
        state: Arc::clone(&released),
    });
    let final_handle = scheduler.clone();

    let (blocking_started_sender, blocking_started_receiver) = mpsc::sync_channel(0);
    let (blocking_release_sender, blocking_release_receiver) = mpsc::sync_channel(0);
    let (blocking_completed_sender, blocking_completed_receiver) = mpsc::sync_channel(1);
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, Some(0), move |_| {
            blocking_started_sender.send(()).unwrap();
            blocking_release_receiver.recv().unwrap();
            blocking_completed_sender.send(()).unwrap();
        })
        .unwrap();
    blocking_started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("blocking dependency must start before final-owner drop");

    let (compute_started_sender, compute_started_receiver) = mpsc::sync_channel(0);
    let (compute_release_sender, compute_release_receiver) = mpsc::sync_channel(0);
    let (compute_completed_sender, compute_completed_receiver) = mpsc::sync_channel(1);
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, Some(0), move |_| {
            compute_started_sender.send(()).unwrap();
            compute_release_receiver.recv().unwrap();
            drop(final_handle);
            blocking_release_sender.send(()).unwrap();
            compute_completed_sender.send(()).unwrap();
        })
        .unwrap();
    compute_started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("compute final owner must start before release");

    drop(scheduler);
    compute_release_sender
        .send(())
        .expect("test controller releases the compute final owner");
    compute_completed_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("worker-owned shutdown must return before its peer dependency");
    blocking_completed_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("accepted blocking dependency must drain exactly once");

    let (lock, signal) = &*released;
    let dropped = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (dropped, timeout) = signal
        .wait_timeout_while(dropped, TEST_EVENT_DEADLINE, |dropped| !*dropped)
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(
        !timeout.timed_out() && *dropped,
        "worker-owned final drop must release scheduler state"
    );
    assert!(
        inner.upgrade().is_none(),
        "worker-owned final drop must release every scheduler Arc"
    );
}

#[test]
fn scheduler_numa_policy_controls_worker_assignments() {
    let scheduler = scheduler_with_queue_config::<256>(
        2,
        "numa-disabled",
        ExecutorConfig::default().max_global_queue_size,
        DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
    )
    .unwrap();

    assert!(scheduler
        .inner
        .worker_numa_nodes
        .iter()
        .all(|node| node.is_none()));

    scheduler.shutdown();
}

#[test]
fn numa_steal_falls_back_to_a_cross_node_victim() {
    let scheduler = ThreadScheduler::<256>::with_worker_numa_nodes(
        vec![Some(0), Some(0), Some(1)].into_boxed_slice(),
        "numa-cross-node-fallback",
    )
    .unwrap();
    assert_eq!(
        &*scheduler.inner.worker_numa_nodes,
        &[Some(0), Some(0), Some(1)]
    );

    let mut releases = [None, None, None];
    for locality_hint in 0..3 {
        let (worker_id, release) = occupy_compute_worker(&scheduler, locality_hint);
        assert!(
            releases[worker_id].replace(release).is_none(),
            "each gate must occupy a distinct worker"
        );
    }

    releases[0]
        .take()
        .expect("worker zero must be occupied")
        .send(())
        .expect("worker zero gate remains connected");

    let (executed_sender, executed_receiver) = mpsc::sync_channel(1);
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, Some(2), move |worker_id| {
            executed_sender
                .send(worker_id)
                .expect("test observer remains connected");
        })
        .unwrap();
    assert_eq!(
        executed_receiver
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("cross-node victim must be reached before the deadline"),
        0,
        "the sole free worker must execute the cross-node victim's job"
    );

    for release in releases.into_iter().flatten() {
        release
            .send(())
            .expect("occupied peer gate remains connected");
    }
    scheduler
        .join()
        .expect("cross-node fallback workload must join cleanly");
    scheduler.shutdown();
}

#[test]
fn configured_global_capacity_is_partitioned_without_exceeding_the_bound() {
    let scheduler = scheduler_with_queue_config::<256>(
        3,
        "partitioned",
        1000,
        DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
    )
    .unwrap();

    let capacities = scheduler
        .inner
        .workers
        .iter()
        .map(|worker| worker.queues.injector_capacity())
        .collect::<Vec<_>>();

    assert_eq!(capacities, vec![256; 3]);
    assert_eq!(capacities.into_iter().sum::<usize>(), 768);
    scheduler.shutdown();
}

#[test]
fn configured_local_capacity_reaches_every_worker_after_normalization() {
    let scheduler = scheduler_with_queue_config::<256>(
        3,
        "local-capacity",
        ExecutorConfig::default().max_global_queue_size,
        17,
    )
    .unwrap();

    let capacities = scheduler
        .inner
        .workers
        .iter()
        .map(|worker| worker.queues.local_queue_initial_capacity())
        .collect::<Vec<_>>();

    assert_eq!(capacities, vec![32; 3]);
    scheduler.shutdown();
}

#[test]
fn measured_default_local_capacity_reaches_every_worker() {
    let scheduler = ThreadScheduler::new(3, "default-local-capacity").unwrap();
    let capacities = scheduler
        .inner
        .workers
        .iter()
        .map(|worker| worker.queues.local_queue_initial_capacity())
        .collect::<Vec<_>>();

    assert_eq!(capacities, vec![128; 3]);
    scheduler.shutdown();
}

#[test]
fn unrepresentable_local_capacity_is_rejected_before_worker_startup() {
    for requested in [isize::MAX as usize, usize::MAX] {
        let result = scheduler_with_queue_config::<256>(
            2,
            "invalid-local-capacity",
            ExecutorConfig::default().max_global_queue_size,
            requested,
        );

        assert!(matches!(
            result,
            Err(ExecutorError::InvalidLocalQueueInitialCapacity { requested: actual })
                if actual == requested
        ));
    }
}

#[test]
fn local_queue_growth_and_cross_worker_steal_execute_each_job_once() {
    const JOBS: usize = 257;

    let scheduler =
        ThreadScheduler::new_with_local_queue_initial_capacity(2, "local-growth-steal", 16)
            .unwrap();
    let (owner_lane, owner_release) = occupy_compute_worker(&scheduler, 0);
    let (thief_lane, thief_release) = occupy_compute_worker(&scheduler, owner_lane + 1);
    assert_ne!(owner_lane, thief_lane);

    let visits: Arc<[AtomicUsize]> = (0..JOBS)
        .map(|_| AtomicUsize::new(0))
        .collect::<Vec<_>>()
        .into();
    let first_stolen = Arc::new(AtomicBool::new(false));
    let (stolen_sender, stolen_receiver) = mpsc::sync_channel(1);
    for index in 0..JOBS {
        let visits = Arc::clone(&visits);
        let first_stolen = Arc::clone(&first_stolen);
        let stolen_sender = stolen_sender.clone();
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, Some(owner_lane), move |worker_id| {
                visits[index].fetch_add(1, Ordering::AcqRel);
                if !first_stolen.swap(true, Ordering::AcqRel) {
                    stolen_sender
                        .send((index, worker_id))
                        .expect("steal observer remains connected");
                }
            })
            .unwrap();
    }

    let (marker_started_sender, marker_started_receiver) = mpsc::sync_channel(0);
    let (marker_release_sender, marker_release_receiver) = mpsc::sync_channel(0);
    scheduler
        .schedule::<SyncTask, _>(Priority::High, Some(owner_lane), move |_| {
            marker_started_sender
                .send(())
                .expect("marker observer remains connected");
            marker_release_receiver
                .recv()
                .expect("test controller releases the marker");
        })
        .unwrap();

    owner_release.send(()).unwrap();
    marker_started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("owner must drain and grow its local queues");
    thief_release.send(()).unwrap();
    let (_, executing_lane) = stolen_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("the released peer must steal from the blocked owner");
    assert_eq!(executing_lane, thief_lane);
    marker_release_sender.send(()).unwrap();

    scheduler.join().unwrap();
    for (index, count) in visits.iter().enumerate() {
        assert_eq!(count.load(Ordering::Acquire), 1, "job {index}");
    }
    assert_eq!(scheduler.pending_tasks(), 0);
    assert_eq!(scheduler.metrics().completed_tasks, JOBS as u64 + 3);
    scheduler.shutdown();
}

#[test]
fn global_capacity_below_two_slots_per_worker_is_rejected() {
    let result =
        scheduler_with_queue_config::<256>(4, "invalid", 7, DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY);

    assert!(matches!(result, Err(ExecutorError::InvalidConfiguration)));
}

#[test]
fn global_capacity_supports_minimum_two_slots_per_worker() {
    let scheduler =
        scheduler_with_queue_config::<256>(4, "minimum", 8, DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY)
            .unwrap();

    assert!(scheduler
        .inner
        .workers
        .iter()
        .all(|worker| worker.queues.injector_capacity() == 2));
    scheduler.shutdown();
}

#[test]
fn saturated_admission_rolls_back_pending_and_recovers() {
    let scheduler = ThreadScheduler::<256>::new(1, "bounded-admission").unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
    started_rx.recv().unwrap();

    for _ in 0..256 {
        scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }
    let rejection = scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, None, |_| {})
        .expect_err("capacity plus one admission must fail");
    assert!(matches!(rejection, ExecutorError::ResourceExhausted(_)));
    assert_eq!(scheduler.pending_tasks(), 256);

    release_tx.send(()).unwrap();
    scheduler.join().unwrap();
    assert_eq!(scheduler.pending_tasks(), 0);

    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
        .unwrap();
    scheduler.join().unwrap();
    assert_eq!(scheduler.pending_tasks(), 0);
}

#[test]
fn saturated_indexed_admission_runs_rejected_chunks_on_caller() {
    let scheduler = scheduler_with_bounded_admission("indexed-caller-runs");
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
    started_rx.recv().unwrap();

    for _ in 0..TEST_ADMISSION_CAPACITY {
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }

    let visits: [AtomicUsize; 2] = std::array::from_fn(|_| AtomicUsize::new(0));
    scheduler
        .for_each_indexed::<SyncTask, _>(Priority::Normal, None, visits.len(), |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(visits.map(|count| count.load(Ordering::Relaxed)), [1, 1]);

    let sum = scheduler
        .map_reduce_indexed::<SyncTask, _, _, _>(
            Priority::Normal,
            None,
            2,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();
    assert_eq!(sum, 3);

    let panic_result =
        scheduler.for_each_indexed::<SyncTask, _>(Priority::Normal, None, 2, |index| {
            if index == 1 {
                panic!("caller-run chunk panic");
            }
        });
    assert_eq!(
        panic_result,
        Err(ExecutorError::SpawnFailed(TaskError::Panicked))
    );

    let reduction_panic = scheduler.map_reduce_indexed::<SyncTask, _, _, _>(
        Priority::Normal,
        None,
        2,
        0usize,
        |index| {
            if index == 1 {
                panic!("caller-run mapper panic");
            }
            index + 1
        },
        usize::wrapping_add,
    );
    assert_eq!(
        reduction_panic,
        Err(ExecutorError::SpawnFailed(TaskError::Panicked))
    );
    assert_eq!(scheduler.admission_caller_runs(), 4);

    release_tx.send(()).unwrap();
    scheduler.join().unwrap();
    assert_eq!(scheduler.pending_tasks(), 0);
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
        .unwrap();
    scheduler.join().unwrap();
    scheduler.shutdown();
}

#[test]
fn saturated_scope_admission_runs_rejected_jobs_on_caller() {
    // A scope owes its caller that every spawned job ran by the time it
    // returns. `flush` used to drop a job the admission queue rejected, so the
    // caller resumed as though borrowed work had happened when it never did —
    // silent, and invisible to the scope's own counters, which the dropped
    // job's completion token decrements either way.
    const SCOPED_JOBS: usize = 4;

    let scheduler = scheduler_with_bounded_admission("scope-caller-runs");
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
    started_rx.recv().unwrap();

    for _ in 0..TEST_ADMISSION_CAPACITY {
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }

    let caller_runs_before = scheduler.admission_caller_runs();
    let visits: [AtomicUsize; SCOPED_JOBS] = std::array::from_fn(|_| AtomicUsize::new(0));
    let lanes: [AtomicUsize; SCOPED_JOBS] = std::array::from_fn(|_| AtomicUsize::new(usize::MAX));

    scheduler
        .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
            for (index, (visit, lane)) in visits.iter().zip(lanes.iter()).enumerate() {
                scope.spawn(move |worker_id| {
                    visit.fetch_add(1, Ordering::Relaxed);
                    lane.store(worker_id, Ordering::Relaxed);
                    let _ = index;
                })?;
            }
            Ok(())
        })
        .expect("a saturated scope must still complete every spawned job");

    // Exactly once each: the refused job runs on the caller instead of being
    // dropped, and it must not also reach a worker.
    for visit in &visits {
        assert_eq!(visit.load(Ordering::Relaxed), 1);
    }
    // The caller's lane is the one past the last worker, never a worker index.
    for lane in &lanes {
        assert_eq!(lane.load(Ordering::Relaxed), scheduler.worker_count());
    }
    assert!(
        scheduler.admission_caller_runs() > caller_runs_before,
        "the caller-run backpressure event must be surfaced, not silent"
    );

    release_tx.send(()).unwrap();
    scheduler.join().unwrap();
    assert_eq!(scheduler.pending_tasks(), 0);
    scheduler.shutdown();
}

#[test]
fn saturated_scope_propagates_a_caller_run_job_panic() {
    // A job the caller runs keeps a worker's panic semantics: the scope reports
    // failure rather than unwinding through the scope body.
    let scheduler = scheduler_with_bounded_admission("scope-caller-panic");
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
    started_rx.recv().unwrap();

    for _ in 0..TEST_ADMISSION_CAPACITY {
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, |_| {})
            .unwrap();
    }

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let scoped = scheduler.scope::<SyncTask, _>(Priority::Normal, None, |scope| {
        scope.spawn(|_| panic!("caller-run scoped job panic"))?;
        Ok(())
    });
    std::panic::set_hook(previous_hook);

    assert_eq!(scoped, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));

    release_tx.send(()).unwrap();
    scheduler.join().unwrap();
    scheduler.shutdown();
}

#[test]
fn blocking_lane_preserves_compute_progress_when_full() {
    let scheduler = ThreadScheduler::new(2, "blocking-lane-progress").unwrap();
    let blocking_started = Arc::new(Barrier::new(3));
    let blocking_release = Arc::new(Barrier::new(3));

    for _ in 0..2 {
        let blocking_started = Arc::clone(&blocking_started);
        let blocking_release = Arc::clone(&blocking_release);
        scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                blocking_started.wait();
                blocking_release.wait();
            })
            .unwrap();
    }
    blocking_started.wait();

    let (compute_sender, compute_receiver) = mpsc::sync_channel(1);
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            compute_sender.send(91usize).unwrap();
        })
        .unwrap();

    assert_eq!(
        compute_receiver
            .recv()
            .expect("compute work must not wait behind blocking work"),
        91
    );
    blocking_release.wait();
    scheduler.join().unwrap();
    scheduler.shutdown();
}

#[test]
fn blocking_lane_accepts_concurrent_producers() {
    const PRODUCERS: usize = 4;
    const JOBS_PER_PRODUCER: usize = 32;
    let scheduler = ThreadScheduler::new(PRODUCERS, "blocking-lane-producers").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        for _ in 0..PRODUCERS {
            let scheduler = scheduler.clone();
            let completed = Arc::clone(&completed);
            scope.spawn(move || {
                for _ in 0..JOBS_PER_PRODUCER {
                    let completed = Arc::clone(&completed);
                    scheduler
                        .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                            completed.fetch_add(1, Ordering::Relaxed);
                        })
                        .unwrap();
                }
            });
        }
    });

    scheduler.join().unwrap();
    scheduler.shutdown();
    assert_eq!(
        completed.load(Ordering::Relaxed),
        PRODUCERS * JOBS_PER_PRODUCER
    );
}

#[test]
fn blocking_lane_preserves_priority_order() {
    let scheduler = ThreadScheduler::<8>::new_with_local_queue_initial_capacity(
        1,
        "blocking-lane-priority",
        DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
    )
    .unwrap();
    let blocking_started = Arc::new(Barrier::new(2));
    let blocking_release = Arc::new(Barrier::new(2));
    let (observed_sender, observed_receiver) = mpsc::channel();

    let started = Arc::clone(&blocking_started);
    let release = Arc::clone(&blocking_release);
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
            started.wait();
            release.wait();
        })
        .unwrap();
    blocking_started.wait();

    let low_sender = observed_sender.clone();
    scheduler
        .schedule::<BlockingTask, _>(Priority::Low, None, move |_| {
            low_sender.send(1usize).unwrap();
        })
        .unwrap();
    scheduler
        .schedule::<BlockingTask, _>(Priority::Critical, None, move |_| {
            observed_sender.send(2usize).unwrap();
        })
        .unwrap();

    blocking_release.wait();
    scheduler.join().unwrap();
    assert_eq!(
        [
            observed_receiver.recv().unwrap(),
            observed_receiver.recv().unwrap()
        ],
        [2, 1]
    );
    scheduler.shutdown();
}

#[test]
fn blocking_lane_rejects_admission_after_shutdown() {
    let scheduler = ThreadScheduler::new(1, "blocking-lane-shutdown").unwrap();
    scheduler.shutdown();

    let result = scheduler.schedule::<BlockingTask, _>(Priority::Normal, None, |_| {});
    assert_eq!(result, Err(ExecutorError::ShuttingDown));
}

#[test]
fn large_pool_wakes_high_index_workers_across_idle_cycles() {
    // Regression for the single-AtomicU64 idle map: workers with id >= 64 were
    // never registered in the wake bitmap, so on a pool larger than 64 they
    // could not be targeted by the wake lottery. With a multi-word bitset every
    // worker is addressable. Drive several submit -> quiesce -> submit cycles so
    // the whole pool parks between rounds and must be re-woken each round; a
    // lost/unreachable wakeup would either drop a task (count mismatch) or hang
    // into the nextest timeout.
    const WORKERS: usize = 100;
    const ROUNDS: usize = 4;
    const TASKS_PER_ROUND: usize = 400;

    let scheduler = ThreadScheduler::new(WORKERS, "test-large-pool").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));

    for _ in 0..ROUNDS {
        let (sender, receiver) = mpsc::channel();
        for _ in 0..TASKS_PER_ROUND {
            let completed = Arc::clone(&completed);
            let sender = sender.clone();
            scheduler
                .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                    completed.fetch_add(1, Ordering::AcqRel);
                    sender.send(()).unwrap();
                })
                .unwrap();
        }
        drop(sender);
        // Barrier: every task of this round must complete before the next round,
        // forcing the pool to fully quiesce (all workers park) in between.
        for _ in 0..TASKS_PER_ROUND {
            receiver.recv().unwrap();
        }
    }

    scheduler.shutdown();
    assert_eq!(completed.load(Ordering::Acquire), ROUNDS * TASKS_PER_ROUND);
}

#[test]
fn quiescent_single_task_selection_reuses_work_class_worker() {
    let scheduler = ThreadScheduler::new(4, "test-quiescent-route").unwrap();
    let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
    let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

    scheduler.shutdown();

    assert_eq!(first, second);
    assert_eq!(first, 3);
}

#[test]
fn serial_handoff_selection_reuses_work_class_worker() {
    let scheduler = ThreadScheduler::new(4, "test-serial-handoff-route").unwrap();
    scheduler.inner.active_workers.store(1, Ordering::Release);

    let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
    let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

    scheduler.inner.active_workers.store(0, Ordering::Release);
    scheduler.shutdown();

    assert_eq!(first, second);
    assert_eq!(first, 3);
}

#[test]
fn queued_parallel_selection_rotates_workers() {
    let scheduler = ThreadScheduler::new(4, "test-parallel-route").unwrap();
    scheduler.inner.pending_tasks.store(1, Ordering::Release);

    let first = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);
    let second = scheduler.select_worker::<BlockingTask>(Priority::Normal, None);

    scheduler.inner.pending_tasks.store(0, Ordering::Release);
    scheduler.shutdown();

    assert_ne!(first, second);
}

#[test]
fn scheduler_scope_runs_borrowing_jobs_before_return() {
    let scheduler = ThreadScheduler::new(2, "test-scope").unwrap();
    let sum = AtomicUsize::new(0);

    scheduler
        .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
            for value in 1..=16 {
                let sum = &sum;
                scope.spawn(move |_| {
                    sum.fetch_add(value, Ordering::Relaxed);
                })?;
            }
            Ok(())
        })
        .unwrap();

    scheduler.shutdown();
    assert_eq!(sum.load(Ordering::Relaxed), 136);
}

#[test]
fn scheduler_scope_nested_saturation_completes() {
    // Regression guard for ISSUE-208: a scoped job that itself opens a nested
    // scope must complete. Before help-while-waiting, `scope().wait()` parked the
    // caller without running scheduler work, so nested fork-join deadlocked
    // (provably with one worker) and corrupted the heap under concurrent nesting
    // (STATUS_HEAP_CORRUPTION). Value-semantic: every inner increment must land.
    for &workers in &[1usize, 2, 4] {
        let scheduler = ThreadScheduler::new(workers, "test-nested-scope").unwrap();
        let outer = 32usize;
        let inner = 16usize;
        let counter = AtomicUsize::new(0);

        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |outer_scope| {
                for _ in 0..outer {
                    let scheduler = &scheduler;
                    let counter = &counter;
                    outer_scope.spawn(move |_| {
                        scheduler
                            .scope::<SyncTask, _>(Priority::Normal, None, |inner_scope| {
                                for _ in 0..inner {
                                    let counter = &counter;
                                    inner_scope.spawn(move |_| {
                                        counter.fetch_add(1, Ordering::Relaxed);
                                    })?;
                                }
                                Ok(())
                            })
                            .expect("nested scope must complete");
                    })?;
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(
            counter.load(Ordering::Relaxed),
            outer * inner,
            "nested saturation lost increments at {workers} worker(s)"
        );
        scheduler.shutdown();
    }
}

fn recursive_scope_sum(scheduler: &ThreadScheduler, lo: u64, hi: u64) -> u64 {
    if hi.saturating_sub(lo) <= 1024 {
        return (lo..hi).sum();
    }
    let mid = lo + (hi - lo) / 2;
    let mut left = 0u64;
    let mut right = 0u64;
    {
        let left = &mut left;
        let right = &mut right;
        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
                scope.spawn(|_| {
                    *left = recursive_scope_sum(scheduler, lo, mid);
                })?;
                scope.spawn(|_| {
                    *right = recursive_scope_sum(scheduler, mid, hi);
                })?;
                Ok(())
            })
            .expect("recursive scope must complete");
    }
    left + right
}

#[test]
fn scheduler_scope_recursive_fork_join_is_sound() {
    // ISSUE-208 corruption guard: the recursive two-branch fork-join is the exact
    // shape of `moirai_iter` `drive` (log2-depth nested scopes, each branch stolen
    // by a peer worker that dereferences the parent scope's stack-owned state).
    // Before help-while-waiting this deadlocked (one worker) and corrupted the
    // heap (STATUS_HEAP_CORRUPTION) under concurrent nesting. Analytical oracle:
    // the arithmetic series sum, asserted value-semantically.
    const N: u64 = 200_000;
    let expected = N * (N - 1) / 2;
    for &workers in &[1usize, 2, 4] {
        let scheduler = ThreadScheduler::new(workers, "test-recursive-scope").unwrap();
        assert_eq!(
            recursive_scope_sum(&scheduler, 0, N),
            expected,
            "recursive fork-join sum diverged at {workers} worker(s)"
        );
        scheduler.shutdown();
    }
}

#[test]
fn scheduler_scope_nested_leaves_scheduler_quiescent() {
    // Accounting guard for help-while-waiting (ADR-019): a worker waiter runs
    // jobs via a re-entrant `execute_job`, which mutates the global
    // `pending_tasks`/`active_workers` counters. A leaked increment would leave
    // `join()` unable to observe quiescence (hang → nextest terminates). Assert
    // the scheduler returns to a consistent quiescent state with no spurious
    // failures after a nested workload.
    for &workers in &[1usize, 2, 4] {
        let scheduler = ThreadScheduler::new(workers, "test-nested-quiescent").unwrap();
        let outer = 16usize;
        let inner = 8usize;
        let counter = AtomicUsize::new(0);

        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |outer_scope| {
                for _ in 0..outer {
                    let scheduler = &scheduler;
                    let counter = &counter;
                    outer_scope.spawn(move |_| {
                        scheduler
                            .scope::<SyncTask, _>(Priority::Normal, None, |inner_scope| {
                                for _ in 0..inner {
                                    let counter = &counter;
                                    inner_scope.spawn(move |_| {
                                        counter.fetch_add(1, Ordering::Relaxed);
                                    })?;
                                }
                                Ok(())
                            })
                            .expect("nested scope must complete");
                    })?;
                }
                Ok(())
            })
            .unwrap();

        // Terminates only if the help path leaked no pending/active count.
        scheduler.join().expect("scheduler must reach quiescence");
        let metrics = scheduler.metrics();

        assert_eq!(counter.load(Ordering::Relaxed), outer * inner);
        assert_eq!(
            metrics.pending_tasks, 0,
            "leaked pending count at {workers} worker(s)"
        );
        assert_eq!(
            metrics.active_workers, 0,
            "leaked active-worker count at {workers} worker(s)"
        );
        assert_eq!(
            metrics.failed_tasks, 0,
            "spurious job failure at {workers} worker(s)"
        );
        scheduler.shutdown();
    }
}

#[test]
fn scheduler_scope_nested_panic_propagates_and_pool_survives() {
    // Adversarial guard for the help-while-waiting scope (ADR-019): when a nested
    // scoped job panics, the nested scope must report SpawnFailed(Panicked), its
    // sibling job must still run, and the outer scope must complete without
    // deadlock or corruption — i.e. a panic on a help-stealing worker unwinds
    // only its own job, never the waiter's help loop.
    for &workers in &[1usize, 2, 4] {
        let scheduler = ThreadScheduler::new(workers, "test-nested-panic").unwrap();
        let outer = 8usize;
        let sibling_ran = AtomicUsize::new(0);
        let nested_panics_reported = AtomicUsize::new(0);

        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |outer_scope| {
                for _ in 0..outer {
                    let scheduler = &scheduler;
                    let sibling_ran = &sibling_ran;
                    let nested_panics_reported = &nested_panics_reported;
                    outer_scope.spawn(move |_| {
                        let result =
                            scheduler.scope::<SyncTask, _>(Priority::Normal, None, |inner| {
                                inner.spawn(|_| panic!("nested scoped job panic"))?;
                                inner.spawn(move |_| {
                                    sibling_ran.fetch_add(1, Ordering::Relaxed);
                                })?;
                                Ok(())
                            });
                        if matches!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked))) {
                            nested_panics_reported.fetch_add(1, Ordering::Relaxed);
                        }
                    })?;
                }
                Ok(())
            })
            .unwrap();

        assert_eq!(
            nested_panics_reported.load(Ordering::Relaxed),
            outer,
            "each nested scope must report its panic at {workers} worker(s)"
        );
        assert_eq!(
            sibling_ran.load(Ordering::Relaxed),
            outer,
            "sibling of a panicking nested job must still run at {workers} worker(s)"
        );
        scheduler.shutdown();
    }
}

#[test]
fn scheduler_scope_reports_panicked_job() {
    let scheduler = ThreadScheduler::new(1, "test-scope-panic").unwrap();
    let completed = AtomicUsize::new(0);

    let result = scheduler.scope::<SyncTask, _>(Priority::Normal, None, |scope| {
        scope.spawn(|_| panic!("scoped job panic"))?;
        let completed = &completed;
        scope.spawn(move |_| {
            completed.fetch_add(1, Ordering::Relaxed);
        })?;
        Ok(())
    });

    scheduler.shutdown();
    assert_eq!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));
    assert_eq!(completed.load(Ordering::Relaxed), 1);
}

#[test]
fn scheduler_join_waits_for_queued_and_active_work() {
    let scheduler = ThreadScheduler::new(2, "test-join").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    // Every job parks on this gate, which the test holds until it has observed
    // the outstanding work. Without it the precondition is a race the workers
    // usually win: two threads drain eight atomic increments long before the
    // scheduling loop returns, so `has_work` reports quiescence and the
    // assertion fails intermittently rather than describing the scheduler.
    // Holding the gate pins the state the test name claims — the pool's two
    // workers active inside a job, the remaining six queued behind them.
    let gate = Arc::new(std::sync::Mutex::new(()));
    let held = gate.lock().expect("fresh test gate is never poisoned");

    for _ in 0..8 {
        let completed = Arc::clone(&completed);
        let gate = Arc::clone(&gate);
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                drop(gate.lock().expect("test gate is never poisoned"));
                completed.fetch_add(1, Ordering::AcqRel);
            })
            .unwrap();
    }

    assert!(scheduler.has_work());
    // Release the gate before joining: `join` blocks this thread until the
    // work drains, so the jobs must be free to finish first.
    drop(held);
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(completed.load(Ordering::Acquire), 8);
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.active_workers, 0);
    assert_eq!(metrics.completed_tasks, 8);
    assert!(!scheduler.has_work());
}

#[test]
fn scheduler_join_waits_for_work_submitted_while_active() {
    let scheduler = ThreadScheduler::new(2, "test-join-transitive").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let (started_sender, started_receiver) = mpsc::channel();
    let (scheduled_sender, scheduled_receiver) = mpsc::channel();

    {
        let completed = Arc::clone(&completed);
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                started_sender.send(()).unwrap();
                scheduled_receiver.recv().unwrap();
                completed.fetch_add(1, Ordering::AcqRel);
            })
            .unwrap();
    }

    started_receiver.recv().unwrap();
    std::thread::scope(|scope| {
        let completed = Arc::clone(&completed);
        let scheduler_ref = &scheduler;
        scope.spawn(move || {
            scheduler_ref
                .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                    completed.fetch_add(1, Ordering::AcqRel);
                })
                .unwrap();
            scheduled_sender.send(()).unwrap();
        });

        scheduler.join().unwrap();
    });

    let metrics = scheduler.metrics();
    scheduler.shutdown();

    assert_eq!(completed.load(Ordering::Acquire), 2);
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.active_workers, 0);
    assert_eq!(metrics.completed_tasks, 2);
}

#[test]
fn indexed_fan_out_runs_all_items() {
    let scheduler = ThreadScheduler::new(2, "test-indexed").unwrap();
    let sum = AtomicUsize::new(0);

    scheduler
        .for_each_indexed::<BlockingTask, _>(Priority::Normal, None, 32, |index| {
            sum.fetch_add(index + 1, Ordering::Relaxed);
        })
        .unwrap();

    scheduler.shutdown();
    assert_eq!(sum.load(Ordering::Relaxed), 528);
}

#[test]
fn indexed_map_reduce_returns_reduced_value() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce").unwrap();

    let sum = scheduler
        .map_reduce_indexed::<BlockingTask, _, _, _>(
            Priority::Normal,
            None,
            32,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();

    scheduler.shutdown();
    assert_eq!(sum, 528);
}

#[test]
fn nested_indexed_saturation_completes() {
    // The outer jobs occupy every worker before entering indexed fan-out. A
    // parking indexed waiter therefore deadlocks with its inner chunks queued
    // and no runnable worker. Nested indexed regions flatten onto their current
    // worker lane, retaining outer parallelism without recursive job stealing.
    const WORKERS: usize = 2;
    const INNER_ITEMS: usize = 1024;
    let scheduler = ThreadScheduler::new(WORKERS, "test-nested-indexed").unwrap();
    let barrier = Barrier::new(WORKERS);
    let outer_lanes = AtomicUsize::new(0);
    let sum = AtomicUsize::new(0);
    let reduced_sum = AtomicUsize::new(0);

    scheduler
        .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
            for outer_index in 0..WORKERS {
                let scheduler = &scheduler;
                let barrier = &barrier;
                let outer_lanes = &outer_lanes;
                let sum = &sum;
                let reduced_sum = &reduced_sum;
                scope.spawn(move |_| {
                    let outer_worker = get_current_worker_id()
                        .expect("scoped outer task must execute on a scheduler worker");
                    outer_lanes.fetch_or(1usize << outer_worker, Ordering::Relaxed);
                    barrier.wait();
                    scheduler
                        .for_each_indexed::<SyncTask, _>(
                            Priority::Normal,
                            None,
                            INNER_ITEMS,
                            |inner_index| {
                                assert_eq!(get_current_worker_id(), Some(outer_worker));
                                sum.fetch_add(
                                    outer_index * INNER_ITEMS + inner_index + 1,
                                    Ordering::Relaxed,
                                );
                            },
                        )
                        .expect("nested indexed fan-out must complete");

                    barrier.wait();
                    let local_sum = scheduler
                        .map_reduce_indexed::<SyncTask, _, _, _>(
                            Priority::Normal,
                            None,
                            INNER_ITEMS,
                            0usize,
                            |inner_index| {
                                assert_eq!(get_current_worker_id(), Some(outer_worker));
                                outer_index * INNER_ITEMS + inner_index + 1
                            },
                            usize::wrapping_add,
                        )
                        .expect("nested indexed map/reduce must complete");
                    reduced_sum.fetch_add(local_sum, Ordering::Relaxed);
                })?;
            }
            Ok(())
        })
        .unwrap();

    let item_count = WORKERS * INNER_ITEMS;
    let expected = item_count * (item_count + 1) / 2;
    assert_eq!(outer_lanes.load(Ordering::Relaxed), (1usize << WORKERS) - 1);
    assert_eq!(sum.load(Ordering::Relaxed), expected);
    assert_eq!(reduced_sum.load(Ordering::Relaxed), expected);
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.active_workers, 0);
    scheduler.shutdown();
}

#[test]
fn indexed_caller_flattens_nested_regions_onto_its_lane() {
    const WORKERS: usize = 2;
    const OUTER_ITEMS: usize = WORKERS + 1;
    const INNER_ITEMS: usize = 32;
    let scheduler = ThreadScheduler::new(WORKERS, "test-indexed-caller-nesting").unwrap();
    let visited = AtomicUsize::new(0);
    let reduced = AtomicUsize::new(0);

    scheduler
        .for_each_indexed::<SyncTask, _>(Priority::Normal, None, OUTER_ITEMS, |outer_index| {
            let outer_lane = get_current_worker_id();
            scheduler
                .for_each_indexed::<SyncTask, _>(
                    Priority::Normal,
                    None,
                    INNER_ITEMS,
                    |inner_index| {
                        assert_eq!(get_current_worker_id(), outer_lane);
                        visited.fetch_add(
                            outer_index * INNER_ITEMS + inner_index + 1,
                            Ordering::Relaxed,
                        );
                    },
                )
                .expect("nested indexed fan-out must remain on its outer lane");

            let local_sum = scheduler
                .map_reduce_indexed::<SyncTask, _, _, _>(
                    Priority::Normal,
                    None,
                    INNER_ITEMS,
                    0usize,
                    |inner_index| {
                        assert_eq!(get_current_worker_id(), outer_lane);
                        outer_index * INNER_ITEMS + inner_index + 1
                    },
                    usize::wrapping_add,
                )
                .expect("nested indexed reduction must remain on its outer lane");
            reduced.fetch_add(local_sum, Ordering::Relaxed);
        })
        .unwrap();

    let item_count = OUTER_ITEMS * INNER_ITEMS;
    let expected = item_count * (item_count + 1) / 2;
    assert_eq!(visited.load(Ordering::Relaxed), expected);
    assert_eq!(reduced.load(Ordering::Relaxed), expected);
    scheduler.join().unwrap();
    scheduler.shutdown();
}

#[test]
fn indexed_map_reduce_small_count_schedules_worker_lanes() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-small").unwrap();

    let sum = scheduler
        .map_reduce_indexed::<BlockingTask, _, _, _>(
            Priority::Normal,
            None,
            32,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(sum, 528);
    assert_eq!(metrics.completed_tasks, 2);
}

#[test]
fn indexed_map_reduce_reports_panicked_mapper() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-panic").unwrap();

    let result = scheduler.map_reduce_indexed::<BlockingTask, _, _, _>(
        Priority::Normal,
        None,
        4,
        0usize,
        |index| {
            if index == 2 {
                panic!("map panic");
            }
            index + 1
        },
        usize::wrapping_add,
    );
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(result, Err(ExecutorError::SpawnFailed(TaskError::Panicked)));
    // Three lanes cover four items: the caller lane plus two scheduled worker
    // chunks. One worker chunk completes and the mapper-panic chunk reaches the
    // scheduler boundary as a failed job.
    assert_eq!(metrics.completed_tasks, 1);
    assert_eq!(metrics.failed_tasks, 1);
}

#[test]
fn indexed_map_reduce_drains_queued_work_after_identity_clone_panics() {
    struct PanicOnSecondClone {
        value: usize,
        clone_attempts: Arc<AtomicUsize>,
    }

    impl Clone for PanicOnSecondClone {
        fn clone(&self) -> Self {
            let attempt = self.clone_attempts.fetch_add(1, Ordering::AcqRel);
            assert_ne!(attempt, 1, "second identity clone panic");
            Self {
                value: self.value,
                clone_attempts: Arc::clone(&self.clone_attempts),
            }
        }
    }

    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-clone-panic").unwrap();
    let clone_attempts = Arc::new(AtomicUsize::new(0));
    let mapped = AtomicUsize::new(0);
    let identity = PanicOnSecondClone {
        value: 0,
        clone_attempts: Arc::clone(&clone_attempts),
    };

    let result = scheduler.map_reduce_indexed::<SyncTask, _, _, _>(
        Priority::Normal,
        None,
        6,
        identity,
        |index| {
            while clone_attempts.load(Ordering::Acquire) < 2 {
                core::hint::spin_loop();
            }
            mapped.fetch_add(1, Ordering::Relaxed);
            PanicOnSecondClone {
                value: index + 1,
                clone_attempts: Arc::clone(&clone_attempts),
            }
        },
        |left, right| PanicOnSecondClone {
            value: left.value + right.value,
            clone_attempts: left.clone_attempts,
        },
    );

    scheduler.shutdown();
    assert!(matches!(
        result,
        Err(ExecutorError::SpawnFailed(TaskError::Panicked))
    ));
    assert_eq!(mapped.load(Ordering::Relaxed), 2);
}

#[test]
fn indexed_map_reduce_caps_chunks_at_worker_plus_caller_lanes() {
    let scheduler = ThreadScheduler::new(2, "test-indexed-reduce-parallel").unwrap();

    let sum = scheduler
        .map_reduce_indexed::<BlockingTask, _, _, _>(
            Priority::Normal,
            None,
            64,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();

    scheduler.shutdown();
    assert_eq!(sum, 2080);
    assert_eq!(metrics.completed_tasks, 2);
}

#[test]
fn indexed_operations_use_every_available_lane_above_cap() {
    const COUNT: usize = 10;
    const WORKERS: usize = 8;
    let scheduler = ThreadScheduler::new(WORKERS, "test-indexed-all-lanes").unwrap();
    let visits: [AtomicUsize; COUNT] = std::array::from_fn(|_| AtomicUsize::new(0));

    scheduler
        .for_each_indexed::<SyncTask, _>(Priority::Normal, None, COUNT, |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let sum = scheduler
        .map_reduce_indexed::<SyncTask, _, _, _>(
            Priority::Normal,
            None,
            COUNT,
            0usize,
            |index| index + 1,
            usize::wrapping_add,
        )
        .unwrap();
    scheduler.join().unwrap();
    let metrics = scheduler.metrics();
    scheduler.shutdown();

    assert_eq!(
        visits.map(|count| count.load(Ordering::Relaxed)),
        [1; COUNT]
    );
    assert_eq!(sum, COUNT * (COUNT + 1) / 2);
    assert_eq!(
        metrics.completed_tasks,
        2 * u64::try_from(WORKERS).expect("worker count must fit scheduler metrics")
    );
}

#[test]
fn scheduler_scope_completes_registered_jobs_before_body_error_returns() {
    let scheduler = ThreadScheduler::new(2, "test-scope-body-error").unwrap();
    let completed = AtomicUsize::new(0);

    let result = scheduler.scope::<SyncTask, _>(Priority::Normal, None, |scope| {
        for _ in 0..8 {
            let completed = &completed;
            scope.spawn(move |_| {
                completed.fetch_add(1, Ordering::Relaxed);
            })?;
        }

        Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation))
    });

    scheduler.shutdown();
    assert_eq!(
        result,
        Err(ExecutorError::SpawnFailed(TaskError::InvalidOperation))
    );
    assert_eq!(completed.load(Ordering::Relaxed), 8);
}

#[test]
fn scheduler_scope_completes_registered_jobs_before_resuming_body_panic() {
    let scheduler = ThreadScheduler::new(2, "test-scope-body-panic").unwrap();
    let completed = AtomicUsize::new(0);

    let result = catch_unwind(AssertUnwindSafe(|| {
        scheduler
            .scope::<SyncTask, _>(Priority::Normal, None, |scope| {
                for _ in 0..8 {
                    let completed = &completed;
                    scope.spawn(move |_| {
                        completed.fetch_add(1, Ordering::Relaxed);
                    })?;
                }

                panic!("scope body panic");
            })
            .unwrap();
    }));

    scheduler.shutdown();
    assert!(result.is_err());
    assert_eq!(completed.load(Ordering::Relaxed), 8);
}

#[test]
fn test_melinoe_partition_routing() {
    use melinoe::sync::partition_map;
    use melinoe::{brand_scope, MelinoeCell};

    let _exec = crate::global();

    brand_scope(|token| {
        let mut cells: Vec<MelinoeCell<'_, usize>> = (0..32).map(|_| MelinoeCell::new(0)).collect();

        let results = partition_map(&mut cells, 4, |start, mut shard| {
            for (i, cell) in shard.iter_mut().enumerate() {
                *cell = start + i;
            }
            shard.len()
        });

        assert_eq!(results.len(), 4);
        assert_eq!(results.iter().sum::<usize>(), 32);

        let snap = token.share();
        for (i, cell) in cells.iter().enumerate() {
            assert_eq!(*cell.borrow(snap), i);
        }
    });
}
