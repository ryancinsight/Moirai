//! Admission under saturation and the bounded blocking lane.

use super::*;

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
