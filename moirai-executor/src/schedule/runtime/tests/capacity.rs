//! Queue capacity configuration: partitioning, normalization, and rejection.

use super::*;

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
        .map(|worker| worker.queues.local_queue_capacities()[Priority::default().index()])
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
        .map(|worker| worker.queues.local_queue_capacities()[Priority::default().index()])
        .collect::<Vec<_>>();

    assert_eq!(capacities, vec![128; 3]);
    scheduler.shutdown();
}

/// Run `burst` jobs that all land while the single worker is blocked, then
/// report the slot count its default plane holds while the worker is still
/// inside the last job, before it can go idle and release the plane.
fn local_plane_slots_after_burst(burst: usize, start: usize) -> usize {
    let scheduler = scheduler_with_queue_config::<256>(
        1,
        "burst-drain",
        ExecutorConfig::default().max_global_queue_size,
        start,
    )
    .unwrap();

    let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
            started_tx.send(()).expect("harness receiver lives");
            release_rx.recv().expect("harness sender lives");
        })
        .unwrap();
    // The only worker is now inside the blocking job, so the burst below
    // accumulates in the injector instead of being consumed as it arrives.
    started_rx.recv().expect("the blocking job must start");

    // The job that completes the burst holds the worker inside the job
    // until the harness has read the plane, so an idle-time shrink
    // cannot race the observation.
    let (last_tx, last_rx) = std::sync::mpsc::channel::<()>();
    let (hold_tx, hold_rx) = std::sync::mpsc::channel::<()>();
    let hold_rx = Arc::new(Mutex::new(hold_rx));
    let remaining = Arc::new(AtomicUsize::new(burst));
    for _ in 0..burst {
        let last_tx = last_tx.clone();
        let hold_rx = Arc::clone(&hold_rx);
        let remaining = Arc::clone(&remaining);
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                    last_tx.send(()).expect("harness receiver lives");
                    hold_rx
                        .lock()
                        .expect("hold receiver is never poisoned")
                        .recv()
                        .expect("harness sender lives");
                }
            })
            .unwrap();
    }
    drop(last_tx);
    release_tx
        .send(())
        .expect("the blocking job must still wait");
    last_rx.recv().expect("the burst must finish");

    let slots =
        scheduler.inner.workers[0].queues.local_queue_capacities()[Priority::default().index()];
    hold_tx.send(()).expect("the last job must still wait");
    scheduler.shutdown();
    slots
}

#[test]
fn local_plane_holds_the_drained_burst_until_the_worker_idles() {
    // ADR-038, MOI-QUEUE-PLANE-SHRINK: `next_job` drains the injector to
    // exhaustion, and a plane only grows while it is in use, so at the moment
    // the worker finishes a burst the plane holds slots covering the burst,
    // independent of the configured initial capacity. Draining to exhaustion
    // is load-bearing: it is what lets a high-priority job preempt work
    // already queued behind it, since the injector is one cross-priority queue
    // and only the local planes are priority-ordered. The worker releases the
    // oversized plane once it idles (`WorkerQueueOwner::shrink_drained_planes`,
    // covered in `schedule::queue::tests`), not by bounding the pass.
    const START: usize = 16;
    let small = local_plane_slots_after_burst(200, START);
    let large = local_plane_slots_after_burst(2_000, START);

    assert!(
        small >= 200 && large >= 2_000,
        "a drained burst is held in one plane, so its slots cover the burst;          got {small} slots for 200 jobs and {large} for 2,000"
    );
    assert!(
        large > small,
        "held slots track burst size rather than the {START}-slot start;          got {small} for 200 jobs and {large} for 2,000"
    );
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

    assert!(
        scheduler
            .inner
            .workers
            .iter()
            .all(|worker| worker.queues.injector_capacity() == 2)
    );
    scheduler.shutdown();
}
