//! Indexed fan-out and map-reduce lane selection and failure paths.

use super::*;

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
