//! Worker placement: NUMA assignment and victim choice, wakeups, and lane reuse.

use super::*;

#[test]
fn scheduler_numa_policy_controls_worker_assignments() {
    let scheduler = scheduler_with_queue_config::<256>(
        2,
        "numa-disabled",
        ExecutorConfig::default().max_global_queue_size,
        DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
    )
    .unwrap();

    assert!(
        scheduler
            .inner
            .worker_numa_nodes
            .iter()
            .all(|node| node.is_none())
    );

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
fn test_melinoe_partition_routing() {
    use melinoe::sync::partition_map;
    use melinoe::{MelinoeCell, brand_scope};

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
