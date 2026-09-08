//! Shutdown ordering: handle drops, cross-class joins, and drain-to-quiescence.

use super::*;

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
