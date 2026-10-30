//! Shutdown ordering: handle drops, cross-class joins, and drain-to-quiescence.

use super::*;
use crate::schedule::{SchedulerControl, WorkClass};

fn assert_scheduler_membership<C: WorkClass>(thread_name: &str) {
    let scheduler = ThreadScheduler::new(1, thread_name).unwrap();
    let same_scheduler = scheduler.clone();
    let same_scheduler_clone = scheduler.clone();
    let other_thread_name = format!("{thread_name}-other");
    let other_scheduler = ThreadScheduler::new(1, &other_thread_name).unwrap();
    let other_scheduler_query = other_scheduler.clone();

    assert!(!SchedulerControl::is_current_worker(&scheduler));
    assert!(!SchedulerControl::is_current_worker(&same_scheduler));
    assert!(!SchedulerControl::is_current_worker(&other_scheduler));

    let (other_started_sender, other_started_receiver) = mpsc::sync_channel(0);
    let (other_release_sender, other_release_receiver) = mpsc::sync_channel(0);
    other_scheduler
        .schedule::<C, _>(Priority::Normal, Some(0), move |worker_id| {
            other_started_sender.send(worker_id).unwrap();
            other_release_receiver.recv().unwrap();
        })
        .unwrap();
    let other_worker_id = other_started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("the other scheduler worker must start before the deadline");

    let (observed_sender, observed_receiver) = mpsc::sync_channel(1);
    scheduler
        .schedule::<C, _>(Priority::Normal, Some(0), move |worker_id| {
            observed_sender
                .send((
                    worker_id,
                    SchedulerControl::is_current_worker(&same_scheduler),
                    SchedulerControl::is_current_worker(&same_scheduler_clone),
                    SchedulerControl::is_current_worker(&other_scheduler_query),
                ))
                .unwrap();
        })
        .unwrap();
    let (worker_id, same, clone, other) = observed_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("the membership query must complete before the deadline");

    assert_eq!(worker_id, other_worker_id);
    assert!(same);
    assert!(clone);
    assert!(!other);

    other_release_sender.send(()).unwrap();
    scheduler.shutdown();
    other_scheduler.shutdown();
}

#[test]
fn scheduler_membership_is_identity_specific() {
    assert_scheduler_membership::<SyncTask>("membership-compute");
    assert_scheduler_membership::<BlockingTask>("membership-blocking");
}

#[test]
fn final_external_handle_drop_requests_without_join_and_drains_accepted_jobs() {
    const COMPUTE_JOBS: usize = 32;
    const BLOCKING_VALUE: usize = 1_024;
    let scheduler = ThreadScheduler::new(1, "test-final-scheduler-drop").unwrap();
    let inner = Arc::downgrade(&scheduler.inner);
    let released = Arc::new((Mutex::new(false), Condvar::new()));
    scheduler.retain_lifetime_owner(DropSignal {
        state: Arc::clone(&released),
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

    let queue_inner = Arc::clone(&surviving_handle.inner);
    let (queue_locked_sender, queue_locked_receiver) = mpsc::sync_channel(0);
    let (queue_release_sender, queue_release_receiver) = mpsc::sync_channel(0);
    let queue_lock_thread = std::thread::spawn(move || {
        queue_inner
            .blocking_lane
            .get()
            .expect("blocking admission initializes the lane")
            .hold_queue_lock_until(queue_locked_sender, queue_release_receiver);
    });
    queue_locked_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("test must hold the blocking queue mutex before final drop");

    let (drop_sender, drop_receiver) = mpsc::sync_channel(1);
    let drop_thread = std::thread::spawn(move || {
        drop(surviving_handle);
        drop_sender.send(()).unwrap();
    });
    drop_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("final scheduler drop must return without joining workers");
    drop_thread.join().unwrap();
    let live_inner = inner
        .upgrade()
        .expect("accepted work retains scheduler state after request-only drop");
    assert!(
        live_inner.shutdown.load(Ordering::Acquire),
        "final scheduler drop must publish closure"
    );
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
    queue_release_sender
        .send(())
        .expect("test controller releases the blocking queue mutex");
    queue_lock_thread.join().unwrap();
    compute_release_sender.send(()).unwrap();
    blocking_release_sender.send(()).unwrap();

    let (lock, signal) = &*released;
    let dropped = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (dropped, timeout) = signal
        .wait_timeout_while(dropped, TEST_EVENT_DEADLINE, |dropped| !*dropped)
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(
        !timeout.timed_out() && *dropped,
        "accepted work must drain before scheduler state is released"
    );

    assert_eq!(
        completed.load(Ordering::Relaxed),
        COMPUTE_JOBS * (COMPUTE_JOBS + 1) / 2 + BLOCKING_VALUE
    );
    assert_eq!(
        capture_drops.load(Ordering::Relaxed),
        COMPUTE_JOBS + 1,
        "every admitted capture must be released exactly once"
    );
    assert!(
        inner.upgrade().is_none(),
        "drained workers must release scheduler state"
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
