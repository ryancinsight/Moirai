use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use moirai_core::{
    Priority,
    error::ExecutorError,
    executor::{ExecutorConfig, config::DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY},
};

use super::fixtures::{PendingTask, pending_async_state_with_first_poll_signal};
use crate::schedule::{SyncTask, ThreadScheduler};

/// Four scheduler phases remain below the 30-second slow-test threshold
/// even if each consumes its complete event deadline.
const TEST_EVENT_DEADLINE: Duration = Duration::from_secs(5);

struct GateRelease(Option<mpsc::Sender<()>>);

impl GateRelease {
    fn release(&mut self) {
        self.0
            .take()
            .expect("gate release is sent once")
            .send(())
            .expect("gated worker remains alive");
    }
}

impl Drop for GateRelease {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            // Receiver exit means the worker already left the gate, so no
            // cleanup action remains for this non-panicking test guard.
            match sender.send(()) {
                Ok(()) | Err(_) => {}
            }
        }
    }
}

/// M1 regression at the real scheduler: a worker's injector is provably
/// full (fill until `ResourceExhausted`) and the only worker is gated, so
/// a wake can never be admitted — the waking thread must poll inline and
/// the woken task must still complete. Before the fix, `Waker::wake`
/// discarded the rejection and the task stayed `QUEUED` forever.
#[test]
fn woken_task_completes_while_worker_injector_is_full() {
    let scheduler = ThreadScheduler::<8>::from_executor_config(&ExecutorConfig {
        worker_threads: 1,
        max_global_queue_size: 8,
        local_queue_initial_capacity: DEFAULT_LOCAL_QUEUE_INITIAL_CAPACITY,
        thread_name_prefix: "wake-full-injector".into(),
        ..ExecutorConfig::default()
    })
    .expect("scheduler");
    let (
        PendingTask {
            state,
            handle,
            waker,
            polls,
        },
        first_poll_receiver,
    ) = pending_async_state_with_first_poll_signal(scheduler.clone(), 1789);

    // First poll runs on the worker and publishes the waker.
    Arc::clone(&state).schedule().expect("first poll admits");
    first_poll_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("first poll must publish its waker before the event deadline");

    // Gate the only worker inside a job so nothing can drain the injector.
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let mut gate_release = GateRelease(Some(release_tx));
    scheduler
        .schedule::<SyncTask, _>(Priority::Normal, None, move |_worker| {
            entered_tx.send(()).expect("test observer alive");
            release_rx.recv().expect("release signal");
        })
        .expect("gate job admits");
    entered_rx
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("worker must enter the gate before the event deadline");
    let waker = waker
        .lock()
        .unwrap()
        .take()
        .expect("first poll published its waker");
    assert_eq!(polls.load(Ordering::SeqCst), 1);

    // Start and park the waking thread before saturating the scheduler. The
    // timed wake phase then measures the wake path, not OS thread creation
    // latency under a concurrently loaded workspace test run.
    let wake_phase = Arc::new(AtomicUsize::new(0));
    let wake_phase_thread = Arc::clone(&wake_phase);
    let (wake_ready_tx, wake_ready_rx) = mpsc::sync_channel(1);
    let (wake_start_tx, wake_start_rx) = mpsc::sync_channel(0);
    let (wake_done_tx, wake_done_rx) = mpsc::sync_channel(1);
    let waking = std::thread::spawn(move || {
        wake_phase_thread.store(1, Ordering::SeqCst);
        wake_ready_tx.send(()).expect("wake-ready observer alive");
        wake_start_rx.recv().expect("wake-start observer alive");
        wake_phase_thread.store(2, Ordering::SeqCst);
        waker.wake();
        wake_phase_thread.store(3, Ordering::SeqCst);
        wake_done_tx.send(()).expect("wake observer alive");
    });
    wake_ready_rx
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("waking thread must park before the event deadline");

    // Fill the gated worker's injector until admission genuinely rejects.
    let filler_runs = Arc::new(AtomicUsize::new(0));
    let mut saw_rejection = false;
    for _ in 0..4096 {
        let filler_runs = Arc::clone(&filler_runs);
        let admitted = scheduler.schedule::<SyncTask, _>(Priority::Normal, None, move |_w| {
            filler_runs.fetch_add(1, Ordering::SeqCst);
        });
        if let Err(rejection) = admitted {
            assert!(matches!(rejection, ExecutorError::ResourceExhausted(_)));
            saw_rejection = true;
            break;
        }
    }
    assert!(
        saw_rejection,
        "an 8-slot injector must fill within 4096 pushes"
    );

    // The injector is full and its only drain is gated: every enqueue
    // retry rejects, so the wake must complete the future inline. The
    // handle resolving *before* the gate opens proves no worker polled.
    wake_start_tx
        .send(())
        .expect("parked waking thread remains alive");
    wake_done_rx
        .recv_timeout(TEST_EVENT_DEADLINE)
        .unwrap_or_else(|error| {
            panic!(
                "saturated wake must complete before the event deadline: {error}; phase={}, state={}, polls={}, finished={}, pending={}",
                wake_phase.load(Ordering::SeqCst),
                state.state.load(Ordering::SeqCst),
                polls.load(Ordering::SeqCst),
                handle.is_finished(),
                scheduler.pending_tasks()
            )
        });
    waking.join().expect("waking thread");
    let poll_count_before_release = polls.load(Ordering::SeqCst);
    let finished_before_release = handle.is_finished();

    gate_release.release();
    scheduler.shutdown();

    assert_eq!(poll_count_before_release, 2);
    assert!(
        finished_before_release,
        "the wake must poll inline before the gated worker is released"
    );
    assert_eq!(handle.join(), Some(Ok(1789)));
}
