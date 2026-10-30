//! Idle-hook failure behavior during scheduler shutdown.

use std::{
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, ThreadId},
};

use super::*;
use crate::schedule::register_idle_hook;

static HOOK_PAYLOAD: Mutex<Option<HookPayload>> = Mutex::new(None);
static HOOK_REGISTERED: OnceLock<()> = OnceLock::new();
static REMAINING_HOOK_REGISTERED: OnceLock<()> = OnceLock::new();
static REMAINING_HOOK_TARGET: Mutex<Option<(ThreadId, Arc<AtomicUsize>)>> = Mutex::new(None);

struct HookPayload {
    target: ThreadId,
    payload: Box<IdleHookFailure>,
    fired: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
}

#[derive(Debug)]
struct IdleHookFailure {
    token: usize,
    observed_token: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Drop for IdleHookFailure {
    fn drop(&mut self) {
        self.observed_token.store(self.token, Ordering::Release);
        self.drops.fetch_add(1, Ordering::Release);
    }
}

fn worker_panic_hook() {
    let payload = {
        let mut slot = HOOK_PAYLOAD
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot
            .as_ref()
            .is_some_and(|payload| payload.target == thread::current().id())
        {
            slot.take()
        } else {
            None
        }
    };
    if let Some(payload) = payload {
        let HookPayload {
            payload,
            fired,
            release,
            ..
        } = payload;
        fired
            .send(())
            .expect("test controller observes the idle hook");
        release
            .recv()
            .expect("test controller releases the idle-hook failure");
        std::panic::resume_unwind(payload);
    }
}

fn install_worker_panic(
    target: ThreadId,
    payload: Box<IdleHookFailure>,
    fired: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
) {
    HOOK_REGISTERED.get_or_init(|| {
        register_idle_hook(worker_panic_hook)
            .expect("worker-failure test reserves one idle-hook slot");
    });
    let replaced = HOOK_PAYLOAD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .replace(HookPayload {
            target,
            payload,
            fired,
            release,
        });
    assert!(replaced.is_none(), "worker panic payload must be unique");
}

fn remaining_idle_hook() {
    let target = REMAINING_HOOK_TARGET
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((thread, calls)) = target.as_ref()
        && *thread == std::thread::current().id()
    {
        calls.fetch_add(1, Ordering::Release);
    }
}

fn install_remaining_hook(target: ThreadId, calls: Arc<AtomicUsize>) {
    REMAINING_HOOK_REGISTERED.get_or_init(|| {
        register_idle_hook(remaining_idle_hook)
            .expect("worker-failure test reserves a trailing idle-hook slot");
    });
    let replaced = REMAINING_HOOK_TARGET
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .replace((target, calls));
    assert!(replaced.is_none(), "remaining-hook target must be unique");
}

#[test]
fn idle_hook_failure_closes_admission_and_drains_accepted_work() {
    let _serial = TEST_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let scheduler = ThreadScheduler::new(1, "idle-hook-failure-drain").unwrap();
    let shutdown_started = Arc::new(Barrier::new(2));
    let (worker, release_worker) = occupy_worker(&scheduler);
    let payload_token = 0x5A17usize;
    let payload_token_observed = Arc::new(AtomicUsize::new(0));
    let payload_drops = Arc::new(AtomicUsize::new(0));
    let remaining_calls = Arc::new(AtomicUsize::new(0));
    let (hook_fired_sender, hook_fired_receiver) = mpsc::sync_channel(0);
    let (hook_release_sender, hook_release_receiver) = mpsc::sync_channel(0);
    install_worker_panic(
        worker,
        Box::new(IdleHookFailure {
            token: payload_token,
            observed_token: Arc::clone(&payload_token_observed),
            drops: Arc::clone(&payload_drops),
        }),
        hook_fired_sender,
        hook_release_receiver,
    );
    install_remaining_hook(worker, Arc::clone(&remaining_calls));
    release_worker
        .send(())
        .expect("occupied worker remains connected");
    hook_fired_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("idle hook fires before the test deadline");

    let capture_drops = Arc::new(AtomicUsize::new(0));
    let (compute_sender, compute_receiver) = mpsc::sync_channel(2);
    for value in [11usize, 12] {
        let sender = compute_sender.clone();
        let capture = DropProbe {
            drops: Arc::clone(&capture_drops),
        };
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, Some(0), move |worker_id| {
                drop(capture);
                sender
                    .send((value, worker_id))
                    .expect("test observer remains connected");
            })
            .expect("compute work is admitted before hook failure");
    }

    let (blocking_sender, blocking_receiver) = mpsc::sync_channel(2);
    let (blocking_started_sender, blocking_started_receiver) = mpsc::sync_channel(0);
    let (blocking_release_sender, blocking_release_receiver) = mpsc::sync_channel(0);
    let blocking_capture = DropProbe {
        drops: Arc::clone(&capture_drops),
    };
    let first_blocking_sender = blocking_sender.clone();
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, Some(0), move |worker_id| {
            blocking_started_sender
                .send(())
                .expect("test observer remains connected");
            blocking_release_receiver
                .recv()
                .expect("test controller releases accepted blocking work");
            drop(blocking_capture);
            first_blocking_sender
                .send((21, worker_id))
                .expect("test observer remains connected");
        })
        .expect("first blocking job is admitted before hook failure");
    blocking_started_receiver
        .recv_timeout(TEST_EVENT_DEADLINE)
        .expect("first blocking job starts before the test deadline");
    let trailing_capture = DropProbe {
        drops: Arc::clone(&capture_drops),
    };
    scheduler
        .schedule::<BlockingTask, _>(Priority::Normal, Some(0), move |worker_id| {
            drop(trailing_capture);
            blocking_sender
                .send((22, worker_id))
                .expect("test observer remains connected");
        })
        .expect("trailing blocking job is admitted before hook failure");

    scheduler
        .inner
        .shutdown_started_barrier
        .set(Arc::clone(&shutdown_started))
        .expect("test installs one shutdown-publication rendezvous");
    hook_release_sender
        .send(())
        .expect("idle-hook failure remains connected");
    shutdown_started.wait();
    assert_eq!(payload_drops.load(Ordering::Acquire), 0);
    assert_eq!(remaining_calls.load(Ordering::Acquire), 0);
    assert_eq!(
        scheduler.schedule::<SyncTask, _>(Priority::Normal, None, |_| {}),
        Err(ExecutorError::ShuttingDown)
    );
    blocking_release_sender
        .send(())
        .expect("accepted blocking work remains connected");
    scheduler.shutdown();

    let mut compute_results = [
        compute_receiver
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("first compute result arrives before the test deadline"),
        compute_receiver
            .recv_timeout(TEST_EVENT_DEADLINE)
            .expect("second compute result arrives before the test deadline"),
    ];
    compute_results.sort_unstable();
    assert_eq!(compute_results, [(11, 0), (12, 0)]);
    assert_eq!(
        [
            blocking_receiver
                .recv_timeout(TEST_EVENT_DEADLINE)
                .expect("first blocking result arrives before the test deadline"),
            blocking_receiver
                .recv_timeout(TEST_EVENT_DEADLINE)
                .expect("second blocking result arrives before the test deadline"),
        ],
        [(21, 1), (22, 1)]
    );
    assert_eq!(capture_drops.load(Ordering::Acquire), 4);
    assert_eq!(payload_drops.load(Ordering::Acquire), 1);
    assert_eq!(
        payload_token_observed.load(Ordering::Acquire),
        payload_token
    );
    assert_eq!(remaining_calls.load(Ordering::Acquire), 0);
    assert_eq!(scheduler.metrics().pending_tasks, 0);
    assert_eq!(scheduler.metrics().active_workers, 0);
    assert_eq!(scheduler.inner.workers[0].queues.len(), 0);
    assert_eq!(
        scheduler.inner.workers[0]
            .lifo_slot
            .state
            .load(Ordering::Acquire),
        0
    );
    assert_eq!(
        scheduler.schedule::<BlockingTask, _>(Priority::Normal, None, |_| {}),
        Err(ExecutorError::ShuttingDown)
    );
    *REMAINING_HOOK_TARGET
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}
