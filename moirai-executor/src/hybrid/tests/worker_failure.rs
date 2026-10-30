use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
    time::Duration,
};

use moirai_core::{
    Priority, TaskError,
    executor::{ExecutorConfig, ExecutorControl, TaskSpawner},
};

use super::super::HybridExecutor;
use crate::{SyncTask, schedule::register_idle_hook};

const EVENT_DEADLINE: Duration = Duration::from_secs(5);
static HOOK: OnceLock<WorkerHook> = OnceLock::new();

struct WorkerHook {
    failure: Mutex<mpsc::Receiver<HookFailure>>,
    entered: mpsc::SyncSender<()>,
}

enum HookFailure {
    Cleanup(WorkerFailure),
    Drain(IdleHookFailure),
}

struct CleanupFailure(Arc<AtomicUsize>);

impl Drop for CleanupFailure {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

struct PendingFuture {
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    polled: Option<mpsc::SyncSender<()>>,
}

impl Future for PendingFuture {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        self.polled
            .take()
            .expect("future is polled once before shutdown")
            .send(())
            .expect("poll observer remains connected");
        Poll::Pending
    }
}

impl Drop for PendingFuture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct IdleHookFailure {
    executor: Arc<HybridExecutor>,
    token: usize,
    disposed: mpsc::SyncSender<(usize, bool)>,
    drops: Arc<AtomicUsize>,
}

impl Drop for IdleHookFailure {
    fn drop(&mut self) {
        self.disposed
            .send((self.token, self.executor.scheduler.is_current_worker()))
            .expect("payload disposal observer remains connected");
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct WorkerFailure {
    executor: Arc<HybridExecutor>,
    entered: mpsc::SyncSender<(bool, bool)>,
    release: mpsc::Receiver<()>,
    first_drops: Arc<AtomicUsize>,
    second_drops: Arc<AtomicUsize>,
}

impl Drop for WorkerFailure {
    fn drop(&mut self) {
        ExecutorControl::shutdown(&*self.executor);
        self.entered
            .send((
                self.executor.scheduler.is_current_worker(),
                self.executor.admission.is_shutdown_worker(),
            ))
            .unwrap();
        self.release.recv_timeout(EVENT_DEADLINE).unwrap();
        self.first_drops.fetch_add(1, Ordering::Relaxed);
        std::panic::panic_any(CleanupFailure(Arc::clone(&self.second_drops)));
    }
}

fn fail_idle_worker() {
    let hook = HOOK.get().unwrap();
    hook.entered.send(()).unwrap();
    let failure = hook
        .failure
        .lock()
        .unwrap()
        .recv_timeout(EVENT_DEADLINE)
        .unwrap();
    match failure {
        HookFailure::Cleanup(failure) => std::panic::panic_any(failure),
        HookFailure::Drain(failure) => std::panic::panic_any(failure),
    }
}

#[test]
fn concurrent_shutdown_waits_for_worker_panic_cleanup() {
    let (failure_sender, failure) = mpsc::sync_channel(1);
    let (hook_entered_sender, hook_entered) = mpsc::sync_channel(1);
    HOOK.set(WorkerHook {
        failure: Mutex::new(failure),
        entered: hook_entered_sender,
    })
    .unwrap_or_else(|_| panic!("fixture installs its receiver once"));
    register_idle_hook(fail_idle_worker).unwrap();
    let executor = Arc::new(
        HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            thread_name_prefix: "worker-panic-cleanup".into(),
            ..ExecutorConfig::default()
        })
        .unwrap(),
    );
    hook_entered.recv_timeout(EVENT_DEADLINE).unwrap();
    let first_drops = Arc::new(AtomicUsize::new(0));
    let second_drops = Arc::new(AtomicUsize::new(0));
    let (entered_sender, entered) = mpsc::sync_channel(1);
    let (release, release_receiver) = mpsc::sync_channel(1);
    failure_sender
        .send(HookFailure::Cleanup(WorkerFailure {
            executor: Arc::clone(&executor),
            entered: entered_sender,
            release: release_receiver,
            first_drops: Arc::clone(&first_drops),
            second_drops: Arc::clone(&second_drops),
        }))
        .unwrap_or_else(|_| panic!("worker must receive its panic payload"));

    let (returned_sender, returned) = mpsc::sync_channel(2);
    let callers = std::array::from_fn::<_, 2, _>(|_| {
        let executor = Arc::clone(&executor);
        let returned_sender = returned_sender.clone();
        let first_drops = Arc::clone(&first_drops);
        let second_drops = Arc::clone(&second_drops);
        std::thread::spawn(move || {
            ExecutorControl::shutdown(&*executor);
            returned_sender
                .send((
                    first_drops.load(Ordering::Relaxed),
                    second_drops.load(Ordering::Relaxed),
                ))
                .unwrap();
        })
    });
    assert_eq!(entered.recv_timeout(EVENT_DEADLINE).unwrap(), (false, true));
    assert!(executor.shutdown.wait_for_waiters(2, EVENT_DEADLINE));
    assert_eq!(first_drops.load(Ordering::Relaxed), 0);
    assert_eq!(second_drops.load(Ordering::Relaxed), 0);
    release.send(()).unwrap();
    for _ in 0..2 {
        assert_eq!(returned.recv_timeout(EVENT_DEADLINE).unwrap(), (1, 1));
    }
    for caller in callers {
        caller.join().unwrap();
    }
}

#[test]
fn idle_hook_failure_drains_accepted_facade_work_before_payload_disposal() {
    let (failure_sender, failure) = mpsc::sync_channel(1);
    let (hook_entered_sender, hook_entered) = mpsc::sync_channel(1);
    HOOK.set(WorkerHook {
        failure: Mutex::new(failure),
        entered: hook_entered_sender,
    })
    .unwrap_or_else(|_| panic!("fixture installs its receiver once"));
    register_idle_hook(fail_idle_worker).unwrap();
    let executor = Arc::new(
        HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            thread_name_prefix: "worker-failure-drain".into(),
            ..ExecutorConfig::default()
        })
        .unwrap(),
    );
    hook_entered.recv_timeout(EVENT_DEADLINE).unwrap();

    let future_polls = Arc::new(AtomicUsize::new(0));
    let future_drops = Arc::new(AtomicUsize::new(0));
    let (future_polled_sender, future_polled) = mpsc::sync_channel(1);
    let pending = executor
        .spawn_async(PendingFuture {
            polls: Arc::clone(&future_polls),
            drops: Arc::clone(&future_drops),
            polled: Some(future_polled_sender),
        })
        .unwrap();

    let capture_drops = Arc::new(AtomicUsize::new(0));
    let compute_capture = DropCounter(Arc::clone(&capture_drops));
    let (compute_sender, compute_receiver) = mpsc::sync_channel(1);
    let compute = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, move || {
            let _capture = compute_capture;
            compute_sender.send(31).unwrap();
            31
        })
        .unwrap();
    let blocking_capture = DropCounter(Arc::clone(&capture_drops));
    let blocking = executor
        .spawn_blocking(move || {
            let _capture = blocking_capture;
            compute_receiver.recv_timeout(EVENT_DEADLINE).unwrap() * 2
        })
        .unwrap();

    let payload_drops = Arc::new(AtomicUsize::new(0));
    let (disposed_sender, disposed) = mpsc::sync_channel(1);
    failure_sender
        .send(HookFailure::Drain(IdleHookFailure {
            executor: Arc::clone(&executor),
            token: 0x51a7,
            disposed: disposed_sender,
            drops: Arc::clone(&payload_drops),
        }))
        .unwrap();

    assert_eq!(compute.join(), Some(Ok(31)));
    assert_eq!(blocking.join(), Some(Ok(62)));
    future_polled.recv_timeout(EVENT_DEADLINE).unwrap();
    assert_eq!(capture_drops.load(Ordering::Relaxed), 2);
    assert_eq!(future_polls.load(Ordering::Relaxed), 1);
    assert_eq!(payload_drops.load(Ordering::Relaxed), 0);

    ExecutorControl::shutdown(&*executor);

    assert_eq!(pending.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(future_drops.load(Ordering::Relaxed), 1);
    assert_eq!(
        disposed.recv_timeout(EVENT_DEADLINE).unwrap(),
        (0x51a7, false)
    );
    assert_eq!(payload_drops.load(Ordering::Relaxed), 1);
    assert!(executor.async_tasks.is_empty());
    assert_eq!(executor.pending_tasks(), 0);
    assert_eq!(executor.active_workers(), 0);
    assert!(!executor.has_work());
    assert_eq!(Arc::strong_count(&executor), 1);
}
