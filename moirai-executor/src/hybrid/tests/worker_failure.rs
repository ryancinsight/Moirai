use std::{
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use moirai_core::executor::{ExecutorConfig, ExecutorControl};

use super::super::HybridExecutor;
use crate::schedule::register_idle_hook;

const EVENT_DEADLINE: Duration = Duration::from_secs(5);
static HOOK: OnceLock<WorkerHook> = OnceLock::new();

struct WorkerHook {
    failure: Mutex<mpsc::Receiver<WorkerFailure>>,
    entered: mpsc::SyncSender<()>,
}

struct CleanupFailure(Arc<AtomicUsize>);

impl Drop for CleanupFailure {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
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
    std::panic::panic_any(failure);
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
        .send(WorkerFailure {
            executor: Arc::clone(&executor),
            entered: entered_sender,
            release: release_receiver,
            first_drops: Arc::clone(&first_drops),
            second_drops: Arc::clone(&second_drops),
        })
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
