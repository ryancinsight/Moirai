use std::{
    fmt::Debug,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use moirai_core::{
    error::ExecutorResult,
    executor::{ExecutorConfig, ExecutorControl},
};

use super::super::HybridExecutor;
use crate::SyncTask;

const EVENT_DEADLINE: Duration = Duration::from_secs(5);

struct CleanupGate {
    used: Arc<AtomicBool>,
    entered: mpsc::SyncSender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl CleanupGate {
    fn touch(&self) {
        self.used.store(true, Ordering::Release);
    }
}

impl Drop for CleanupGate {
    fn drop(&mut self) {
        self.entered
            .send(())
            .expect("cleanup-entry observer remains connected");
        self.release
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .recv_timeout(EVENT_DEADLINE)
            .expect("test releases admitted input cleanup");
    }
}

fn assert_cleanup_retains_facade_count<R>(
    expected: R,
    operation: impl FnOnce(&HybridExecutor, CleanupGate) -> ExecutorResult<R> + Send + 'static,
) where
    R: Debug + PartialEq + Send + 'static,
{
    let executor = Arc::new(
        HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            thread_name_prefix: "facade-count".into(),
            ..ExecutorConfig::default()
        })
        .expect("facade-count executor constructs"),
    );
    let used = Arc::new(AtomicBool::new(false));
    let (entered_sender, entered) = mpsc::sync_channel(1);
    let (release, release_receiver) = mpsc::sync_channel(1);
    let gate = CleanupGate {
        used: Arc::clone(&used),
        entered: entered_sender,
        release: Mutex::new(release_receiver),
    };
    let call_executor = Arc::clone(&executor);
    let caller = std::thread::spawn(move || operation(&call_executor, gate));
    entered
        .recv_timeout(EVENT_DEADLINE)
        .expect("facade input cleanup starts");
    assert!(used.load(Ordering::Acquire));
    assert!(executor.async_tasks.is_empty());
    assert!(!executor.has_work());
    assert_eq!(executor.admission.active_calls(), 1);

    let shutdown_executor = Arc::clone(&executor);
    let (returned_sender, returned) = mpsc::sync_channel(1);
    let shutdown = std::thread::spawn(move || {
        ExecutorControl::shutdown(&*shutdown_executor);
        returned_sender
            .send(())
            .expect("shutdown observer remains connected");
    });
    assert!(executor.shutdown.wait_for_drain(1, true, EVENT_DEADLINE));
    assert_eq!(returned.try_recv(), Err(mpsc::TryRecvError::Empty));
    release.send(()).expect("cleanup gate remains connected");

    let actual = caller
        .join()
        .expect("facade caller does not panic")
        .expect("facade operation succeeds");
    assert_eq!(actual, expected);
    returned
        .recv_timeout(EVENT_DEADLINE)
        .expect("shutdown returns after facade cleanup releases its count");
    shutdown.join().expect("shutdown caller does not panic");
    assert_eq!(executor.admission.active_calls(), 0);
}

#[test]
fn scoped_body_cleanup_retains_facade_count_with_empty_table() {
    assert_cleanup_retains_facade_count((), |executor, gate| {
        executor.scope::<SyncTask, _>(move |_| {
            gate.touch();
            Ok(())
        })
    });
}

#[test]
fn indexed_caller_cleanup_retains_facade_count_with_empty_table() {
    const COUNT: usize = 8;
    const SUM: usize = COUNT * (COUNT + 1) / 2;
    assert_cleanup_retains_facade_count(SUM, |executor, gate| {
        let sum = Arc::new(AtomicUsize::new(0));
        let result = Arc::clone(&sum);
        executor.for_each_indexed::<SyncTask, _>(COUNT, move |index| {
            gate.touch();
            sum.fetch_add(index + 1, Ordering::AcqRel);
        })?;
        Ok(result.load(Ordering::Acquire))
    });
}

#[test]
fn reducer_cleanup_retains_facade_count_with_empty_table() {
    assert_cleanup_retains_facade_count(10usize, |executor, gate| {
        executor.map_reduce_indexed::<SyncTask, _, _, _>(
            4,
            0,
            |index| index + 1,
            move |left, right| {
                gate.touch();
                left + right
            },
        )
    });
}
