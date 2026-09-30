use moirai_core::executor::{
    ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus,
};

use super::{super::HybridExecutor, fixtures::gate_single_worker};
use crate::{AsyncTask, BlockingTask};

#[test]
fn cancel_queued_task_skips_body_and_completes_cancelled() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let (release, gate_handle) = gate_single_worker::<BlockingTask>(&executor);

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_in_task = Arc::clone(&counter);
    let handle = executor
        .spawn_blocking(move || {
            counter_in_task.fetch_add(1, Ordering::Relaxed);
            7usize
        })
        .unwrap();
    let id = handle.id();

    assert_eq!(executor.task_status(id), Some(TaskStatus::Queued));
    executor.cancel_task(id).unwrap();
    release.send(()).unwrap();

    // The handle resolves to the cancelled outcome and the body never ran.
    assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Cancelled)));
    assert_eq!(counter.load(Ordering::Relaxed), 0);
    assert_eq!(executor.task_status(id), Some(TaskStatus::Cancelled));
    assert_eq!(
        executor
            .metrics()
            .tasks_cancelled
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    gate_handle.join().unwrap().unwrap();
    executor.shutdown();
}

#[test]
fn cancel_queued_async_task_skips_future_and_completes_cancelled() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let (release, gate_handle) = gate_single_worker::<AsyncTask>(&executor);

    let polls = Arc::new(AtomicUsize::new(0));
    let polls_in_future = Arc::clone(&polls);
    let handle = executor
        .spawn_async(async move {
            polls_in_future.fetch_add(1, Ordering::Relaxed);
            3usize
        })
        .unwrap();
    let id = handle.id();

    assert_eq!(executor.task_status(id), Some(TaskStatus::Queued));
    executor.cancel_task(id).unwrap();
    release.send(()).unwrap();

    assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Cancelled)));
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.task_status(id), Some(TaskStatus::Cancelled));

    gate_handle.join().unwrap().unwrap();
    executor.shutdown();
}

#[test]
fn cancel_completed_task_is_noop_ok_and_unknown_id_errors() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let handle = executor.spawn_blocking(|| 5usize).unwrap();
    let id = handle.id();
    assert_eq!(handle.join().unwrap().unwrap(), 5);

    // Already completed: no-op Ok, status remains Completed.
    executor.cancel_task(id).unwrap();
    assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));

    assert_eq!(
        executor.cancel_task(moirai_core::TaskId::new(u64::MAX / 2)),
        Err(moirai_core::error::ExecutorError::SpawnFailed(
            moirai_core::TaskError::InvalidOperation
        ))
    );
    executor.shutdown();
}

#[test]
fn cancel_running_task_is_not_preempted() {
    // Contract: cancelling a task that already started has no effect — the
    // body runs to completion and the result is the real value.
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let (release_sender, release_receiver) = std::sync::mpsc::channel::<()>();
    let (started_sender, started_receiver) = std::sync::mpsc::channel::<()>();
    let handle = executor
        .spawn_blocking(move || {
            started_sender.send(()).unwrap();
            release_receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            11usize
        })
        .unwrap();
    started_receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();

    let id = handle.id();
    executor.cancel_task(id).unwrap();
    release_sender.send(()).unwrap();

    assert_eq!(handle.join(), Some(Ok(11)));
    assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
    executor.shutdown();
}
