#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

//! The executor derives its task-registry retention from `CleanupConfig`.

use super::HybridExecutor;
use moirai_core::executor::{
    CleanupConfig, ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus,
};
use std::{
    future::Future,
    pin::pin,
    task::{Context, Poll, Waker},
    time::Duration,
};

/// Tasks per registry block; three blocks guarantee the sweep reaches block 0
/// after every one of its tasks has finished.
const SEQUENTIAL_TASKS: usize = 3 * crate::registry::state::TASK_STATE_BLOCK_SIZE;

fn executor(cleanup: CleanupConfig) -> HybridExecutor {
    HybridExecutor::new(ExecutorConfig {
        worker_threads: 2,
        cleanup,
        ..ExecutorConfig::default()
    })
    .unwrap()
}

fn run_sequentially(executor: &HybridExecutor, count: usize) -> Vec<moirai_core::task::TaskId> {
    (0..count)
        .map(|_| {
            let handle = executor.spawn_blocking(|| ()).unwrap();
            let id = handle.id();
            handle.join().unwrap().unwrap();
            id
        })
        .collect()
}

#[test]
fn expired_task_metadata_is_released_while_completion_and_cancel_stay_answerable() {
    let executor = executor(CleanupConfig {
        task_retention_duration: Duration::ZERO,
        max_retained_tasks: 0,
        ..CleanupConfig::default()
    });

    let ids = run_sequentially(&executor, SEQUENTIAL_TASKS);
    let (first, last) = (ids[0], ids[ids.len() - 1]);

    assert_eq!(executor.task_status(first), None);
    assert!(executor.task_stats(first).is_none());
    assert_eq!(executor.task_status(last), Some(TaskStatus::Completed));

    executor.cancel_task(first).unwrap();

    let wait = executor.wait_for_task(first, None);
    let mut wait = pin!(wait);
    let mut context = Context::from_waker(Waker::noop());
    assert!(matches!(
        wait.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
    executor.shutdown();
}

#[test]
fn disabled_automatic_cleanup_retains_every_task() {
    let executor = executor(CleanupConfig {
        enable_automatic_cleanup: false,
        task_retention_duration: Duration::ZERO,
        max_retained_tasks: 0,
    });

    let ids = run_sequentially(&executor, SEQUENTIAL_TASKS);

    assert_eq!(executor.task_status(ids[0]), Some(TaskStatus::Completed));
    executor.shutdown();
}
