use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use moirai_core::{
    Priority,
    error::ExecutorError,
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus},
    task::TaskId,
};

use super::{super::HybridExecutor, fixtures::gate_single_worker};
use crate::SyncTask;

struct UnpolledFuture {
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Future for UnpolledFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        Poll::Pending
    }
}

impl Drop for UnpolledFuture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn rejected_first_async_admission_releases_tracking_and_future() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        max_global_queue_size: 2,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (release, gate) = gate_single_worker::<SyncTask>(&executor);
    let first = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    let second = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    let rejected_id = TaskId::new(
        second
            .id()
            .0
            .checked_add(1)
            .expect("invariant: test task id has a successor"),
    );
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));

    let rejection = match executor.spawn_async(UnpolledFuture {
        polls: Arc::clone(&polls),
        drops: Arc::clone(&drops),
    }) {
        Ok(_) => panic!("the third queued task must exceed the two-slot admission bound"),
        Err(error) => error,
    };

    assert!(matches!(rejection, ExecutorError::ResourceExhausted(_)));
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(
        executor.task_status(rejected_id),
        Some(TaskStatus::Cancelled)
    );

    release.send(()).unwrap();
    assert_eq!(gate.join(), Some(Ok(())));
    assert_eq!(first.join(), Some(Ok(())));
    assert_eq!(second.join(), Some(Ok(())));
    ExecutorControl::shutdown(&executor);
}

#[test]
fn closed_async_registration_rejects_and_drops_unpolled_future() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    assert!(executor.async_tasks.close_and_snapshot().is_empty());
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));

    let rejection = match executor.spawn_async(UnpolledFuture {
        polls: Arc::clone(&polls),
        drops: Arc::clone(&drops),
    }) {
        Ok(_) => panic!("closed async registration must reject the spawn"),
        Err(error) => error,
    };

    assert!(matches!(rejection, ExecutorError::ShuttingDown));
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    let successor = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    let rejected_id = TaskId::new(
        successor
            .id()
            .0
            .checked_sub(1)
            .expect("invariant: successor follows the rejected task"),
    );
    assert_eq!(
        executor.task_status(rejected_id),
        Some(TaskStatus::Cancelled)
    );
    assert_eq!(successor.join(), Some(Ok(())));
    ExecutorControl::shutdown(&executor);
}
