use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
    time::Duration,
};

use moirai_core::{
    Priority,
    error::TaskError,
    executor::{ExecutorConfig, ExecutorControl, TaskSpawner},
};

use super::super::HybridExecutor;
use crate::SyncTask;

struct IdleFuture {
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Future for IdleFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        Poll::Pending
    }
}

impl Drop for IdleFuture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn concurrent_shutdown_snapshot_does_not_own_task_completion() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let first = executor
        .spawn_async(IdleFuture {
            polls: Arc::clone(&polls),
            drops: Arc::clone(&drops),
        })
        .unwrap();
    let second = executor
        .spawn_async(IdleFuture {
            polls: Arc::clone(&polls),
            drops: Arc::clone(&drops),
        })
        .unwrap();
    let (sentinel_sender, sentinel_receiver) = mpsc::sync_channel(0);
    let sentinel = executor
        .spawn_result::<SyncTask, _>(Priority::Low, None, move || {
            sentinel_sender
                .send(())
                .expect("sentinel observer remains connected");
        })
        .unwrap();
    sentinel_receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("queued sentinel must run after both first polls");
    assert_eq!(sentinel.join(), Some(Ok(())));
    assert_eq!(polls.load(Ordering::Relaxed), 2);

    let paused_snapshot = executor.async_tasks.close_and_snapshot();
    assert_eq!(paused_snapshot.len(), 2);
    ExecutorControl::shutdown(&executor);

    assert_eq!(first.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(second.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(polls.load(Ordering::Relaxed), 2);
    assert_eq!(drops.load(Ordering::Relaxed), 2);
    assert!(executor.async_tasks.close_and_snapshot().is_empty());
    assert_eq!(paused_snapshot.len(), 2);
    drop(paused_snapshot);
}
