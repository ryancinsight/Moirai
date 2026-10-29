use super::Moirai;
use moirai_core::{
    TaskError,
    executor::{TaskManager, TaskStatus},
};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll, Waker},
    time::Duration,
};

struct Parked {
    entered: Option<mpsc::SyncSender<()>>,
    release: Option<mpsc::Receiver<()>>,
    waker: Arc<Mutex<Option<Waker>>>,
    drops: Arc<AtomicUsize>,
}

impl Future for Parked {
    type Output = usize;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<usize> {
        *self
            .waker
            .lock()
            .expect("test waker lock remains unpoisoned") = Some(cx.waker().clone());
        if let Some(entered) = self.entered.take() {
            entered
                .send(())
                .expect("first-poll observer remains connected");
        }
        if let Some(release) = self.release.take() {
            release
                .recv_timeout(Duration::from_secs(5))
                .expect("poll owner is released by the test");
        }
        Poll::Pending
    }
}

impl Drop for Parked {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

enum Settlement {
    Cancel,
    Shutdown,
}

fn settle_parked(settlement: Settlement) {
    let runtime = Moirai::builder()
        .worker_threads(1)
        .build()
        .expect("test runtime builds");
    let (entered, observed) = mpsc::sync_channel(1);
    let waker = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let handle = runtime.spawn_async(Parked {
        entered: Some(entered),
        release: None,
        waker: Arc::clone(&waker),
        drops: Arc::clone(&drops),
    });
    observed
        .recv_timeout(Duration::from_secs(5))
        .expect("future reaches first poll");
    // On the sole worker this sentinel runs after the pending poll returns.
    assert_eq!(runtime.spawn_async(async { 13_usize }).join(), Some(Ok(13)));
    match settlement {
        Settlement::Shutdown => runtime.shutdown(),
        Settlement::Cancel => runtime
            .cancel_task(handle.id())
            .expect("registered task cancels"),
    }
    let id = handle.id();
    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(runtime.task_status(id), Some(TaskStatus::Cancelled));
    // The external event source still owns a waker, but no longer owns a future.
    waker
        .lock()
        .expect("test waker lock remains unpoisoned")
        .take()
        .expect("first poll publishes its waker")
        .wake();
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    runtime.shutdown();
}

#[test]
fn cancel_settles_without_an_external_wake() {
    settle_parked(Settlement::Cancel);
}

#[test]
fn shutdown_settles_without_an_external_wake() {
    settle_parked(Settlement::Shutdown);
}

#[test]
fn cancellation_does_not_drop_a_future_while_it_is_polling() {
    let runtime = Moirai::builder()
        .worker_threads(1)
        .build()
        .expect("test runtime builds");
    let (entered, observed) = mpsc::sync_channel(1);
    let (release, held) = mpsc::channel();
    let drops = Arc::new(AtomicUsize::new(0));
    let handle = runtime.spawn_async(Parked {
        entered: Some(entered),
        release: Some(held),
        waker: Arc::new(Mutex::new(None)),
        drops: Arc::clone(&drops),
    });
    observed
        .recv_timeout(Duration::from_secs(5))
        .expect("future enters poll");
    runtime
        .cancel_task(handle.id())
        .expect("registered task cancels");
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    release.send(()).expect("poll owner is waiting");
    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    runtime.shutdown();
}

#[test]
fn pending_task_without_a_saved_waker_remains_owned_until_cancelled() {
    let runtime = Moirai::builder()
        .worker_threads(1)
        .build()
        .expect("test runtime builds");
    let (entered, observed) = mpsc::sync_channel(1);
    let handle = runtime.spawn_async(async move {
        let mut entered = Some(entered);
        std::future::poll_fn(move |_| {
            if let Some(entered) = entered.take() {
                entered
                    .send(())
                    .expect("first-poll observer remains connected");
            }
            Poll::<usize>::Pending
        })
        .await
    });
    observed
        .recv_timeout(Duration::from_secs(5))
        .expect("future enters poll");
    assert_eq!(runtime.spawn_async(async { 13_usize }).join(), Some(Ok(13)));
    assert!(!handle.is_finished());
    assert_eq!(runtime.task_status(handle.id()), Some(TaskStatus::Running));
    runtime
        .cancel_task(handle.id())
        .expect("registered task cancels");
    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    runtime.shutdown();
}
