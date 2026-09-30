use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use moirai_core::executor::{
    ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus,
};

use super::super::HybridExecutor;
use crate::{SyncTask, counting_wake::CountingWake};

#[test]
fn scoped_chunks_complete_before_inherent_shutdown() {
    const LEN: usize = 257;
    const CHUNK: usize = 7;
    const ROUNDS: usize = 64;

    let mut executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 2,
        ..ExecutorConfig::default()
    })
    .unwrap();

    for round in 0..ROUNDS {
        let mut values = vec![usize::MAX; LEN];
        executor
            .scope::<SyncTask, _>(|scope| {
                for (chunk_index, chunk) in values.chunks_mut(CHUNK).enumerate() {
                    let first = chunk_index * CHUNK;
                    scope.spawn(move |_| {
                        for (offset, value) in chunk.iter_mut().enumerate() {
                            *value = first + offset;
                        }
                    })?;
                }
                Ok(())
            })
            .unwrap();

        for (index, value) in values.into_iter().enumerate() {
            assert_eq!(value, index, "round {round} lost logical slot {index}");
        }
    }
    HybridExecutor::shutdown(&mut executor).unwrap();
}

/// An async task parked on an external event is woken after shutdown: the
/// scheduler refuses the poll, so the task ends cancelled at once. A waiter
/// on it resolves rather than pending until the event source drops its
/// last waker clone, and status agrees with what the handle reports.
#[test]
fn waking_an_async_task_after_shutdown_resolves_its_waiter_as_cancelled() {
    use std::future::Future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let waker_slot = Arc::new(Mutex::new(None::<Waker>));
    let (published_tx, published_rx) = mpsc::sync_channel(1);
    let waker_for_future = Arc::clone(&waker_slot);
    let handle = executor
        .spawn_async(async move {
            let mut publish = Some(published_tx);
            std::future::poll_fn(move |cx| {
                *waker_for_future.lock().unwrap() = Some(cx.waker().clone());
                if let Some(publish) = publish.take() {
                    publish.send(()).expect("waker observer is alive");
                }
                Poll::<usize>::Pending
            })
            .await
        })
        .unwrap();
    published_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("async future must publish a waker before the deadline");
    let event_source_waker = waker_slot
        .lock()
        .unwrap()
        .take()
        .expect("waker publication stores the waker");
    let id = handle.id();

    let wake = Arc::new(CountingWake(AtomicUsize::new(0)));
    let waiter = Waker::from(Arc::clone(&wake));
    let mut context = Context::from_waker(&waiter);
    let mut wait = std::pin::pin!(executor.wait_for_task(id, None));
    assert!(wait.as_mut().poll(&mut context).is_pending());
    assert_eq!(executor.task_status(id), Some(TaskStatus::Running));

    // The event source keeps a waker clone, so the task state stays
    // reachable after the wake below and cannot resolve by being dropped.
    let event_source_clone = event_source_waker.clone();
    executor.shutdown();
    event_source_waker.wake();

    assert_eq!(
        wake.0.load(Ordering::Acquire),
        1,
        "the refused wake must complete the task and wake its waiter"
    );
    assert_eq!(wait.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    assert_eq!(executor.task_status(id), Some(TaskStatus::Cancelled));
    assert_eq!(
        handle.join(),
        Some(Err(moirai_core::error::TaskError::Cancelled))
    );
    drop(event_source_clone);
}

#[test]
fn shutdown_timeout_bounds_the_callers_wait() {
    let executor = Arc::new(
        HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            ..ExecutorConfig::default()
        })
        .unwrap(),
    );

    let (release_sender, release_receiver) = std::sync::mpsc::channel::<()>();
    let (started_sender, started_receiver) = std::sync::mpsc::channel::<()>();
    executor
        .spawn_detached(move || {
            started_sender.send(()).unwrap();
            release_receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
        })
        .unwrap();
    started_receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();

    // The worker is blocked, so a full drain cannot finish; the call must
    // return after its bound while the drain continues behind it. Observe
    // the return as an event instead of comparing wall-clock elapsed time.
    let (returned_sender, returned_receiver) = std::sync::mpsc::sync_channel(0);
    let shutdown_executor = Arc::clone(&executor);
    let shutdown_thread = std::thread::spawn(move || {
        shutdown_executor.shutdown_timeout(std::time::Duration::from_millis(50));
        returned_sender
            .send(shutdown_executor.is_shutting_down())
            .expect("shutdown observer must remain connected");
    });
    assert!(
        returned_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("shutdown_timeout must return within the test bound")
    );

    // Release the worker so the background drain and drop complete.
    release_sender.send(()).unwrap();
    shutdown_thread.join().unwrap();
    drop(executor);
}

#[test]
fn join_waits_for_public_result_tasks_without_shutdown() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 2,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let handles = (0..8)
        .map(|value| executor.spawn_blocking(move || value + 1).unwrap())
        .collect::<Vec<_>>();

    assert!(executor.has_work());
    executor.join().unwrap();
    assert!(!executor.has_work());

    let results = handles
        .into_iter()
        .map(|handle| handle.join().unwrap().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(results, (1..=8).collect::<Vec<_>>());
    executor.shutdown();
}
