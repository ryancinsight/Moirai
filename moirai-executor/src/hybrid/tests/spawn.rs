use moirai_core::executor::{
    ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus,
};

use super::super::HybridExecutor;

#[test]
fn spawn_blocking_returns_value_and_updates_status() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 2,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let handle = executor.spawn_blocking(|| 21 * 2).unwrap();
    let id = handle.id();
    let result = handle.join().unwrap().unwrap();

    assert_eq!(result, 42);
    assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
    executor.shutdown();
}

#[test]
fn spawn_blocking_reports_panicked_result() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let handle = executor
        .spawn_blocking(|| -> usize { panic!("blocking task panic") })
        .unwrap();

    assert_eq!(handle.join(), Some(Err(moirai_core::TaskError::Panicked)));
    executor.shutdown();
}

#[test]
fn spawn_detached_runs_every_task_and_drains_on_shutdown() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 4,
        ..ExecutorConfig::default()
    })
    .unwrap();

    const TASKS: usize = 256;
    let counter = Arc::new(AtomicUsize::new(0));
    for _ in 0..TASKS {
        let c = Arc::clone(&counter);
        // Returns `()`: no handle, no `Arc<TaskResultSlot>` allocated.
        executor
            .spawn_detached(move || {
                c.fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
    }

    // `shutdown` drains all pending work before returning, so every detached
    // closure must have executed exactly once.
    executor.shutdown();
    assert_eq!(counter.load(Ordering::Relaxed), TASKS);
}

#[test]
fn spawn_detached_isolates_panics() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    // A panicking detached task must not abort its worker thread.
    executor
        .spawn_detached(|| panic!("detached task panic"))
        .unwrap();

    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    executor
        .spawn_detached(move || {
            c.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();

    executor.shutdown();
    // The single worker survived the panic and ran the following task.
    assert_eq!(counter.load(Ordering::Relaxed), 1);
}

#[test]
fn spawn_async_uses_unified_scheduler() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 2,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let handle = executor.spawn_async(async { 7usize }).unwrap();
    let result = handle.join().unwrap().unwrap();

    assert_eq!(result, 7);
    assert_eq!(executor.worker_count(), 2);
    executor.shutdown();
}

#[test]
fn spawn_async_requeues_after_wake_without_blocking_worker() {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    use std::task::Waker;
    use std::time::Duration;

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let ready = Arc::new(AtomicBool::new(false));
    let waker_slot = Arc::new(Mutex::new(None::<Waker>));
    let (waker_published_tx, waker_published_rx) = mpsc::sync_channel(1);
    let ready_for_future = Arc::clone(&ready);
    let waker_for_future = Arc::clone(&waker_slot);
    let handle = executor
        .spawn_async(async move {
            let mut publish = Some(waker_published_tx);
            std::future::poll_fn(move |cx| {
                if ready_for_future.load(Ordering::Acquire) {
                    std::task::Poll::Ready(21usize)
                } else {
                    *waker_for_future.lock().unwrap() = Some(cx.waker().clone());
                    if let Some(publish) = publish.take() {
                        publish.send(()).expect("waker observer is alive");
                    }
                    std::task::Poll::Pending
                }
            })
            .await
        })
        .unwrap();

    waker_published_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("async future must publish a waker before timeout");
    let waker = waker_slot
        .lock()
        .unwrap()
        .take()
        .expect("waker publication stores the waker");

    let (ran_sender, ran_receiver) = mpsc::channel();
    let independent = executor
        .spawn_blocking(move || {
            ran_sender.send(()).unwrap();
            13usize
        })
        .unwrap();

    ran_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("pending async future must not block the only worker");

    ready.store(true, Ordering::Release);
    waker.wake();

    assert_eq!(independent.join().unwrap().unwrap(), 13);
    assert_eq!(handle.join().unwrap().unwrap(), 21);
    executor.shutdown();
}

#[test]
fn spawn_async_completes_single_self_wake() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let poll_count = Arc::new(AtomicUsize::new(0));
    let poll_count_for_future = Arc::clone(&poll_count);
    let handle = executor
        .spawn_async(async move {
            std::future::poll_fn(move |context| {
                match poll_count_for_future.fetch_add(1, Ordering::AcqRel) {
                    0 => {
                        context.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                    previous => std::task::Poll::Ready(previous + 1),
                }
            })
            .await
        })
        .unwrap();

    assert_eq!(handle.join().unwrap().unwrap(), 2);
    assert_eq!(poll_count.load(Ordering::Acquire), 2);
    executor.shutdown();
}
