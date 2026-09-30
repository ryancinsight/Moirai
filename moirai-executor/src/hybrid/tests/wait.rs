use moirai_core::executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner};

use super::{super::HybridExecutor, fixtures::gate_single_worker};
use crate::{BlockingTask, counting_wake::CountingWake};

#[test]
fn wait_for_task_is_woken_by_completion_not_polling() {
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let (release, gate_handle) = gate_single_worker::<BlockingTask>(&executor);
    let handle = executor.spawn_blocking(|| 9usize).unwrap();
    let id = handle.id();

    let wake = Arc::new(CountingWake(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wake));
    let mut context = Context::from_waker(&waker);
    let mut wait = std::pin::pin!(executor.wait_for_task(id, None));

    // One poll registers the completion waker; the task is still queued.
    assert!(wait.as_mut().poll(&mut context).is_pending());
    assert_eq!(wake.0.load(Ordering::Acquire), 0);

    release.send(()).unwrap();
    assert_eq!(handle.join().unwrap().unwrap(), 9);

    // Completion wakes the registered waker exactly once (no poll loop);
    // observe the wake with a bounded deadline.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while wake.0.load(Ordering::Acquire) == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "completion must wake the registered waiter"
        );
        std::thread::yield_now();
    }
    assert_eq!(wake.0.load(Ordering::Acquire), 1);
    assert_eq!(wait.as_mut().poll(&mut context), Poll::Ready(Ok(())));

    gate_handle.join().unwrap().unwrap();
    executor.shutdown();
}

#[test]
fn concurrent_waits_on_one_task_are_all_woken_by_completion() {
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::task::{Context, Poll, Waker};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    let (release, gate_handle) = gate_single_worker::<BlockingTask>(&executor);
    let handle = executor.spawn_blocking(|| 9usize).unwrap();
    let id = handle.id();

    let first_wake = Arc::new(CountingWake(std::sync::atomic::AtomicUsize::new(0)));
    let second_wake = Arc::new(CountingWake(std::sync::atomic::AtomicUsize::new(0)));
    let first_waker = Waker::from(Arc::clone(&first_wake));
    let second_waker = Waker::from(Arc::clone(&second_wake));
    let mut first_context = Context::from_waker(&first_waker);
    let mut second_context = Context::from_waker(&second_waker);
    let mut first = std::pin::pin!(executor.wait_for_task(id, None));
    let mut second = std::pin::pin!(executor.wait_for_task(id, None));

    // Both futures register while the task is still queued behind the gate.
    assert!(first.as_mut().poll(&mut first_context).is_pending());
    assert!(second.as_mut().poll(&mut second_context).is_pending());

    release.send(()).unwrap();
    assert_eq!(handle.join().unwrap().unwrap(), 9);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    for (name, wake) in [("first", &first_wake), ("second", &second_wake)] {
        while wake.0.load(Ordering::Acquire) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "completion must wake the {name} waiter"
            );
            std::thread::yield_now();
        }
        assert_eq!(wake.0.load(Ordering::Acquire), 1, "{name} waiter");
    }
    assert_eq!(first.as_mut().poll(&mut first_context), Poll::Ready(Ok(())));
    assert_eq!(
        second.as_mut().poll(&mut second_context),
        Poll::Ready(Ok(()))
    );

    gate_handle.join().unwrap().unwrap();
    executor.shutdown();
}

#[test]
fn wait_for_task_timeout_expires_and_unknown_task_errors() {
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::task::{Context, Poll, Waker};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();

    // Unknown task resolves immediately with a typed error.
    let waker = Waker::from(Arc::new(CountingWake(AtomicUsize::new(0))));
    let mut context = Context::from_waker(&waker);
    let mut unknown =
        std::pin::pin!(executor.wait_for_task(moirai_core::TaskId::new(u64::MAX / 2), None));
    assert_eq!(
        unknown.as_mut().poll(&mut context),
        Poll::Ready(Err(moirai_core::error::ExecutorError::SpawnFailed(
            moirai_core::TaskError::InvalidOperation
        )))
    );

    // A never-completing task with an already-expired deadline returns the
    // typed timeout on its first poll. This exercises the deadline branch
    // without sleeping the test thread to cross a wall-clock boundary.
    let (release, gate_handle) = gate_single_worker::<BlockingTask>(&executor);
    let handle = executor.spawn_blocking(|| 1usize).unwrap();
    let mut wait =
        std::pin::pin!(executor.wait_for_task(handle.id(), Some(std::time::Duration::ZERO)));
    assert_eq!(
        wait.as_mut().poll(&mut context),
        Poll::Ready(Err(moirai_core::error::ExecutorError::SpawnFailed(
            moirai_core::TaskError::Timeout
        )))
    );

    release.send(()).unwrap();
    assert_eq!(handle.join().unwrap().unwrap(), 1);
    gate_handle.join().unwrap().unwrap();
    executor.shutdown();
}
