use moirai_core::{
    TaskId,
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner},
};

use super::{super::HybridExecutor, fixtures::gate_single_worker};
use crate::{BlockingTask, TaskRegistry, counting_wake::CountingWake};

struct DeadlineEvent {
    outcome: std::sync::atomic::AtomicU8,
    waker: std::sync::Mutex<Option<std::task::Waker>>,
}

impl DeadlineEvent {
    fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            outcome: std::sync::atomic::AtomicU8::new(0),
            waker: std::sync::Mutex::new(None),
        })
    }

    fn fire(&self) {
        self.publish(1);
    }

    fn fail(&self) {
        self.publish(2);
    }

    fn publish(&self, outcome: u8) {
        self.outcome
            .store(outcome, std::sync::atomic::Ordering::Release);
        let waker = self.waker.lock().unwrap().take();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

struct EventDeadline(std::sync::Arc<DeadlineEvent>);

impl Drop for EventDeadline {
    fn drop(&mut self) {
        let waker = self.0.waker.lock().unwrap().take();
        drop(waker);
    }
}

impl std::future::Future for EventDeadline {
    type Output = Result<(), ()>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        match self.0.outcome.load(std::sync::atomic::Ordering::Acquire) {
            1 => return std::task::Poll::Ready(Ok(())),
            2 => return std::task::Poll::Ready(Err(())),
            _ => {}
        }
        let incoming = context.waker().clone();
        let previous = self.0.waker.lock().unwrap().replace(incoming);
        drop(previous);
        match self.0.outcome.load(std::sync::atomic::Ordering::Acquire) {
            1 => std::task::Poll::Ready(Ok(())),
            2 => std::task::Poll::Ready(Err(())),
            _ => std::task::Poll::Pending,
        }
    }
}

struct WakeSignal(std::sync::mpsc::SyncSender<()>);

impl std::task::Wake for WakeSignal {
    fn wake(self: std::sync::Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        if let Err(error) = self.0.try_send(()) {
            assert!(
                matches!(error, std::sync::mpsc::TrySendError::Full(())),
                "deadline wake receiver must remain connected"
            );
        }
    }
}

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

#[test]
fn controlled_deadline_and_future_drop_remove_owned_registrations() {
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    let registry = Arc::new(TaskRegistry::new());
    let task_id = TaskId(registry.register_task());
    let event = DeadlineEvent::new();
    let (wake, waker) = crate::counting_wake::counting_waker();
    let mut context = Context::from_waker(&waker);
    let mut wait = std::pin::pin!(super::super::manager::completion_wait(
        Arc::clone(&registry),
        task_id,
        Some(EventDeadline(Arc::clone(&event))),
    ));

    assert_eq!(wait.as_mut().poll(&mut context), Poll::Pending);
    assert_eq!(
        Arc::strong_count(&wake),
        4,
        "the context, deadline, and completion registration retain the target"
    );
    event.fire();
    assert_eq!(wake.wakes(), 1);
    assert_eq!(
        wait.as_mut().poll(&mut context),
        Poll::Ready(Err(moirai_core::error::ExecutorError::SpawnFailed(
            moirai_core::TaskError::Timeout
        )))
    );
    assert_eq!(Arc::strong_count(&wake), 2);

    let event = DeadlineEvent::new();
    let (dropped_wake, dropped_waker) = crate::counting_wake::counting_waker();
    let mut dropped_context = Context::from_waker(&dropped_waker);
    {
        let mut dropped = std::pin::pin!(super::super::manager::completion_wait(
            Arc::clone(&registry),
            task_id,
            Some(EventDeadline(event)),
        ));
        assert_eq!(dropped.as_mut().poll(&mut dropped_context), Poll::Pending);
        assert_eq!(Arc::strong_count(&dropped_wake), 4);
    }
    assert_eq!(Arc::strong_count(&dropped_wake), 2);

    let event = DeadlineEvent::new();
    let (failed_wake, failed_waker) = crate::counting_wake::counting_waker();
    let mut failed_context = Context::from_waker(&failed_waker);
    let mut failed = std::pin::pin!(super::super::manager::completion_wait(
        Arc::clone(&registry),
        task_id,
        Some(EventDeadline(Arc::clone(&event))),
    ));
    assert_eq!(failed.as_mut().poll(&mut failed_context), Poll::Pending);
    event.fail();
    assert_eq!(failed_wake.wakes(), 1);
    assert_eq!(
        failed.as_mut().poll(&mut failed_context),
        Poll::Ready(Err(moirai_core::error::ExecutorError::SpawnFailed(
            moirai_core::TaskError::ExecutionFailed(moirai_core::error::TaskErrorKind::Io)
        )))
    );
    assert_eq!(Arc::strong_count(&failed_wake), 2);

    let event = DeadlineEvent::new();
    let (completed_wake, completed_waker) = crate::counting_wake::counting_waker();
    let mut completed_context = Context::from_waker(&completed_waker);
    let mut completed = std::pin::pin!(super::super::manager::completion_wait(
        Arc::clone(&registry),
        task_id,
        Some(EventDeadline(Arc::clone(&event))),
    ));
    assert_eq!(
        completed.as_mut().poll(&mut completed_context),
        Poll::Pending
    );
    assert_eq!(Arc::strong_count(&event), 2);
    assert_eq!(Arc::strong_count(&completed_wake), 4);

    registry.mark_started(task_id.0, 0);
    registry.mark_completed(task_id.0);
    assert_eq!(
        completed.as_mut().poll(&mut completed_context),
        Poll::Ready(Ok(()))
    );
    assert_eq!(
        Arc::strong_count(&event),
        1,
        "a ready wait releases its deadline while the wait future remains allocated"
    );
    assert_eq!(
        Arc::strong_count(&completed_wake),
        2,
        "a ready wait releases deadline and task-registration wakers"
    );
}

#[test]
fn abandoned_task_manager_waits_release_one_thousand_distinct_wakers() {
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    const ABANDONED: usize = 1_000;
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (release, gate_handle) = gate_single_worker::<BlockingTask>(&executor);
    let handle = executor.spawn_blocking(|| 11_usize).unwrap();

    for index in 0..ABANDONED {
        let (target, waker) = crate::counting_wake::counting_waker();
        let mut context = Context::from_waker(&waker);
        {
            let mut wait = std::pin::pin!(executor.wait_for_task(handle.id(), None));
            assert_eq!(wait.as_mut().poll(&mut context), Poll::Pending);
        }
        drop(waker);
        assert_eq!(
            Arc::strong_count(&target),
            1,
            "abandoned wait {index} retained its waker payload"
        );
    }

    let (survivor, survivor_waker) = crate::counting_wake::counting_waker();
    let mut survivor_context = Context::from_waker(&survivor_waker);
    let mut survivor_wait = std::pin::pin!(executor.wait_for_task(handle.id(), None));
    assert_eq!(
        survivor_wait.as_mut().poll(&mut survivor_context),
        Poll::Pending
    );
    release.send(()).unwrap();
    assert_eq!(handle.join().unwrap().unwrap(), 11);
    executor.join().unwrap();
    assert_eq!(survivor.wakes(), 1);
    assert_eq!(
        survivor_wait.as_mut().poll(&mut survivor_context),
        Poll::Ready(Ok(()))
    );

    gate_handle.join().unwrap().unwrap();
    executor.shutdown();
}

#[test]
fn task_manager_repoll_replaces_its_registered_waker() {
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (release, gate_handle) = gate_single_worker::<BlockingTask>(&executor);
    let handle = executor.spawn_blocking(|| 13_usize).unwrap();
    let (old, old_waker) = crate::counting_wake::counting_waker();
    let (current, current_waker) = crate::counting_wake::counting_waker();
    let mut old_context = Context::from_waker(&old_waker);
    let mut current_context = Context::from_waker(&current_waker);
    let mut wait = std::pin::pin!(executor.wait_for_task(handle.id(), None));

    assert_eq!(wait.as_mut().poll(&mut old_context), Poll::Pending);
    assert_eq!(Arc::strong_count(&old), 3);
    assert_eq!(wait.as_mut().poll(&mut current_context), Poll::Pending);
    assert_eq!(Arc::strong_count(&old), 2);
    assert_eq!(Arc::strong_count(&current), 3);

    release.send(()).unwrap();
    assert_eq!(handle.join().unwrap().unwrap(), 13);
    executor.join().unwrap();
    assert_eq!(old.wakes(), 0);
    assert_eq!(current.wakes(), 1);
    assert_eq!(
        wait.as_mut().poll(&mut current_context),
        Poll::Ready(Ok(()))
    );
    assert_eq!(Arc::strong_count(&current), 2);

    gate_handle.join().unwrap().unwrap();
    executor.shutdown();
}

#[test]
fn wait_for_task_positive_timeout_wakes_and_reports_timeout() {
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (release, gate_handle) = gate_single_worker::<BlockingTask>(&executor);
    let handle = executor.spawn_blocking(|| 1usize).unwrap();
    let (wake_tx, wake_rx) = std::sync::mpsc::sync_channel(1);
    let waker = Waker::from(Arc::new(WakeSignal(wake_tx)));
    let mut context = Context::from_waker(&waker);
    let mut wait =
        std::pin::pin!(executor.wait_for_task(handle.id(), Some(Duration::from_millis(20))));

    assert!(wait.as_mut().poll(&mut context).is_pending());
    wake_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("PAL timer must wake the task waiter at its deadline");
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

#[test]
fn wait_for_task_completion_precedes_unexpired_timeout_and_drop_cancels_timer() {
    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};
    use std::time::Duration;

    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (release, gate_handle) = gate_single_worker::<BlockingTask>(&executor);
    let handle = executor.spawn_blocking(|| 1usize).unwrap();
    let (wake_tx, wake_rx) = std::sync::mpsc::sync_channel(1);
    let waker = Waker::from(Arc::new(WakeSignal(wake_tx)));
    let mut context = Context::from_waker(&waker);
    {
        let mut wait =
            std::pin::pin!(executor.wait_for_task(handle.id(), Some(Duration::from_secs(5))));
        assert!(wait.as_mut().poll(&mut context).is_pending());
        release.send(()).unwrap();
        assert_eq!(handle.join().unwrap().unwrap(), 1);
        wake_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("completion must wake the timer-backed task waiter");
        assert_eq!(wait.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    }

    gate_handle.join().unwrap().unwrap();
    executor.shutdown();
}
