use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
    time::Duration,
};

use moirai_core::{
    Priority,
    error::{ExecutorError, TaskError},
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus},
    task::TaskId,
};

use super::{super::HybridExecutor, fixtures::gate_single_worker};
use crate::SyncTask;

struct DestructionProbe {
    accessed: AtomicBool,
    drops: AtomicUsize,
    polls: AtomicUsize,
}

impl DestructionProbe {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            accessed: AtomicBool::new(false),
            drops: AtomicUsize::new(0),
            polls: AtomicUsize::new(0),
        })
    }

    fn enter(&self) {
        assert!(
            !self.accessed.swap(true, Ordering::AcqRel),
            "future access must remain exclusive"
        );
    }

    fn leave(&self) {
        assert!(
            self.accessed.swap(false, Ordering::AcqRel),
            "future access guard must be held"
        );
    }

    fn destroy(&self) {
        self.enter();
        self.drops.fetch_add(1, Ordering::AcqRel);
        self.leave();
        panic!("future destructor panic");
    }
}

struct ReadyDropPanic {
    probe: Arc<DestructionProbe>,
}

impl Future for ReadyDropPanic {
    type Output = usize;

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        self.probe.enter();
        self.probe.polls.fetch_add(1, Ordering::AcqRel);
        self.probe.leave();
        Poll::Ready(17)
    }
}

impl Drop for ReadyDropPanic {
    fn drop(&mut self) {
        self.probe.destroy();
    }
}

struct ParkedDropPanic {
    probe: Arc<DestructionProbe>,
    parked: Option<mpsc::SyncSender<()>>,
}

impl Future for ParkedDropPanic {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        self.probe.enter();
        self.probe.polls.fetch_add(1, Ordering::AcqRel);
        if let Some(parked) = self.parked.take() {
            parked
                .send(())
                .expect("parked-future observer must remain connected");
        }
        self.probe.leave();
        Poll::Pending
    }
}

impl Drop for ParkedDropPanic {
    fn drop(&mut self) {
        self.probe.destroy();
    }
}

#[test]
fn ready_future_destructor_panic_settles_as_panicked() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let probe = DestructionProbe::new();
    let handle = executor
        .spawn_async(ReadyDropPanic {
            probe: Arc::clone(&probe),
        })
        .unwrap();
    let id = handle.id();

    assert_eq!(handle.join(), Some(Err(TaskError::Panicked)));
    assert_eq!(probe.drops.load(Ordering::Acquire), 1);
    assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
    assert_eq!(executor.metrics().tasks_failed.load(Ordering::Acquire), 1);
    assert_eq!(
        executor.metrics().tasks_completed.load(Ordering::Acquire),
        0
    );
    assert_eq!(
        executor.metrics().tasks_cancelled.load(Ordering::Acquire),
        0
    );

    let sentinel = executor.spawn_blocking(|| 23usize).unwrap();
    assert_eq!(sentinel.join(), Some(Ok(23)));
    executor.join().unwrap();
    assert!(!executor.has_work());
    executor.shutdown();
}

#[test]
fn shutdown_future_destructor_panic_settles_as_panicked() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let probe = DestructionProbe::new();
    let (parked_sender, parked_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_async(ParkedDropPanic {
            probe: Arc::clone(&probe),
            parked: Some(parked_sender),
        })
        .unwrap();
    let id = handle.id();

    parked_receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("future must park before the test deadline");
    let sentinel = executor.spawn_blocking(|| 29usize).unwrap();
    assert_eq!(sentinel.join(), Some(Ok(29)));

    executor.shutdown();

    assert_eq!(handle.join(), Some(Err(TaskError::Panicked)));
    assert_eq!(probe.drops.load(Ordering::Acquire), 1);
    assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
    assert_eq!(executor.metrics().tasks_failed.load(Ordering::Acquire), 1);
    assert_eq!(
        executor.metrics().tasks_cancelled.load(Ordering::Acquire),
        0
    );
    assert!(!executor.has_work());
}

#[test]
fn cancelled_future_destructor_panic_settles_as_panicked() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let probe = DestructionProbe::new();
    let (parked_sender, parked_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_async(ParkedDropPanic {
            probe: Arc::clone(&probe),
            parked: Some(parked_sender),
        })
        .unwrap();
    let id = handle.id();

    parked_receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("future must park before the test deadline");
    executor.cancel_task(id).unwrap();

    assert_eq!(handle.join(), Some(Err(TaskError::Panicked)));
    assert_eq!(probe.drops.load(Ordering::Acquire), 1);
    assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
    assert_eq!(executor.metrics().tasks_failed.load(Ordering::Acquire), 1);
    assert_eq!(
        executor.metrics().tasks_cancelled.load(Ordering::Acquire),
        0
    );

    let sentinel = executor.spawn_blocking(|| 31usize).unwrap();
    assert_eq!(sentinel.join(), Some(Ok(31)));
    executor.join().unwrap();
    assert!(!executor.has_work());
    ExecutorControl::shutdown(&executor);
}

#[test]
fn rejected_future_destructor_panic_surfaces_as_panicked() {
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
    let probe = DestructionProbe::new();

    let rejection = match executor.spawn_async(ReadyDropPanic {
        probe: Arc::clone(&probe),
    }) {
        Ok(_) => panic!("the third queued task must exceed the admission bound"),
        Err(error) => error,
    };

    assert!(matches!(
        rejection,
        ExecutorError::SpawnFailed(TaskError::Panicked)
    ));
    assert_eq!(probe.drops.load(Ordering::Acquire), 1);
    assert_eq!(probe.polls.load(Ordering::Acquire), 0);
    assert_eq!(
        executor.task_status(rejected_id),
        Some(TaskStatus::Completed)
    );

    release.send(()).unwrap();
    assert_eq!(gate.join(), Some(Ok(())));
    assert_eq!(first.join(), Some(Ok(())));
    assert_eq!(second.join(), Some(Ok(())));
    ExecutorControl::shutdown(&executor);
}

#[test]
fn closed_registration_future_destructor_panic_surfaces_as_panicked() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    assert!(executor.async_tasks.close_and_snapshot().is_empty());
    let probe = DestructionProbe::new();

    let rejection = match executor.spawn_async(ReadyDropPanic {
        probe: Arc::clone(&probe),
    }) {
        Ok(_) => panic!("closed async registration must reject the spawn"),
        Err(error) => error,
    };

    assert!(matches!(
        rejection,
        ExecutorError::SpawnFailed(TaskError::Panicked)
    ));
    assert_eq!(probe.drops.load(Ordering::Acquire), 1);
    assert_eq!(probe.polls.load(Ordering::Acquire), 0);
    ExecutorControl::shutdown(&executor);
}
