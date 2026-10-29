use std::{
    future::Future,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

use moirai_core::{
    Priority,
    error::TaskError,
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskStatus},
    task::TaskHandle,
};

use super::super::HybridExecutor;
use crate::{BlockingTask, SyncTask, WorkClass};

const EVENT_DEADLINE: Duration = Duration::from_secs(5);
const TASK_VALUE: usize = 47;

struct JoinOnWake {
    handle: Mutex<Option<TaskHandle<usize>>>,
    observed: mpsc::SyncSender<Option<Result<usize, TaskError>>>,
}

impl Wake for JoinOnWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let handle = self
            .handle
            .lock()
            .unwrap()
            .take()
            .expect("completion callback owns the task handle");
        self.observed
            .send(handle.join())
            .expect("completion observer remains connected");
    }
}

fn assert_metrics(executor: &HybridExecutor, spawned: u64, completed: u64, cancelled: u64) {
    let metrics = executor.metrics();
    assert_eq!(metrics.tasks_spawned.load(Ordering::Acquire), spawned);
    assert_eq!(metrics.tasks_completed.load(Ordering::Acquire), completed);
    assert_eq!(metrics.tasks_failed.load(Ordering::Acquire), 0);
    assert_eq!(metrics.tasks_cancelled.load(Ordering::Acquire), cancelled);
    assert_eq!(metrics.publication_panics.load(Ordering::Acquire), 0);
}

fn assert_completion_callback<C: WorkClass>() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (started_sender, started_receiver) = mpsc::sync_channel(1);
    let (release_sender, release_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_result::<C, _>(Priority::Normal, None, move || {
            started_sender
                .send(())
                .expect("start observer remains connected");
            release_receiver
                .recv_timeout(EVENT_DEADLINE)
                .expect("task receives its release before the event deadline");
            TASK_VALUE
        })
        .unwrap();
    let id = handle.id();
    started_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("task starts before the event deadline");

    let (observed_sender, observed_receiver) = mpsc::sync_channel(1);
    let callback = Waker::from(Arc::new(JoinOnWake {
        handle: Mutex::new(Some(handle)),
        observed: observed_sender,
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(executor.wait_for_task(id, None));
    assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));

    release_sender
        .send(())
        .expect("running task remains connected");
    assert_eq!(
        observed_receiver
            .recv_timeout(EVENT_DEADLINE)
            .expect("completion callback joins before the event deadline"),
        Some(Ok(TASK_VALUE))
    );
    assert_eq!(wait.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));
    assert_metrics(&executor, 1, 1, 0);

    ExecutorControl::shutdown(&executor);
}

fn gate_worker<C: WorkClass>(executor: &HybridExecutor) -> (mpsc::SyncSender<()>, TaskHandle<()>) {
    let (started_sender, started_receiver) = mpsc::sync_channel(1);
    let (release_sender, release_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_result::<C, _>(Priority::Normal, None, move || {
            started_sender
                .send(())
                .expect("gate start observer remains connected");
            release_receiver
                .recv_timeout(EVENT_DEADLINE)
                .expect("gate receives its release before the event deadline");
        })
        .unwrap();
    started_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("gate starts before the event deadline");
    (release_sender, handle)
}

fn assert_cancellation_callback<C: WorkClass>() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (release_sender, gate) = gate_worker::<C>(&executor);
    let body_ran = Arc::new(AtomicBool::new(false));
    let body_ran_in_task = Arc::clone(&body_ran);
    let handle = executor
        .spawn_result::<C, _>(Priority::Normal, None, move || {
            body_ran_in_task.store(true, Ordering::Release);
            TASK_VALUE
        })
        .unwrap();
    let id = handle.id();

    let (observed_sender, observed_receiver) = mpsc::sync_channel(1);
    let callback = Waker::from(Arc::new(JoinOnWake {
        handle: Mutex::new(Some(handle)),
        observed: observed_sender,
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(executor.wait_for_task(id, None));
    assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(executor.task_status(id), Some(TaskStatus::Queued));

    executor.cancel_task(id).unwrap();
    release_sender
        .send(())
        .expect("gate remains connected until release");
    assert_eq!(
        observed_receiver
            .recv_timeout(EVENT_DEADLINE)
            .expect("cancellation callback joins before the event deadline"),
        Some(Err(TaskError::Cancelled))
    );
    assert_eq!(wait.as_mut().poll(&mut context), Poll::Ready(Ok(())));
    assert_eq!(gate.join(), Some(Ok(())));
    assert!(!body_ran.load(Ordering::Acquire));
    assert_eq!(executor.task_status(id), Some(TaskStatus::Cancelled));
    assert_metrics(&executor, 2, 1, 1);

    ExecutorControl::shutdown(&executor);
}

#[test]
fn completion_callbacks_join_published_results() {
    assert_completion_callback::<SyncTask>();
    assert_completion_callback::<BlockingTask>();
}

#[test]
fn cancellation_callbacks_join_published_errors() {
    assert_cancellation_callback::<SyncTask>();
    assert_cancellation_callback::<BlockingTask>();
}
