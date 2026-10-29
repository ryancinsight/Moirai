use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

use moirai_core::{
    Priority,
    error::TaskError,
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner},
};

use super::super::HybridExecutor;
use crate::SyncTask;

const EVENT_DEADLINE: Duration = Duration::from_secs(5);

struct PanickingOutput {
    drops: Arc<AtomicUsize>,
}

impl Drop for PanickingOutput {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
        panic!("output destructor panic");
    }
}

enum FutureDestruction {
    Complete,
    Panic,
}

struct GatedCompletionFuture {
    destruction: FutureDestruction,
    ready: Arc<AtomicBool>,
    task_waker: Arc<Mutex<Option<Waker>>>,
    parked: Option<mpsc::SyncSender<()>>,
    future_drops: Arc<AtomicUsize>,
    output_drops: Arc<AtomicUsize>,
}

impl Future for GatedCompletionFuture {
    type Output = PanickingOutput;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.ready.load(Ordering::Acquire) {
            return Poll::Ready(PanickingOutput {
                drops: Arc::clone(&self.output_drops),
            });
        }

        *self.task_waker.lock().unwrap() = Some(context.waker().clone());
        if let Some(parked) = self.parked.take() {
            parked
                .send(())
                .expect("parked-future observer remains connected");
        }
        Poll::Pending
    }
}

impl Drop for GatedCompletionFuture {
    fn drop(&mut self) {
        self.future_drops.fetch_add(1, Ordering::AcqRel);
        if matches!(self.destruction, FutureDestruction::Panic) {
            panic!("future destructor panic");
        }
    }
}

struct PanicOnWake {
    invoked: mpsc::SyncSender<()>,
}

impl Wake for PanicOnWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.invoked
            .send(())
            .expect("callback observer remains connected");
        panic!("registry completion callback panic");
    }
}

#[test]
fn callback_panic_waits_for_panicking_completion_values() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let ready = Arc::new(AtomicBool::new(false));
    let task_waker = Arc::new(Mutex::new(None));
    let future_drops = Arc::new(AtomicUsize::new(0));
    let output_drops = Arc::new(AtomicUsize::new(0));
    let (parked_sender, parked_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_async(GatedCompletionFuture {
            destruction: FutureDestruction::Panic,
            ready: Arc::clone(&ready),
            task_waker: Arc::clone(&task_waker),
            parked: Some(parked_sender),
            future_drops: Arc::clone(&future_drops),
            output_drops: Arc::clone(&output_drops),
        })
        .unwrap();
    let id = handle.id();
    parked_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("future parks before the event deadline");

    let (callback_sender, callback_receiver) = mpsc::sync_channel(1);
    let callback = Waker::from(Arc::new(PanicOnWake {
        invoked: callback_sender,
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(executor.wait_for_task(id, None));
    assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));

    ready.store(true, Ordering::Release);
    task_waker
        .lock()
        .unwrap()
        .take()
        .expect("parked future retains its task waker")
        .wake();
    callback_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("completion callback runs before the event deadline");

    assert!(matches!(handle.join(), Some(Err(TaskError::Panicked))));
    let sentinel = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || 53usize)
        .unwrap();
    assert_eq!(sentinel.join(), Some(Ok(53)));
    assert_eq!(output_drops.load(Ordering::Acquire), 1);
    assert_eq!(future_drops.load(Ordering::Acquire), 1);
    assert_eq!(
        executor
            .metrics()
            .publication_panics
            .load(Ordering::Acquire),
        1
    );

    ExecutorControl::shutdown(&executor);
}

#[test]
fn uncollected_output_destruction_is_a_publication_failure() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let ready = Arc::new(AtomicBool::new(false));
    let task_waker = Arc::new(Mutex::new(None));
    let future_drops = Arc::new(AtomicUsize::new(0));
    let output_drops = Arc::new(AtomicUsize::new(0));
    let (parked_sender, parked_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_async(GatedCompletionFuture {
            destruction: FutureDestruction::Complete,
            ready: Arc::clone(&ready),
            task_waker: Arc::clone(&task_waker),
            parked: Some(parked_sender),
            future_drops: Arc::clone(&future_drops),
            output_drops: Arc::clone(&output_drops),
        })
        .unwrap();
    parked_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("future parks before release");
    drop(handle);
    ready.store(true, Ordering::Release);
    task_waker
        .lock()
        .unwrap()
        .take()
        .expect("future saved its waker")
        .wake();
    assert_eq!(executor.join(), Ok(()));
    let sentinel = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || 59usize)
        .unwrap();
    assert_eq!(sentinel.join(), Some(Ok(59)));
    assert_eq!(output_drops.load(Ordering::Acquire), 1);
    assert_eq!(future_drops.load(Ordering::Acquire), 1);
    assert_eq!(
        executor
            .metrics()
            .publication_panics
            .load(Ordering::Acquire),
        1
    );
    ExecutorControl::shutdown(&executor);
}
