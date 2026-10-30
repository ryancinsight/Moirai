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
    error::TaskError,
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus},
    task::TaskHandle,
};

use super::super::HybridExecutor;

struct GatedDropPanic {
    ready: Arc<AtomicBool>,
    waker: Arc<Mutex<Option<Waker>>>,
    parked: Option<mpsc::SyncSender<()>>,
    drops: Arc<AtomicUsize>,
}

impl Future for GatedDropPanic {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.ready.load(Ordering::Acquire) {
            return Poll::Ready(37);
        }

        *self.waker.lock().unwrap() = Some(context.waker().clone());
        if let Some(parked) = self.parked.take() {
            parked
                .send(())
                .expect("parked-future observer must remain connected");
        }
        Poll::Pending
    }
}

impl Drop for GatedDropPanic {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
        panic!("future destructor panic");
    }
}

struct JoinOnWake {
    handle: Arc<Mutex<Option<TaskHandle<usize>>>>,
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
            .expect("completion observer must remain connected");
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
            .expect("panic callback observer must remain connected");
        panic!("registry completion callback panic");
    }
}

struct GatedFixture {
    future: GatedDropPanic,
    ready: Arc<AtomicBool>,
    task_waker: Arc<Mutex<Option<Waker>>>,
    drops: Arc<AtomicUsize>,
    parked: mpsc::Receiver<()>,
}

impl GatedFixture {
    fn new() -> Self {
        let ready = Arc::new(AtomicBool::new(false));
        let task_waker = Arc::new(Mutex::new(None));
        let drops = Arc::new(AtomicUsize::new(0));
        let (parked_sender, parked) = mpsc::sync_channel(1);
        let future = GatedDropPanic {
            ready: Arc::clone(&ready),
            waker: Arc::clone(&task_waker),
            parked: Some(parked_sender),
            drops: Arc::clone(&drops),
        };
        Self {
            future,
            ready,
            task_waker,
            drops,
            parked,
        }
    }
}

#[test]
fn registry_callback_can_join_the_published_handle_on_the_poll_worker() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let GatedFixture {
        future,
        ready,
        task_waker,
        drops,
        parked,
    } = GatedFixture::new();
    let handle = executor.spawn_async(future).unwrap();
    let id = handle.id();
    parked
        .recv_timeout(Duration::from_secs(5))
        .expect("future must park before the test deadline");

    let handle = Arc::new(Mutex::new(Some(handle)));
    let (observed_sender, observed_receiver) = mpsc::sync_channel(1);
    let callback = Waker::from(Arc::new(JoinOnWake {
        handle: Arc::clone(&handle),
        observed: observed_sender,
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(executor.wait_for_task(id, None));
    assert!(wait.as_mut().poll(&mut context).is_pending());

    ready.store(true, Ordering::Release);
    task_waker
        .lock()
        .unwrap()
        .take()
        .expect("parked future must retain its task waker")
        .wake();

    assert_eq!(
        observed_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("completion callback must finish before the deadline"),
        Some(Err(TaskError::Panicked))
    );
    assert!(handle.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::Acquire), 1);

    let sentinel = executor.spawn_blocking(|| 41usize).unwrap();
    assert_eq!(sentinel.join(), Some(Ok(41)));
    ExecutorControl::shutdown(&executor);
}

#[test]
fn panicking_registry_callback_does_not_retain_the_result_sender() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let GatedFixture {
        future,
        ready,
        task_waker,
        drops,
        parked,
    } = GatedFixture::new();
    let handle = executor.spawn_async(future).unwrap();
    let id = handle.id();
    parked
        .recv_timeout(Duration::from_secs(5))
        .expect("future must park before the test deadline");

    let (invoked_sender, invoked_receiver) = mpsc::sync_channel(1);
    let callback = Waker::from(Arc::new(PanicOnWake {
        invoked: invoked_sender,
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(executor.wait_for_task(id, None));
    assert!(wait.as_mut().poll(&mut context).is_pending());

    ready.store(true, Ordering::Release);
    let retained_task_waker = task_waker
        .lock()
        .unwrap()
        .take()
        .expect("parked future must retain its task waker");
    retained_task_waker.wake_by_ref();
    invoked_receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("panicking completion callback must run before the deadline");

    assert_eq!(handle.join(), Some(Err(TaskError::Panicked)));
    assert_eq!(drops.load(Ordering::Acquire), 1);
    assert_eq!(executor.task_status(id), Some(TaskStatus::Completed));

    let sentinel = executor.spawn_blocking(|| 43usize).unwrap();
    assert_eq!(sentinel.join(), Some(Ok(43)));
    drop(retained_task_waker);
    ExecutorControl::shutdown(&executor);
}
