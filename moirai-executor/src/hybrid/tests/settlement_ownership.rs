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
    error::{ExecutorError, TaskError},
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner},
    task::TaskHandle,
};

use super::{super::HybridExecutor, fixtures::gate_single_worker};
use crate::SyncTask;

const EVENT_DEADLINE: Duration = Duration::from_secs(5);

struct InlineCompletion {
    ready: Arc<AtomicBool>,
    waker: Arc<Mutex<Option<Waker>>>,
    parked: Option<mpsc::SyncSender<()>>,
    drops: Arc<AtomicUsize>,
}

impl Future for InlineCompletion {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.ready.load(Ordering::Acquire) {
            return Poll::Ready(1789);
        }
        *self.waker.lock().unwrap() = Some(context.waker().clone());
        if let Some(parked) = self.parked.take() {
            parked.send(()).expect("park observer remains connected");
        }
        Poll::Pending
    }
}

impl Drop for InlineCompletion {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
    }
}

enum CallbackExit {
    Release(Mutex<mpsc::Receiver<()>>),
    Panic(Mutex<Option<GatedPanicPayload>>),
}

struct GatedPanicPayload {
    entered: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
    drops: Arc<AtomicUsize>,
}

impl Drop for GatedPanicPayload {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
        self.entered
            .send(())
            .expect("payload-drop observer remains connected");
        self.release
            .recv()
            .expect("payload-drop release remains connected");
    }
}

struct SettlementCallback {
    handle: Mutex<Option<TaskHandle<usize>>>,
    observed: mpsc::SyncSender<Option<Result<usize, TaskError>>>,
    entered: mpsc::SyncSender<()>,
    exit: CallbackExit,
}

impl Wake for SettlementCallback {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let result = self
            .handle
            .lock()
            .unwrap()
            .take()
            .expect("settlement callback owns the result handle")
            .join();
        self.observed
            .send(result)
            .expect("result observer remains connected");
        self.entered
            .send(())
            .expect("callback observer remains connected");
        match &self.exit {
            CallbackExit::Release(release) => release
                .lock()
                .unwrap()
                .recv()
                .expect("callback release remains connected"),
            CallbackExit::Panic(payload) => std::panic::panic_any(
                payload
                    .lock()
                    .unwrap()
                    .take()
                    .expect("settlement callback owns its panic payload"),
            ),
        }
    }
}

struct SettlementFixture {
    worker_release: mpsc::Sender<()>,
    worker_gate: TaskHandle<()>,
    executor: HybridExecutor,
    handle: TaskHandle<usize>,
    task_waker: Waker,
    ready: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
}

fn native_inline_fixture() -> SettlementFixture {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        max_global_queue_size: 8,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let ready = Arc::new(AtomicBool::new(false));
    let waker = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let (parked_sender, parked) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_async(InlineCompletion {
            ready: Arc::clone(&ready),
            waker: Arc::clone(&waker),
            parked: Some(parked_sender),
            drops: Arc::clone(&drops),
        })
        .unwrap();
    parked
        .recv_timeout(EVENT_DEADLINE)
        .expect("future parks before the event deadline");
    let task_waker = waker
        .lock()
        .unwrap()
        .take()
        .expect("future publishes its task waker");
    let (worker_release, worker_gate) = gate_single_worker::<SyncTask>(&executor);
    for _ in 0..4096 {
        let admitted = executor
            .scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, |_| {});
        if let Err(error) = admitted {
            assert!(matches!(error, ExecutorError::ResourceExhausted(_)));
            return SettlementFixture {
                worker_release,
                worker_gate,
                executor,
                handle,
                task_waker,
                ready,
                drops,
            };
        }
    }
    panic!("the bounded native injector must reject before 4096 admissions");
}

#[test]
fn external_inline_callback_keeps_settlement_owner_until_release() {
    let fixture = native_inline_fixture();
    let id = fixture.handle.id();
    let (observed_sender, observed) = mpsc::sync_channel(1);
    let (entered_sender, entered) = mpsc::sync_channel(1);
    let (release_sender, release_receiver) = mpsc::sync_channel(0);
    let callback = Waker::from(Arc::new(SettlementCallback {
        handle: Mutex::new(Some(fixture.handle)),
        observed: observed_sender,
        entered: entered_sender,
        exit: CallbackExit::Release(Mutex::new(release_receiver)),
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(fixture.executor.wait_for_task(id, None));
    assert!(wait.as_mut().poll(&mut context).is_pending());

    fixture.ready.store(true, Ordering::Release);
    let stale_waker = fixture.task_waker.clone();
    let waking = std::thread::spawn(move || fixture.task_waker.wake());
    assert_eq!(
        observed.recv_timeout(EVENT_DEADLINE).unwrap(),
        Some(Ok(1789))
    );
    entered.recv_timeout(EVENT_DEADLINE).unwrap();
    assert!(
        fixture
            .executor
            .async_tasks
            .close_and_snapshot()
            .contains_key(&id.0)
    );
    release_sender.send(()).unwrap();
    waking.join().unwrap();
    stale_waker.wake_by_ref();
    assert!(fixture.executor.async_tasks.close_and_snapshot().is_empty());
    assert_eq!(fixture.drops.load(Ordering::Acquire), 1);
    fixture.worker_release.send(()).unwrap();
    assert_eq!(fixture.worker_gate.join(), Some(Ok(())));
    ExecutorControl::shutdown(&fixture.executor);
}

#[test]
fn panicking_inline_callback_retains_owner_through_payload_cleanup() {
    let fixture = native_inline_fixture();
    let id = fixture.handle.id();
    let (observed_sender, observed) = mpsc::sync_channel(1);
    let (entered_sender, entered) = mpsc::sync_channel(1);
    let (payload_entered_sender, payload_entered) = mpsc::sync_channel(1);
    let (payload_release, payload_release_receiver) = mpsc::channel();
    let payload_drops = Arc::new(AtomicUsize::new(0));
    let callback = Waker::from(Arc::new(SettlementCallback {
        handle: Mutex::new(Some(fixture.handle)),
        observed: observed_sender,
        entered: entered_sender,
        exit: CallbackExit::Panic(Mutex::new(Some(GatedPanicPayload {
            entered: payload_entered_sender,
            release: payload_release_receiver,
            drops: Arc::clone(&payload_drops),
        }))),
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(fixture.executor.wait_for_task(id, None));
    assert!(wait.as_mut().poll(&mut context).is_pending());

    fixture.ready.store(true, Ordering::Release);
    let stale_waker = fixture.task_waker.clone();
    let waking = std::thread::spawn(move || fixture.task_waker.wake());
    let observed_result = observed.recv_timeout(EVENT_DEADLINE);
    let callback_entry = entered.recv_timeout(EVENT_DEADLINE);
    let payload_drop_entry = payload_entered.recv_timeout(EVENT_DEADLINE);
    let retained_during_payload_drop = fixture
        .executor
        .async_tasks
        .close_and_snapshot()
        .contains_key(&id.0);
    let payload_release_result = payload_release.send(());
    let external_wake_result = waking.join().map_err(drop);
    stale_waker.wake_by_ref();
    let retired_after_cleanup = fixture.executor.async_tasks.close_and_snapshot().is_empty();
    let future_drop_count = fixture.drops.load(Ordering::Acquire);
    let payload_drop_count = payload_drops.load(Ordering::Acquire);
    let publication_panics = fixture
        .executor
        .metrics()
        .publication_panics
        .load(Ordering::Acquire);
    let worker_release_result = fixture.worker_release.send(());
    let worker_result = fixture.worker_gate.join();
    ExecutorControl::shutdown(&fixture.executor);

    assert_eq!(observed_result, Ok(Some(Ok(1789))));
    assert_eq!(callback_entry, Ok(()));
    assert_eq!(payload_drop_entry, Ok(()));
    assert!(retained_during_payload_drop);
    assert_eq!(payload_release_result, Ok(()));
    assert_eq!(external_wake_result, Ok(()));
    assert!(retired_after_cleanup);
    assert_eq!(payload_drop_count, 1);
    assert_eq!(future_drop_count, 1);
    assert_eq!(publication_panics, 1);
    assert_eq!(worker_release_result, Ok(()));
    assert_eq!(worker_result, Some(Ok(())));
}

#[test]
fn shared_completion_waits_for_external_settlement_after_scheduler_drain() {
    let fixture = native_inline_fixture();
    let id = fixture.handle.id();
    let executor = Arc::new(fixture.executor);
    let (observed_sender, observed) = mpsc::sync_channel(1);
    let (entered_sender, entered) = mpsc::sync_channel(1);
    let (release_sender, release_receiver) = mpsc::sync_channel(0);
    let callback = Waker::from(Arc::new(SettlementCallback {
        handle: Mutex::new(Some(fixture.handle)),
        observed: observed_sender,
        entered: entered_sender,
        exit: CallbackExit::Release(Mutex::new(release_receiver)),
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(executor.wait_for_task(id, None));
    assert!(wait.as_mut().poll(&mut context).is_pending());

    fixture.ready.store(true, Ordering::Release);
    let stale_waker = fixture.task_waker.clone();
    let waking = std::thread::spawn(move || fixture.task_waker.wake());
    assert_eq!(observed.recv_timeout(EVENT_DEADLINE), Ok(Some(Ok(1789))));
    entered
        .recv_timeout(EVENT_DEADLINE)
        .expect("external settlement callback enters");
    fixture
        .worker_release
        .send(())
        .expect("worker gate remains connected");
    assert_eq!(fixture.worker_gate.join(), Some(Ok(())));
    executor.join().expect("scheduler drains accepted work");
    assert!(!executor.has_work());

    let shutdown_executor = Arc::clone(&executor);
    let (returned_sender, returned) = mpsc::sync_channel(1);
    let shutdown = std::thread::spawn(move || {
        ExecutorControl::shutdown(&*shutdown_executor);
        returned_sender
            .send(())
            .expect("shutdown observer remains connected");
    });
    assert!(executor.shutdown.wait_for_drain(0, false, EVENT_DEADLINE));
    assert!(
        executor
            .async_tasks
            .close_and_snapshot()
            .contains_key(&id.0)
    );
    assert_eq!(returned.try_recv(), Err(mpsc::TryRecvError::Empty));

    release_sender
        .send(())
        .expect("external settlement callback remains connected");
    waking
        .join()
        .expect("external wake does not propagate panic");
    returned
        .recv_timeout(EVENT_DEADLINE)
        .expect("shutdown returns after settlement registration retires");
    shutdown.join().expect("shutdown caller does not panic");
    stale_waker.wake_by_ref();
    assert!(executor.async_tasks.close_and_snapshot().is_empty());
    assert_eq!(fixture.drops.load(Ordering::Acquire), 1);
}
