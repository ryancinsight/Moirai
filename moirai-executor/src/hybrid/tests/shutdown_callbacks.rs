use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
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

use super::{super::HybridExecutor, fixtures::gate_single_worker};
use crate::SyncTask;

struct IdleFuture {
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    parked: Option<mpsc::SyncSender<()>>,
}

impl Future for IdleFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::AcqRel);
        if let Some(parked) = self.parked.take() {
            parked
                .send(())
                .expect("idle-future observer must remain connected");
        }
        Poll::Pending
    }
}

impl Drop for IdleFuture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
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
            .expect("callback observer must remain connected");
        panic!("shutdown completion callback panic");
    }
}

#[test]
fn panicking_waiter_does_not_interrupt_saturated_shutdown_cancellation() {
    let executor = Arc::new(
        HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            max_global_queue_size: 2,
            ..ExecutorConfig::default()
        })
        .unwrap(),
    );
    let polls = Arc::new(AtomicUsize::new(0));
    let first_drops = Arc::new(AtomicUsize::new(0));
    let second_drops = Arc::new(AtomicUsize::new(0));
    let (parked_sender, parked_receiver) = mpsc::sync_channel(2);
    let first = executor
        .spawn_async(IdleFuture {
            polls: Arc::clone(&polls),
            drops: Arc::clone(&first_drops),
            parked: Some(parked_sender.clone()),
        })
        .unwrap();
    let second = executor
        .spawn_async(IdleFuture {
            polls: Arc::clone(&polls),
            drops: Arc::clone(&second_drops),
            parked: Some(parked_sender),
        })
        .unwrap();
    for _ in 0..2 {
        parked_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("both futures must park before the test deadline");
    }

    let (sentinel_sender, sentinel_receiver) = mpsc::sync_channel(1);
    let initial_sentinel = executor
        .spawn_result::<SyncTask, _>(Priority::Low, None, move || {
            sentinel_sender
                .send(())
                .expect("initial sentinel observer must remain connected");
        })
        .unwrap();
    sentinel_receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("initial sentinel must follow both first polls");
    assert_eq!(initial_sentinel.join(), Some(Ok(())));
    assert_eq!(polls.load(Ordering::Acquire), 2);

    let (callback_sender, callback_receiver) = mpsc::sync_channel(1);
    let callback = Waker::from(Arc::new(PanicOnWake {
        invoked: callback_sender,
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(executor.wait_for_task(first.id(), None));
    assert!(wait.as_mut().poll(&mut context).is_pending());

    let (release_worker, gate) = gate_single_worker::<SyncTask>(&executor);
    let first_filler = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    let second_filler = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();

    let (shutdown_sender, shutdown_receiver) = mpsc::sync_channel(1);
    let shutdown_executor = Arc::clone(&executor);
    let shutdown = std::thread::spawn(move || {
        ExecutorControl::shutdown(&*shutdown_executor);
        shutdown_sender
            .send(())
            .expect("shutdown observer must remain connected");
    });

    callback_receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("panicking callback must run during inline cancellation");
    release_worker
        .send(())
        .expect("gate worker must remain connected");
    shutdown_receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("shutdown must finish after releasing the worker");
    shutdown.join().unwrap();

    assert_eq!(first.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(second.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(first_drops.load(Ordering::Acquire), 1);
    assert_eq!(second_drops.load(Ordering::Acquire), 1);
    assert_eq!(gate.join(), Some(Ok(())));
    assert_eq!(first_filler.join(), Some(Ok(())));
    assert_eq!(second_filler.join(), Some(Ok(())));
    assert!(!executor.has_work());
    assert!(executor.async_tasks.close_and_snapshot().is_empty());
    assert_eq!(
        executor
            .metrics()
            .publication_panics
            .load(Ordering::Acquire),
        1
    );
}
