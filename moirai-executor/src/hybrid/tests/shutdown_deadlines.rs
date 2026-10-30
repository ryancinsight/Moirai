use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
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
    task::TaskHandle,
};

use super::super::HybridExecutor;
use crate::SyncTask;

const EVENT_DEADLINE: Duration = Duration::from_secs(5);
const RETURN_DEADLINE: Duration = Duration::from_secs(2);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(20);

struct IdleFuture {
    parked: Option<mpsc::SyncSender<()>>,
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Future for IdleFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::AcqRel);
        if let Some(parked) = self.parked.take() {
            parked
                .send(())
                .expect("idle-future observer remains connected");
        }
        Poll::Pending
    }
}

impl Drop for IdleFuture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
    }
}

struct BlockingCompletion {
    entered: mpsc::SyncSender<()>,
    release: Mutex<mpsc::Receiver<()>>,
    finished: mpsc::SyncSender<()>,
}

impl Wake for BlockingCompletion {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.entered
            .send(())
            .expect("completion-entry observer remains connected");
        self.release
            .lock()
            .unwrap()
            .recv_timeout(EVENT_DEADLINE)
            .expect("test releases the completion callback before its deadline");
        self.finished
            .send(())
            .expect("completion-finish observer remains connected");
    }
}

fn gate_worker(executor: &HybridExecutor) -> (mpsc::SyncSender<()>, TaskHandle<()>) {
    let (started_sender, started_receiver) = mpsc::sync_channel(1);
    let (release_sender, release_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, move || {
            started_sender
                .send(())
                .expect("gate-start observer remains connected");
            release_receiver
                .recv_timeout(EVENT_DEADLINE)
                .expect("test releases the worker gate before its deadline");
        })
        .unwrap();
    started_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("worker gate starts before the event deadline");
    (release_sender, handle)
}

#[test]
fn shutdown_timeout_bounds_inline_cancellation_callbacks() {
    let executor = Arc::new(
        HybridExecutor::new(ExecutorConfig {
            worker_threads: 1,
            max_global_queue_size: 2,
            ..ExecutorConfig::default()
        })
        .unwrap(),
    );
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let (parked_sender, parked_receiver) = mpsc::sync_channel(1);
    let handle = executor
        .spawn_async(IdleFuture {
            parked: Some(parked_sender),
            polls: Arc::clone(&polls),
            drops: Arc::clone(&drops),
        })
        .unwrap();
    parked_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("idle future parks before the event deadline");
    assert_eq!(polls.load(Ordering::Acquire), 1);

    let (callback_entered_sender, callback_entered_receiver) = mpsc::sync_channel(1);
    let (callback_release_sender, callback_release_receiver) = mpsc::sync_channel(1);
    let (callback_finished_sender, callback_finished_receiver) = mpsc::sync_channel(1);
    let callback = Waker::from(Arc::new(BlockingCompletion {
        entered: callback_entered_sender,
        release: Mutex::new(callback_release_receiver),
        finished: callback_finished_sender,
    }));
    let mut context = Context::from_waker(&callback);
    let mut wait = std::pin::pin!(executor.wait_for_task(handle.id(), None));
    assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));

    let (release_worker, gate) = gate_worker(&executor);
    let first_filler = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    let second_filler = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    assert_eq!(executor.pending_tasks(), 2);

    let (timeout_returned_sender, timeout_returned_receiver) = mpsc::sync_channel(1);
    let timeout_executor = Arc::clone(&executor);
    let timeout_caller = std::thread::spawn(move || {
        timeout_executor.shutdown_timeout(SHUTDOWN_TIMEOUT);
        timeout_returned_sender
            .send(())
            .expect("timeout-return observer remains connected");
    });

    callback_entered_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("shutdown cancellation enters the completion callback");
    timeout_returned_receiver
        .recv_timeout(RETURN_DEADLINE)
        .expect("shutdown_timeout returns while its helper remains blocked");
    assert_eq!(
        callback_finished_receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    );
    assert!(handle.is_finished());
    timeout_caller.join().unwrap();

    callback_release_sender
        .send(())
        .expect("blocked completion callback remains connected");
    release_worker
        .send(())
        .expect("worker gate remains connected");
    callback_finished_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("completion callback finishes after release");
    assert!(matches!(
        wait.as_mut().poll(&mut context),
        Poll::Ready(Ok(()))
    ));
    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(drops.load(Ordering::Acquire), 1);

    let (shutdown_finished_sender, shutdown_finished_receiver) = mpsc::sync_channel(1);
    let shutdown_executor = Arc::clone(&executor);
    let full_shutdown = std::thread::spawn(move || {
        ExecutorControl::shutdown(&*shutdown_executor);
        shutdown_finished_sender
            .send(())
            .expect("full-shutdown observer remains connected");
    });
    shutdown_finished_receiver
        .recv_timeout(EVENT_DEADLINE)
        .expect("full shutdown joins the background drain");
    full_shutdown.join().unwrap();

    assert_eq!(gate.join(), Some(Ok(())));
    assert_eq!(first_filler.join(), Some(Ok(())));
    assert_eq!(second_filler.join(), Some(Ok(())));
    assert!(!executor.has_work());
}
