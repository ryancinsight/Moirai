use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::task::{Context, Poll, Waker};
use std::sync::{Arc, Mutex, mpsc};
use std::task::Wake;

use super::{TaskHandle, TaskId, TaskResultSender};
use crate::error::TaskError;

#[derive(Default)]
struct WakeCount {
    count: AtomicUsize,
}

impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

fn counting_waker() -> (Waker, Arc<WakeCount>) {
    let count = Arc::new(WakeCount::default());
    (Waker::from(Arc::clone(&count)), count)
}

fn poll_handle<T>(handle: &mut TaskHandle<T>, waker: &Waker) -> Poll<Option<Result<T, TaskError>>> {
    let mut context = Context::from_waker(waker);
    Pin::new(handle).poll(&mut context)
}

#[test]
fn ready_and_detached_handles_resolve_when_awaited() {
    let (waker, _) = counting_waker();
    let mut ready = TaskHandle::ready(TaskId::new(1), Ok(42usize));
    let mut detached = TaskHandle::<usize>::new_detached(TaskId::new(2));

    assert_eq!(poll_handle(&mut ready, &waker), Poll::Ready(Some(Ok(42))));
    assert_eq!(poll_handle(&mut detached, &waker), Poll::Ready(None));
    assert!(!detached.is_finished());
}

#[test]
fn borrowed_await_keeps_completion_visible_after_consumption() {
    let (waker, _) = counting_waker();
    let mut handle = TaskHandle::ready(TaskId::new(9), Ok(168usize));

    assert_eq!(poll_handle(&mut handle, &waker), Poll::Ready(Some(Ok(168))));
    assert!(handle.is_finished());
    assert_eq!(poll_handle(&mut handle, &waker), Poll::Ready(None));
    assert_eq!(handle.join(), None);
}

#[test]
fn pending_handle_wakes_and_returns_the_sent_value() {
    let (mut handle, sender) = TaskHandle::new_pending(TaskId::new(3));
    let (waker, wake_count) = counting_waker();

    assert_eq!(poll_handle(&mut handle, &waker), Poll::Pending);
    sender.send(Ok(84usize));

    assert_eq!(wake_count.count.load(Ordering::Relaxed), 1);
    assert_eq!(poll_handle(&mut handle, &waker), Poll::Ready(Some(Ok(84))));
}

#[test]
fn repeated_poll_replaces_the_registered_waker() {
    let (mut handle, sender) = TaskHandle::new_pending(TaskId::new(4));
    let (first_waker, first_count) = counting_waker();
    let (second_waker, second_count) = counting_waker();

    assert_eq!(poll_handle(&mut handle, &first_waker), Poll::Pending);
    assert_eq!(poll_handle(&mut handle, &second_waker), Poll::Pending);
    sender.send(Ok(126usize));

    assert_eq!(first_count.count.load(Ordering::Relaxed), 0);
    assert_eq!(second_count.count.load(Ordering::Relaxed), 1);
    assert_eq!(
        poll_handle(&mut handle, &second_waker),
        Poll::Ready(Some(Ok(126)))
    );
}

#[test]
fn sender_drop_wakes_awaiting_handle_with_cancellation() {
    let (mut handle, sender) = TaskHandle::<usize>::new_pending(TaskId::new(5));
    let (waker, wake_count) = counting_waker();

    assert_eq!(poll_handle(&mut handle, &waker), Poll::Pending);
    drop(sender);

    assert_eq!(wake_count.count.load(Ordering::Relaxed), 1);
    assert_eq!(
        poll_handle(&mut handle, &waker),
        Poll::Ready(Some(Err(TaskError::Cancelled)))
    );
}

struct DropCount(Arc<AtomicUsize>);

impl Drop for DropCount {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn dropping_pending_handle_leaves_sender_completion_safe() {
    let (mut handle, sender) = TaskHandle::new_pending(TaskId::new(6));
    let (waker, wake_count) = counting_waker();
    let drop_count = Arc::new(AtomicUsize::new(0));

    assert!(matches!(poll_handle(&mut handle, &waker), Poll::Pending));
    drop(handle);
    sender.send(Ok(DropCount(Arc::clone(&drop_count))));

    assert_eq!(wake_count.count.load(Ordering::Relaxed), 0);
    assert_eq!(drop_count.load(Ordering::Relaxed), 1);
}

#[test]
fn dropping_an_awaited_handle_releases_its_waiter() {
    let (mut handle, sender) = TaskHandle::<usize>::new_pending(TaskId::new(91));
    let (waker, count) = counting_waker();
    assert_eq!(poll_handle(&mut handle, &waker), Poll::Pending);
    drop(waker);
    drop(handle);
    assert_eq!(Arc::strong_count(&count), 1);
    sender.send(Ok(37));
    assert_eq!(count.count.load(Ordering::Relaxed), 0);
}

#[test]
fn sender_drop_wake_panic_preserves_cancelled_result() {
    struct PanicWake(Arc<AtomicUsize>);
    impl Wake for PanicWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
            panic!("sender drop wake");
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let waker = Waker::from(Arc::new(PanicWake(Arc::clone(&calls))));
    let (mut handle, sender) = TaskHandle::<usize>::new_pending(TaskId::new(92));
    assert_eq!(poll_handle(&mut handle, &waker), Poll::Pending);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(sender)))
        .expect_err("wake callback must panic");
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"sender drop wake"));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        poll_handle(&mut handle, Waker::noop()),
        Poll::Ready(Some(Err(TaskError::Cancelled)))
    );
    assert!(handle.is_finished());
    assert_eq!(poll_handle(&mut handle, Waker::noop()), Poll::Ready(None));
}

#[test]
fn panicking_raw_waker_drop_does_not_strand_the_slot() {
    struct PanicOnDrop {
        panic_drops: Arc<AtomicUsize>,
        wake_calls: Arc<AtomicUsize>,
    }
    impl Wake for PanicOnDrop {
        fn wake(self: Arc<Self>) {
            self.wake_calls.fetch_add(1, Ordering::Relaxed);
        }
    }
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            self.panic_drops.fetch_add(1, Ordering::Relaxed);
            panic!("raw waker drop");
        }
    }

    let panic_drops = Arc::new(AtomicUsize::new(0));
    let wake_calls = Arc::new(AtomicUsize::new(0));
    let owner = Arc::new(PanicOnDrop {
        panic_drops: Arc::clone(&panic_drops),
        wake_calls: Arc::clone(&wake_calls),
    });
    let waker = Waker::from(Arc::clone(&owner));
    let (mut handle, sender) = TaskHandle::<DropCount>::new_pending(TaskId::new(93));
    assert!(matches!(poll_handle(&mut handle, &waker), Poll::Pending));
    drop(owner);
    drop(waker);

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(handle)))
        .expect_err("raw waker drop must panic");
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"raw waker drop"));
    assert_eq!(panic_drops.load(Ordering::Relaxed), 1);
    assert_eq!(wake_calls.load(Ordering::Relaxed), 0);
    let result_drops = Arc::new(AtomicUsize::new(0));
    sender.send(Ok(DropCount(Arc::clone(&result_drops))));
    assert_eq!(result_drops.load(Ordering::Relaxed), 1);
}

#[test]
fn producer_owned_wake_may_finish_after_handle_drop() {
    struct GatedWake {
        calls: Arc<AtomicUsize>,
        entered: mpsc::SyncSender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl Wake for GatedWake {
        fn wake(self: Arc<Self>) {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.entered
                .send(())
                .expect("invariant: test receives the wake entry");
            self.release
                .lock()
                .expect("invariant: release mutex remains available")
                .recv()
                .expect("invariant: test releases the callback");
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let result_drops = Arc::new(AtomicUsize::new(0));
    let (entered_tx, entered_rx) = mpsc::sync_channel(0);
    let (release_tx, release_rx) = mpsc::sync_channel(0);
    let waker = Waker::from(Arc::new(GatedWake {
        calls: Arc::clone(&calls),
        entered: entered_tx,
        release: Mutex::new(release_rx),
    }));
    let (mut handle, sender) = TaskHandle::<DropCount>::new_pending(TaskId::new(94));
    assert!(matches!(poll_handle(&mut handle, &waker), Poll::Pending));
    let producer_drops = Arc::clone(&result_drops);
    let producer = std::thread::spawn(move || {
        sender.send(Ok(DropCount(producer_drops)));
    });

    entered_rx
        .recv()
        .expect("invariant: producer enters the wake callback");
    drop(handle);
    release_tx
        .send(())
        .expect("invariant: callback remains available for release");
    producer.join().expect("invariant: producer completes");
    drop(waker);

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(result_drops.load(Ordering::Relaxed), 1);
}

struct CancelOnDrop {
    sender: Mutex<Option<TaskResultSender<usize>>>,
    wake_count: Arc<AtomicUsize>,
}

impl Wake for CancelOnDrop {
    fn wake(self: Arc<Self>) {
        self.wake_count.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let sender = self
            .sender
            .get_mut()
            .expect("invariant: the sender mutex is never locked")
            .take();
        drop(sender);
    }
}

#[test]
fn repoll_publishes_new_waker_before_dropping_the_old_one() {
    let (mut handle, sender) = TaskHandle::new_pending(TaskId::new(7));
    let wake_count = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(CancelOnDrop {
        sender: Mutex::new(Some(sender)),
        wake_count: Arc::clone(&wake_count),
    });
    let waker = Waker::from(Arc::clone(&cancel));

    assert_eq!(poll_handle(&mut handle, &waker), Poll::Pending);
    drop(cancel);
    drop(waker);

    assert_eq!(
        poll_handle(&mut handle, Waker::noop()),
        Poll::Ready(Some(Err(TaskError::Cancelled)))
    );
    assert_eq!(wake_count.load(Ordering::Relaxed), 0);
}

#[test]
fn blocking_join_publishes_thread_waiter_before_dropping_waker() {
    let (mut handle, sender) = TaskHandle::new_pending(TaskId::new(8));
    let wake_count = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(CancelOnDrop {
        sender: Mutex::new(Some(sender)),
        wake_count: Arc::clone(&wake_count),
    });
    let waker = Waker::from(Arc::clone(&cancel));

    assert_eq!(poll_handle(&mut handle, &waker), Poll::Pending);
    drop(cancel);
    drop(waker);

    assert_eq!(handle.join(), Some(Err(TaskError::Cancelled)));
    assert_eq!(wake_count.load(Ordering::Relaxed), 0);
}
