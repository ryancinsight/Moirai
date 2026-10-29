use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::task::{Context, Poll, Waker};
use std::sync::{Arc, Mutex};
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

    assert_eq!(wake_count.count.load(Ordering::Relaxed), 1);
    assert_eq!(drop_count.load(Ordering::Relaxed), 1);
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
