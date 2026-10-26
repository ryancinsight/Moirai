use super::*;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::task::{Context, Poll, Wake, Waker};

fn poll_future<F: Future + Unpin>(future: &mut F) -> Poll<F::Output> {
    let mut context = Context::from_waker(Waker::noop());
    Pin::new(future).poll(&mut context)
}

fn poll_future_with_waker<F: Future + Unpin>(future: &mut F, waker: &Waker) -> Poll<F::Output> {
    let mut context = Context::from_waker(waker);
    Pin::new(future).poll(&mut context)
}

struct CountingWake(Arc<AtomicUsize>);

impl Wake for CountingWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Release);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Release);
    }
}

#[test]
fn test_mpsc_send_recv() {
    let (tx, mut rx) = channel(10);
    assert_eq!(tx.try_send(1), Ok(()));
    assert_eq!(tx.try_send(2), Ok(()));
    assert_eq!(tx.try_send(3), Ok(()));
    assert_eq!(rx.try_recv(), Some(1));
    assert_eq!(rx.try_recv(), Some(2));
    assert_eq!(rx.try_recv(), Some(3));
    assert!(rx.try_recv().is_none());
}

#[test]
fn test_mpsc_closed_sender() {
    let (tx, mut rx) = channel::<i32>(10);
    assert_eq!(tx.try_send(1), Ok(()));
    drop(tx);
    assert_eq!(rx.try_recv(), Some(1));
    assert!(rx.try_recv().is_none());
}

#[test]
fn test_mpsc_closed_receiver() {
    let (tx, rx) = channel::<i32>(10);
    drop(rx);
    assert!(tx.try_send(1).is_err());
}

#[test]
fn test_mpsc_capacity() {
    let (tx, mut rx) = channel(2);
    assert!(tx.try_send(1).is_ok());
    assert!(tx.try_send(2).is_ok());
    assert!(tx.try_send(3).is_err());
    let _ = rx.try_recv();
    assert!(tx.try_send(3).is_ok());
}

#[test]
fn test_mpsc_sender_clone() {
    let (tx1, mut rx) = channel(10);
    let tx2 = tx1.clone();
    assert_eq!(tx1.try_send(1), Ok(()));
    assert_eq!(tx2.try_send(2), Ok(()));
    drop(tx1);
    drop(tx2);
    assert_eq!(rx.try_recv(), Some(1));
    assert_eq!(rx.try_recv(), Some(2));
    assert!(rx.try_recv().is_none());
}

#[test]
fn test_mpsc_sender_strong_count() {
    let (tx1, _) = channel::<i32>(10);
    assert_eq!(tx1.sender_strong_count(), 1);
    let tx2 = tx1.clone();
    assert_eq!(tx1.sender_strong_count(), 2);
    drop(tx2);
    assert_eq!(tx1.sender_strong_count(), 1);
}

#[test]
fn test_mpsc_send_pending_then_recv() {
    let (tx, mut rx) = channel(1);
    assert_eq!(tx.try_send(1), Ok(()));
    let mut send = tx.send(2);
    assert!(matches!(poll_future(&mut send), Poll::Pending));
    let _ = rx.try_recv();
    assert!(matches!(poll_future(&mut send), Poll::Ready(Ok(()))));
}

#[test]
fn test_mpsc_async_recv_pending_then_send() {
    let (tx, mut rx) = channel(1);
    let mut recv = rx.recv();
    assert!(matches!(poll_future(&mut recv), Poll::Pending));
    assert_eq!(tx.try_send(42), Ok(()));
    assert!(matches!(poll_future(&mut recv), Poll::Ready(Ok(42))));
}

#[test]
fn test_mpsc_send_future_dropped_cancels_waiter() {
    let (tx, _rx) = channel(1);
    assert_eq!(tx.try_send(1), Ok(()));
    // Send future goes pending, then is dropped without completing
    let mut send = tx.send(2);
    assert!(matches!(poll_future(&mut send), Poll::Pending));
    drop(send);
    // The full send_waiters queue must be empty after the drop
    match tx.shared.lock() {
        Ok(shared) => assert!(shared.send_waiters.is_empty()),
        Err(poisoned) => panic!("channel state poisoned during waiter cancellation: {poisoned}"),
    }
}

#[test]
fn test_mpsc_recv_future_dropped_cancels_waiter() {
    let (tx, mut rx) = channel(1);
    assert_eq!(tx.try_send(1), Ok(()));
    // Consume the item, then recv goes pending waiting for next item
    let _ = rx.try_recv();
    let mut recv = rx.recv();
    assert!(matches!(poll_future(&mut recv), Poll::Pending));
    drop(recv);
    // The full recv_waiters queue must be empty after the drop
    match tx.shared.lock() {
        Ok(shared) => assert!(shared.recv_waiters.is_empty()),
        Err(poisoned) => panic!("channel state poisoned during waiter cancellation: {poisoned}"),
    }
}

#[test]
fn oldest_pending_sender_is_woken_first() {
    let (tx, mut rx) = channel(1);
    tx.try_send(1).expect("initial send must fill the channel");

    let first_wakes = Arc::new(AtomicUsize::new(0));
    let second_wakes = Arc::new(AtomicUsize::new(0));
    let first_waker = Waker::from(Arc::new(CountingWake(Arc::clone(&first_wakes))));
    let second_waker = Waker::from(Arc::new(CountingWake(Arc::clone(&second_wakes))));
    let mut first = tx.send(2);
    let mut second = tx.send(3);

    assert!(poll_future_with_waker(&mut first, &first_waker).is_pending());
    assert!(poll_future_with_waker(&mut second, &second_waker).is_pending());

    assert_eq!(rx.try_recv(), Some(1));
    assert_eq!(first_wakes.load(Ordering::Acquire), 1);
    assert_eq!(second_wakes.load(Ordering::Acquire), 0);

    assert!(matches!(
        poll_future_with_waker(&mut first, &first_waker),
        Poll::Ready(Ok(()))
    ));
    assert_eq!(rx.try_recv(), Some(2));
    assert_eq!(second_wakes.load(Ordering::Acquire), 1);
    assert!(matches!(
        poll_future_with_waker(&mut second, &second_waker),
        Poll::Ready(Ok(()))
    ));
    assert_eq!(rx.try_recv(), Some(3));
}
