#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

//! Bounded async multi-producer single-consumer channel.
//!
//! Waiter bookkeeping is delegated to the shared `WaitQueue`: the same
//! FIFO-by-monotonic-id registration, grant hand-off, and O(log n)
//! cancellation that `Notify`, `Semaphore`, and `RwLock` use. This module
//! keeps only the channel's own admission predicate (buffer capacity) and
//! its two grants — a send frees a receive slot, a receive frees a send slot.

use std::collections::VecDeque;
use std::future::Future;
use std::marker::Unpin;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use super::wait_queue::{WaitQueue, WaiterPoll};

struct SharedState<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    sender_count: usize,
    closed: bool,
    send_waiters: WaitQueue<()>,
    recv_waiters: WaitQueue<()>,
}

/// Sending half of the bounded channel; clone to add producers.
pub struct Sender<T> {
    shared: Arc<Mutex<SharedState<T>>>,
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        let mut shared = self.shared.lock().unwrap();
        shared.sender_count += 1;
        Sender {
            shared: self.shared.clone(),
        }
    }
}

impl<T> Sender<T> {
    /// Send a value, waiting for buffer capacity.
    ///
    /// The returned future resolves `Err(value)` when the channel closes
    /// before the value is accepted.
    pub fn send(&self, value: T) -> SendFuture<'_, T> {
        SendFuture {
            sender: self,
            value: Some(value),
            id: None,
        }
    }

    /// Send without waiting; returns the value when full or closed.
    ///
    /// # Errors
    ///
    /// Returns `Err(value)` when the buffer is at capacity or the channel
    /// is closed.
    pub fn try_send(&self, value: T) -> Result<(), T> {
        let mut shared = self.shared.lock().unwrap();
        if shared.closed {
            return Err(value);
        }
        if shared.buffer.len() >= shared.capacity {
            return Err(value);
        }
        shared.buffer.push_back(value);
        // Grant the oldest parked receiver a slot and wake it outside the
        // lock: a task waker may re-enter the channel, so holding the mutex
        // across `wake` risks a self-deadlock.
        let waker = shared.recv_waiters.grant_oldest(());
        drop(shared);
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }

    /// Return whether the channel is closed.
    pub fn is_closed(&self) -> bool {
        self.shared.lock().unwrap().closed
    }

    /// Count of live sender handles.
    pub fn sender_strong_count(&self) -> usize {
        self.shared.lock().unwrap().sender_count
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.sender_count -= 1;
        if shared.sender_count != 0 {
            return;
        }
        shared.closed = true;
        let wakers = shared.recv_waiters.grant_all(());
        drop(shared);
        for waker in wakers {
            waker.wake();
        }
    }
}

/// Future returned by [`Sender::send`].
pub struct SendFuture<'a, T> {
    sender: &'a Sender<T>,
    value: Option<T>,
    id: Option<u64>,
}

impl<'a, T: Unpin> Future for SendFuture<'a, T> {
    type Output = Result<(), T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut shared = this.sender.shared.lock().unwrap();

        if shared.closed {
            if let Some(id) = this.id.take() {
                shared.send_waiters.deregister(id);
            }
            let value = this.value.take().unwrap();
            return Poll::Ready(Err(value));
        }

        if shared.buffer.len() < shared.capacity {
            if let Some(id) = this.id.take() {
                shared.send_waiters.deregister(id);
            }
            shared.buffer.push_back(this.value.take().unwrap());
            let waker = shared.recv_waiters.grant_oldest(());
            drop(shared);
            if let Some(waker) = waker {
                waker.wake();
            }
            return Poll::Ready(Ok(()));
        }

        // Still full: refresh the existing registration, or join the queue.
        // A stale grant (the slot was taken by another sender) re-registers
        // behind the current waiters rather than losing its place.
        this.id = Some(match this.id {
            Some(id) => match shared.send_waiters.poll_waiter(id, cx.waker()) {
                WaiterPoll::Pending => id,
                WaiterPoll::Granted(()) | WaiterPoll::NotRegistered => {
                    shared.send_waiters.register(cx.waker().clone())
                }
            },
            None => shared.send_waiters.register(cx.waker().clone()),
        });
        Poll::Pending
    }
}

impl<'a, T> Drop for SendFuture<'a, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id
            && let Ok(mut shared) = self.sender.shared.lock()
        {
            shared.send_waiters.deregister(id);
        }
    }
}

/// Receiving half of the bounded channel.
pub struct Receiver<T> {
    shared: Arc<Mutex<SharedState<T>>>,
}

impl<T> Receiver<T> {
    /// Receive the next value, waiting for one to arrive.
    ///
    /// The returned future resolves `Err(())` when the channel is closed
    /// and drained.
    pub fn recv(&mut self) -> RecvFuture<'_, T> {
        RecvFuture {
            receiver: self,
            id: None,
        }
    }

    /// Receive without waiting; `None` when the buffer is empty.
    pub fn try_recv(&mut self) -> Option<T> {
        let mut shared = self.shared.lock().unwrap();
        let value = shared.buffer.pop_front();
        let waker = if value.is_some() {
            shared.send_waiters.grant_oldest(())
        } else {
            None
        };
        drop(shared);
        if let Some(waker) = waker {
            waker.wake();
        }
        value
    }

    /// Close the channel, waking every parked sender and receiver.
    pub fn close(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.closed = true;
        let mut wakers = shared.send_waiters.grant_all(());
        wakers.extend(shared.recv_waiters.grant_all(()));
        drop(shared);
        for waker in wakers {
            waker.wake();
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.closed = true;
        let wakers = shared.send_waiters.grant_all(());
        drop(shared);
        for waker in wakers {
            waker.wake();
        }
    }
}

/// Future returned by [`Receiver::recv`].
pub struct RecvFuture<'a, T> {
    receiver: &'a mut Receiver<T>,
    id: Option<u64>,
}

impl<'a, T> Future for RecvFuture<'a, T> {
    type Output = Result<T, ()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut shared = this.receiver.shared.lock().unwrap();
        if let Some(value) = shared.buffer.pop_front() {
            if let Some(id) = this.id.take() {
                shared.recv_waiters.deregister(id);
            }
            let waker = shared.send_waiters.grant_oldest(());
            drop(shared);
            if let Some(waker) = waker {
                waker.wake();
            }
            return Poll::Ready(Ok(value));
        }
        if shared.closed {
            if let Some(id) = this.id.take() {
                shared.recv_waiters.deregister(id);
            }
            return Poll::Ready(Err(()));
        }

        this.id = Some(match this.id {
            Some(id) => match shared.recv_waiters.poll_waiter(id, cx.waker()) {
                WaiterPoll::Pending => id,
                WaiterPoll::Granted(()) | WaiterPoll::NotRegistered => {
                    shared.recv_waiters.register(cx.waker().clone())
                }
            },
            None => shared.recv_waiters.register(cx.waker().clone()),
        });
        Poll::Pending
    }
}

impl<'a, T> Drop for RecvFuture<'a, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id
            && let Ok(mut shared) = self.receiver.shared.lock()
        {
            shared.recv_waiters.deregister(id);
        }
    }
}

/// Create a bounded channel with the given buffer capacity.
#[must_use]
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Mutex::new(SharedState {
        buffer: VecDeque::with_capacity(capacity),
        capacity,
        sender_count: 1,
        closed: false,
        send_waiters: WaitQueue::new(),
        recv_waiters: WaitQueue::new(),
    }));
    (
        Sender {
            shared: shared.clone(),
        },
        Receiver { shared },
    )
}

#[cfg(test)]
mod tests {
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
        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();
        tx.try_send(3).unwrap();
        assert_eq!(rx.try_recv(), Some(1));
        assert_eq!(rx.try_recv(), Some(2));
        assert_eq!(rx.try_recv(), Some(3));
        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn test_mpsc_closed_sender() {
        let (tx, mut rx) = channel::<i32>(10);
        tx.try_send(1).unwrap();
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
        tx1.try_send(1).unwrap();
        tx2.try_send(2).unwrap();
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
        tx.try_send(1).unwrap();
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
        tx.try_send(42).unwrap();
        assert!(matches!(poll_future(&mut recv), Poll::Ready(Ok(42))));
    }

    #[test]
    fn test_mpsc_send_future_dropped_cancels_waiter() {
        let (tx, _rx) = channel(1);
        tx.try_send(1).unwrap();
        // Send future goes pending, then is dropped without completing
        let mut send = tx.send(2);
        assert!(matches!(poll_future(&mut send), Poll::Pending));
        drop(send);
        // The full send_waiters queue must be empty after the drop
        assert!(tx.shared.lock().unwrap().send_waiters.is_empty());
    }

    #[test]
    fn test_mpsc_recv_future_dropped_cancels_waiter() {
        let (tx, mut rx) = channel(1);
        tx.try_send(1).unwrap();
        // Consume the item, then recv goes pending waiting for next item
        let _ = rx.try_recv();
        let mut recv = rx.recv();
        assert!(matches!(poll_future(&mut recv), Poll::Pending));
        drop(recv);
        // The full recv_waiters queue must be empty after the drop
        assert!(tx.shared.lock().unwrap().recv_waiters.is_empty());
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
}
