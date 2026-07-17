use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::marker::Unpin;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

struct SharedState<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    sender_count: usize,
    closed: bool,
    send_waiters: BTreeMap<u64, Waker>,
    recv_waiters: BTreeMap<u64, Waker>,
    next_send_id: u64,
    next_recv_id: u64,
}

pub struct Sender<T> {
    shared: std::sync::Arc<Mutex<SharedState<T>>>,
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
    pub fn send(&self, value: T) -> SendFuture<'_, T> {
        SendFuture {
            sender: self,
            value: Some(value),
            id: None,
        }
    }

    pub fn try_send(&self, value: T) -> Result<(), T> {
        let mut shared = self.shared.lock().unwrap();
        if shared.closed {
            return Err(value);
        }
        if shared.buffer.len() < shared.capacity {
            shared.buffer.push_back(value);
            if let Some((_, waker)) = shared.recv_waiters.pop_first() {
                waker.wake();
            }
            Ok(())
        } else {
            Err(value)
        }
    }

    pub fn is_closed(&self) -> bool {
        self.shared.lock().unwrap().closed
    }

    pub fn sender_strong_count(&self) -> usize {
        self.shared.lock().unwrap().sender_count
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.sender_count -= 1;
        if shared.sender_count == 0 {
            shared.closed = true;
            let recv_wakers: Vec<_> = std::mem::take(&mut shared.recv_waiters)
                .into_iter()
                .collect();
            drop(shared);
            for (_, waker) in recv_wakers {
                waker.wake();
            }
        }
    }
}

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
            let value = this.value.take().unwrap();
            this.id = None;
            return Poll::Ready(Err(value));
        }

        if shared.buffer.len() < shared.capacity {
            shared.buffer.push_back(this.value.take().unwrap());
            if let Some((_, waker)) = shared.recv_waiters.pop_first() {
                waker.wake();
            }
            this.id = None;
            Poll::Ready(Ok(()))
        } else if let Some(id) = this.id {
            if let Some(waker) = shared.send_waiters.get_mut(&id) {
                *waker = cx.waker().clone();
            }
            Poll::Pending
        } else {
            let id = shared.next_send_id;
            shared.next_send_id += 1;
            this.id = Some(id);
            shared.send_waiters.insert(id, cx.waker().clone());
            Poll::Pending
        }
    }
}

impl<'a, T> Drop for SendFuture<'a, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut shared) = self.sender.shared.lock() {
                shared.send_waiters.remove(&id);
            }
        }
    }
}

pub struct Receiver<T> {
    shared: std::sync::Arc<Mutex<SharedState<T>>>,
}

impl<T> Receiver<T> {
    pub fn recv(&mut self) -> RecvFuture<'_, T> {
        RecvFuture {
            receiver: self,
            id: None,
        }
    }

    pub fn try_recv(&mut self) -> Option<T> {
        let mut shared = self.shared.lock().unwrap();
        let value = shared.buffer.pop_front();
        if value.is_some() {
            if let Some((_, waker)) = shared.send_waiters.pop_first() {
                waker.wake();
            }
        }
        value
    }

    pub fn close(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.closed = true;
        let send_wakers: Vec<_> = std::mem::take(&mut shared.send_waiters)
            .into_iter()
            .collect();
        let recv_wakers: Vec<_> = std::mem::take(&mut shared.recv_waiters)
            .into_iter()
            .collect();
        drop(shared);
        for (_, waker) in send_wakers.into_iter().chain(recv_wakers) {
            waker.wake();
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.closed = true;
        let send_wakers: Vec<_> = std::mem::take(&mut shared.send_waiters)
            .into_iter()
            .collect();
        drop(shared);
        for (_, waker) in send_wakers {
            waker.wake();
        }
    }
}

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
            if let Some((_, waker)) = shared.send_waiters.pop_first() {
                waker.wake();
            }
            this.id = None;
            Poll::Ready(Ok(value))
        } else if shared.closed && shared.buffer.is_empty() {
            this.id = None;
            Poll::Ready(Err(()))
        } else if let Some(id) = this.id {
            if let Some(waker) = shared.recv_waiters.get_mut(&id) {
                *waker = cx.waker().clone();
            }
            Poll::Pending
        } else {
            let id = shared.next_recv_id;
            shared.next_recv_id += 1;
            this.id = Some(id);
            shared.recv_waiters.insert(id, cx.waker().clone());
            Poll::Pending
        }
    }
}

impl<'a, T> Drop for RecvFuture<'a, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut shared) = self.receiver.shared.lock() {
                shared.recv_waiters.remove(&id);
            }
        }
    }
}

pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let shared = std::sync::Arc::new(Mutex::new(SharedState {
        buffer: VecDeque::with_capacity(capacity),
        capacity,
        sender_count: 1,
        closed: false,
        send_waiters: BTreeMap::new(),
        recv_waiters: BTreeMap::new(),
        next_send_id: 0,
        next_recv_id: 0,
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
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWake;
    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    fn poll_future<F: Future + Unpin>(future: &mut F) -> Poll<F::Output> {
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
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
