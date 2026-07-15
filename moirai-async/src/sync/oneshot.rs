use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

enum OneshotState<T> {
    Empty,
    Value(T),
    Closed,
}

struct SharedState<T> {
    state: OneshotState<T>,
    rx_waker: Option<Waker>,
    tx_waker: Option<Waker>,
}

pub struct Sender<T> {
    shared: std::sync::Arc<Mutex<SharedState<T>>>,
}

impl<T> Sender<T> {
    pub fn send(self, value: T) -> Result<(), T> {
        let mut shared = self.shared.lock().unwrap();
        match shared.state {
            OneshotState::Empty => {
                shared.state = OneshotState::Value(value);
                if let Some(waker) = shared.rx_waker.take() {
                    waker.wake();
                }
                Ok(())
            }
            OneshotState::Closed => Err(value),
            OneshotState::Value(_) => unreachable!(),
        }
    }

    pub fn is_closed(&self) -> bool {
        let shared = self.shared.lock().unwrap();
        matches!(shared.state, OneshotState::Closed)
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        if matches!(shared.state, OneshotState::Empty) {
            shared.state = OneshotState::Closed;
            if let Some(waker) = shared.rx_waker.take() {
                waker.wake();
            }
        }
    }
}

pub struct Receiver<T> {
    shared: std::sync::Arc<Mutex<SharedState<T>>>,
}

impl<T> Receiver<T> {
    pub fn recv(&mut self) -> RecvFuture<'_, T> {
        RecvFuture { receiver: self }
    }

    pub fn try_recv(&mut self) -> Option<T> {
        let mut shared = self.shared.lock().unwrap();
        match std::mem::replace(&mut shared.state, OneshotState::Closed) {
            OneshotState::Value(v) => Some(v),
            OneshotState::Empty => {
                shared.state = OneshotState::Empty;
                None
            }
            OneshotState::Closed => None,
        }
    }

    pub fn close(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.state = OneshotState::Closed;
        if let Some(waker) = shared.tx_waker.take() {
            waker.wake();
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.state = OneshotState::Closed;
        if let Some(waker) = shared.tx_waker.take() {
            waker.wake();
        }
    }
}

pub struct RecvFuture<'a, T> {
    receiver: &'a mut Receiver<T>,
}

impl<T> Drop for RecvFuture<'_, T> {
    fn drop(&mut self) {
        if let Ok(mut shared) = self.receiver.shared.lock() {
            shared.rx_waker = None;
        }
    }
}

impl<'a, T> Future for RecvFuture<'a, T> {
    type Output = Result<T, ()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut shared = self.receiver.shared.lock().unwrap();
        match std::mem::replace(&mut shared.state, OneshotState::Closed) {
            OneshotState::Value(v) => Poll::Ready(Ok(v)),
            OneshotState::Closed => Poll::Ready(Err(())),
            OneshotState::Empty => {
                shared.state = OneshotState::Empty;
                shared.rx_waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let shared = std::sync::Arc::new(Mutex::new(SharedState {
        state: OneshotState::Empty,
        rx_waker: None,
        tx_waker: None,
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
    use std::sync::Arc;
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

    #[test]
    fn test_oneshot_send_recv() {
        let (tx, mut rx) = channel();
        tx.send(42).unwrap();
        assert_eq!(rx.try_recv(), Some(42));
        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn test_oneshot_recv_pending_then_ready() {
        let (tx, mut rx) = channel();
        let mut recv = rx.recv();
        assert!(matches!(poll_future(&mut recv), Poll::Pending));
        tx.send(99).unwrap();
        assert!(matches!(poll_future(&mut recv), Poll::Ready(Ok(99))));
    }

    #[test]
    fn test_oneshot_sender_dropped_recv_err() {
        let (tx, mut rx) = channel::<i32>();
        drop(tx);
        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn test_oneshot_recv_closed_err() {
        let (_, mut rx) = channel::<i32>();
        rx.close();
        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn test_oneshot_is_closed() {
        let (tx, mut rx) = channel::<i32>();
        assert!(!tx.is_closed());
        rx.close();
        assert!(tx.is_closed());
    }

    #[test]
    fn test_oneshot_double_send_err() {
        let (tx, rx) = channel();
        drop(rx);
        assert!(tx.send(1).is_err());
    }

    #[test]
    fn test_oneshot_recv_future_ready() {
        let (tx, mut rx) = channel();
        tx.send(7).unwrap();
        let mut recv = rx.recv();
        assert!(matches!(poll_future(&mut recv), Poll::Ready(Ok(7))));
    }
}
