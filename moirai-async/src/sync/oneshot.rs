#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

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

/// Sending half; consumed by the single send.
pub struct Sender<T> {
    shared: std::sync::Arc<Mutex<SharedState<T>>>,
}

impl<T> Sender<T> {
    /// Send the value, completing the channel.
    ///
    /// # Errors
    ///
    /// Returns `Err(value)` when the receiver already closed.
    pub fn send(self, value: T) -> Result<(), T> {
        // The waker leaves the state lock before it is woken: `Waker::wake` may
        // poll the task inline on this thread, and that poll re-locks this
        // state. Same discipline as `mpsc`, `rwlock` and `hybrid::notify`.
        let waker = {
            let mut shared = self.shared.lock().unwrap();
            match shared.state {
                OneshotState::Empty => {
                    shared.state = OneshotState::Value(value);
                    shared.rx_waker.take()
                }
                OneshotState::Closed => return Err(value),
                OneshotState::Value(_) => unreachable!(),
            }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
        Ok(())
    }

    /// Return whether the receiver closed the channel.
    pub fn is_closed(&self) -> bool {
        let shared = self.shared.lock().unwrap();
        matches!(shared.state, OneshotState::Closed)
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let waker = {
            let mut shared = self.shared.lock().unwrap();
            if matches!(shared.state, OneshotState::Empty) {
                shared.state = OneshotState::Closed;
                shared.rx_waker.take()
            } else {
                None
            }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// Receiving half of the single-value channel.
pub struct Receiver<T> {
    shared: std::sync::Arc<Mutex<SharedState<T>>>,
}

impl<T> Receiver<T> {
    /// Receive the value, waiting for the send.
    ///
    /// The returned future resolves `Err(())` when the sender dropped
    /// without sending.
    pub fn recv(&mut self) -> RecvFuture<'_, T> {
        RecvFuture { receiver: self }
    }

    /// Poll for the value, registering `cx`'s waker while it is not yet
    /// sent. Resolves `Err(())` when the sender dropped without sending.
    pub fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Result<T, ()>> {
        let mut shared = self.shared.lock().unwrap();
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

    /// Take the value without waiting; `None` when not yet sent.
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

    /// Close the channel, waking a parked sender.
    pub fn close(&mut self) {
        let waker = {
            let mut shared = self.shared.lock().unwrap();
            shared.state = OneshotState::Closed;
            shared.tx_waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let waker = {
            let mut shared = self.shared.lock().unwrap();
            shared.state = OneshotState::Closed;
            shared.tx_waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// Future returned by [`Receiver::recv`].
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

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.receiver.poll_recv(cx)
    }
}

/// Create a single-value channel.
#[must_use]
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
    use std::task::{Context, Poll, Waker};

    fn poll_future<F: Future + Unpin>(future: &mut F) -> Poll<F::Output> {
        let mut context = Context::from_waker(Waker::noop());
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
