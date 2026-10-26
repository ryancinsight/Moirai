#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

//! Async multi-producer single-consumer channel with explicit capacity modes.
//!
//! Waiter bookkeeping is delegated to the shared `WaitQueue`: the same
//! FIFO-by-monotonic-id registration, grant hand-off, and O(log n)
//! cancellation that `Notify`, `Semaphore`, and `RwLock` use. This module
//! keeps only the channel's own admission predicate (buffer capacity) and
//! its two grants — a send frees a receive slot, a receive frees a send slot.

use std::collections::VecDeque;
use std::future::Future;
use std::marker::{PhantomData, Unpin};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use super::wait_queue::{WaitQueue, WaiterPoll};

struct SharedState<T> {
    buffer: VecDeque<T>,
    capacity: Option<usize>,
    sender_count: usize,
    closed: bool,
    send_waiters: WaitQueue<()>,
    recv_waiters: WaitQueue<()>,
}

/// Bounded channel admission; sends wait when the buffer is full.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bounded;

/// Unbounded channel admission; synchronous sends allocate as needed.
///
/// Memory is bounded by available process memory, so producers must not use
/// this mode as a substitute for application-level resource limits.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unbounded;

/// Sending half of a channel; clone to add producers.
///
/// The default mode preserves bounded [`channel`] semantics. The unbounded
/// constructor returns `Sender<T, Unbounded>` with synchronous `enqueue`.
pub struct Sender<T, Mode = Bounded> {
    shared: Arc<Mutex<SharedState<T>>>,
    mode: PhantomData<fn() -> Mode>,
}

impl<T, Mode> std::fmt::Debug for Sender<T, Mode> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shared = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        formatter
            .debug_struct("Sender")
            .field("sender_count", &shared.sender_count)
            .field("queued", &shared.buffer.len())
            .field("capacity", &shared.capacity)
            .field("closed", &shared.closed)
            .finish_non_exhaustive()
    }
}

impl<T, Mode> Clone for Sender<T, Mode> {
    fn clone(&self) -> Self {
        let mut shared = self.shared.lock().unwrap();
        shared.sender_count += 1;
        Sender {
            shared: self.shared.clone(),
            mode: PhantomData,
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
    /// Returns `Err(value)` when the buffer is at capacity or the channel is closed.
    pub fn try_send(&self, value: T) -> Result<(), T> {
        self.try_enqueue(value)
    }
}

impl<T> Sender<T, Unbounded> {
    /// Enqueue a value synchronously, without waiting for capacity.
    ///
    /// # Errors
    ///
    /// Returns the exact `Err(value)` when the receiver has closed or dropped.
    /// Allocation failure follows the process allocator's normal behavior.
    pub fn enqueue(&self, value: T) -> Result<(), T> {
        self.try_enqueue(value)
    }
}

impl<T, Mode> Sender<T, Mode> {
    /// Send without waiting; returns the value when full or closed.
    ///
    /// # Errors
    ///
    /// Returns `Err(value)` when the buffer is at capacity or the channel
    /// is closed.
    fn try_enqueue(&self, value: T) -> Result<(), T> {
        let mut shared = self.shared.lock().unwrap();
        if shared.closed {
            return Err(value);
        }
        if shared
            .capacity
            .is_some_and(|capacity| shared.buffer.len() >= capacity)
        {
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

impl<T, Mode> Drop for Sender<T, Mode> {
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

/// Future returned by a bounded [`Sender`]'s `send` method.
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

        if shared
            .capacity
            .is_none_or(|capacity| shared.buffer.len() < capacity)
        {
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

/// Receiving half of a channel, with one persistent receive registration.
pub struct Receiver<T, Mode = Bounded> {
    shared: Arc<Mutex<SharedState<T>>>,
    id: Option<u64>,
    mode: PhantomData<fn() -> Mode>,
}

impl<T, Mode> std::fmt::Debug for Receiver<T, Mode> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shared = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        formatter
            .debug_struct("Receiver")
            .field("sender_count", &shared.sender_count)
            .field("queued", &shared.buffer.len())
            .field("capacity", &shared.capacity)
            .field("closed", &shared.closed)
            .finish_non_exhaustive()
    }
}

impl<T, Mode> Receiver<T, Mode> {
    /// Receive the next value, returning `Err(())` after closure and drain.
    ///
    /// Dropping a polled future cancels its waiter without consuming a value.
    pub fn recv(&mut self) -> RecvFuture<'_, T, Mode> {
        RecvFuture {
            receiver: self,
            polled: false,
        }
    }

    /// Poll for the next value, returning `None` only after closure and drain.
    ///
    /// `Pending` retains this receiver's registration until the next poll,
    /// cancellation through a polled [`recv`](Self::recv), or receiver drop.
    /// Re-polling replaces the stored waker with the actual caller's waker.
    pub fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Option<T>> {
        let mut shared = self.shared.lock().unwrap();
        if let Some(value) = shared.buffer.pop_front() {
            if let Some(id) = self.id.take() {
                shared.recv_waiters.deregister(id);
            }
            let waker = shared.send_waiters.grant_oldest(());
            drop(shared);
            if let Some(waker) = waker {
                waker.wake();
            }
            return Poll::Ready(Some(value));
        }
        if shared.closed {
            if let Some(id) = self.id.take() {
                shared.recv_waiters.deregister(id);
            }
            return Poll::Ready(None);
        }
        self.id = Some(match self.id {
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

    /// Receive without waiting; `None` when the buffer is empty.
    pub fn try_recv(&mut self) -> Option<T> {
        let mut shared = self.shared.lock().unwrap();
        let value = shared.buffer.pop_front();
        if (value.is_some() || shared.closed)
            && let Some(id) = self.id.take()
        {
            shared.recv_waiters.deregister(id);
        }
        let waker = value
            .as_ref()
            .and_then(|_| shared.send_waiters.grant_oldest(()));
        drop(shared);
        if let Some(waker) = waker {
            waker.wake();
        }
        value
    }

    /// Close admission, retaining accepted values for the receiver to drain.
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

impl<T, Mode> Drop for Receiver<T, Mode> {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.closed = true;
        if let Some(id) = self.id.take() {
            shared.recv_waiters.deregister(id);
        }
        let buffer = std::mem::take(&mut shared.buffer);
        let wakers = shared.send_waiters.grant_all(());
        drop(shared);
        // Payload destructors may re-enter a surviving sender. Release both
        // queued ownership and the state mutex before invoking user code.
        drop(buffer);
        for waker in wakers {
            waker.wake();
        }
    }
}

/// Future returned by [`Receiver::recv`].
pub struct RecvFuture<'a, T, Mode = Bounded> {
    receiver: &'a mut Receiver<T, Mode>,
    polled: bool,
}

impl<T, Mode> Future for RecvFuture<'_, T, Mode> {
    type Output = Result<T, ()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.polled = true;
        this.receiver.poll_recv(cx).map(|value| value.ok_or(()))
    }
}

impl<T, Mode> Drop for RecvFuture<'_, T, Mode> {
    fn drop(&mut self) {
        if self.polled
            && let Some(id) = self.receiver.id.take()
            && let Ok(mut shared) = self.receiver.shared.lock()
        {
            shared.recv_waiters.deregister(id);
        }
    }
}

/// Create a bounded channel with the given buffer capacity.
#[must_use]
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    create(Some(capacity))
}

/// Create an unbounded channel with synchronous admission without a capacity bound.
///
/// Accepted messages are FIFO; closure rejects new sends and retains queued
/// values until received or until the receiver is dropped. Producers allocate
/// as required, so callers remain responsible for application resource bounds.
#[must_use]
pub fn unbounded_channel<T>() -> (Sender<T, Unbounded>, Receiver<T, Unbounded>) {
    create(None)
}

fn create<T, Mode>(capacity: Option<usize>) -> (Sender<T, Mode>, Receiver<T, Mode>) {
    let shared = Arc::new(Mutex::new(SharedState {
        buffer: capacity.map_or_else(VecDeque::new, VecDeque::with_capacity),
        capacity,
        sender_count: 1,
        closed: false,
        send_waiters: WaitQueue::new(),
        recv_waiters: WaitQueue::new(),
    }));
    (
        Sender {
            shared: shared.clone(),
            mode: PhantomData,
        },
        Receiver {
            shared,
            id: None,
            mode: PhantomData,
        },
    )
}

#[cfg(test)]
#[path = "mpsc_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mpsc_unbounded_tests.rs"]
mod unbounded_tests;
