//! Watch channel for state monitoring with change notifications
//!
//! Provides watch channel implementation that allows monitoring state changes
//! with async notifications, following SLAP principle design.

#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use super::subscribers::{SubscriberRegistry, wake_drained};

/// Watch channel for state monitoring with change notifications
pub struct Watch<T> {
    _phantom: std::marker::PhantomData<T>,
}

struct WatchState<T> {
    value: T,
    version: u64,
    closed: bool,
    /// One slot per receiver, shared with `Broadcast` (see `subscribers`). The
    /// cursor is the version that receiver last observed; the slot's waker is
    /// registered while it waits for a newer one.
    subscribers: SubscriberRegistry<u64>,
}

impl<T: Clone + Send + 'static> Watch<T> {
    /// Create a new watch channel with an initial value
    /// Returns (sender, receiver) tuple per channel pattern conventions
    #[allow(clippy::new_ret_no_self)] // Standard channel pattern per Rust Book Ch.16
    pub fn new(initial: T) -> (WatchSender<T>, WatchReceiver<T>) {
        let state = Arc::new(Mutex::new(WatchState {
            value: initial,
            version: 0,
            closed: false,
            subscribers: SubscriberRegistry::with_initial(0),
        }));

        let sender = WatchSender {
            state: state.clone(),
        };

        let receiver = WatchReceiver {
            state: state.clone(),
            id: 0,
            version: 0,
        };

        (sender, receiver)
    }
}

/// Sender half of watch channel
pub struct WatchSender<T> {
    state: Arc<Mutex<WatchState<T>>>,
}

impl<T: Clone> WatchSender<T> {
    /// Send a new value, notifying all receivers
    pub fn send(&self, value: T) -> Result<(), WatchError> {
        let wakers = {
            let mut state = self.state.lock().unwrap();
            if state.closed {
                return Err(WatchError::Closed);
            }
            state.value = value;
            state.version += 1;
            state.subscribers.drain_wakers()
        };
        wake_drained(wakers);
        Ok(())
    }

    /// Get the current value
    pub fn borrow(&self) -> T {
        self.state.lock().unwrap().value.clone()
    }

    /// Modify the value in place and notify receivers
    pub fn send_modify<F>(&self, modify: F) -> Result<(), WatchError>
    where
        F: FnOnce(&mut T),
    {
        let wakers = {
            let mut state = self.state.lock().unwrap();
            if state.closed {
                return Err(WatchError::Closed);
            }
            modify(&mut state.value);
            state.version += 1;
            state.subscribers.drain_wakers()
        };
        wake_drained(wakers);
        Ok(())
    }

    /// Get the number of active receivers
    pub fn receiver_count(&self) -> usize {
        self.state.lock().unwrap().subscribers.len()
    }
}

impl<T> Drop for WatchSender<T> {
    fn drop(&mut self) {
        let wakers = {
            let mut state = self.state.lock().unwrap();
            state.closed = true;
            state.subscribers.drain_wakers()
        };
        wake_drained(wakers);
    }
}

/// Receiver half of watch channel
pub struct WatchReceiver<T> {
    state: Arc<Mutex<WatchState<T>>>,
    id: u64,
    version: u64,
}

impl<T: Clone> WatchReceiver<T> {
    /// Get the current value
    pub fn borrow(&self) -> T {
        let state = self.state.lock().unwrap();
        state.value.clone()
    }

    /// Wait for the value to change
    pub fn changed(&mut self) -> WatchChanged<'_, T> {
        WatchChanged { receiver: self }
    }

    /// Check if the value has changed since last check
    pub fn has_changed(&mut self) -> bool {
        let mut state = self.state.lock().unwrap();
        let changed = state.version > self.version;
        if changed {
            let current_version = state.version;
            self.version = current_version;
            if let Some(subscriber) = state.subscribers.get_mut(self.id) {
                subscriber.cursor = current_version;
            }
        }
        changed
    }
}

impl<T> Clone for WatchReceiver<T> {
    fn clone(&self) -> Self {
        let mut state = self.state.lock().unwrap();
        let current_version = state.version;
        let id = state.subscribers.register(current_version);

        WatchReceiver {
            state: self.state.clone(),
            id,
            version: current_version,
        }
    }
}

impl<T> Drop for WatchReceiver<T> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.subscribers.remove(self.id);
        }
    }
}

/// Future for waiting for watch value changes
pub struct WatchChanged<'a, T> {
    receiver: &'a mut WatchReceiver<T>,
}

impl<'a, T: Clone> Future for WatchChanged<'a, T> {
    type Output = Result<(), WatchError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let receiver = &mut *self.receiver;
        let mut state = receiver.state.lock().unwrap();

        if state.closed {
            return Poll::Ready(Err(WatchError::Closed));
        }

        let current_version = state.version;
        if current_version > receiver.version {
            receiver.version = current_version;
            if let Some(subscriber) = state.subscribers.get_mut(receiver.id) {
                subscriber.cursor = current_version;
            }
            return Poll::Ready(Ok(()));
        }

        if let Some(subscriber) = state.subscribers.get_mut(receiver.id) {
            subscriber.waker = Some(cx.waker().clone());
        }

        Poll::Pending
    }
}

impl<'a, T> Drop for WatchChanged<'a, T> {
    fn drop(&mut self) {
        // If this future is dropped while pending, the waker left in the
        // subscriber slot would be called by the next `send()` on a
        // now-deallocated task allocation — a use-after-free of the waker.
        // Clear it here so the sender only wakes live futures.
        if let Ok(mut state) = self.receiver.state.lock()
            && let Some(subscriber) = state.subscribers.get_mut(self.receiver.id)
        {
            subscriber.waker = None;
        }
    }
}

/// Error types for watch channel operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchError {
    /// Channel has been closed
    Closed,
}

impl std::fmt::Display for WatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatchError::Closed => write!(f, "watch channel is closed"),
        }
    }
}

impl std::error::Error for WatchError {}
