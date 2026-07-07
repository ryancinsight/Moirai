//! Watch channel for state monitoring with change notifications
//!
//! Provides watch channel implementation that allows monitoring state changes
//! with async notifications, following SLAP principle design.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Watch channel for state monitoring with change notifications
pub struct Watch<T> {
    _phantom: std::marker::PhantomData<T>,
}

struct WatchState<T> {
    value: T,
    version: u64,
    closed: bool,
    /// Receiver state keyed by receiver id. Keyed (rather than a linear `Vec`)
    /// so the per-`poll`/`has_changed`/drop lookup of a receiver's slot is
    /// O(log n) instead of O(n), shortening lock-hold when many receivers
    /// subscribe to one channel. The send fan-out iterates all receivers
    /// regardless, which is inherently O(n). Mirrors `Broadcast`'s structure.
    receivers: BTreeMap<u64, WatchReceiverState>,
    next_receiver_id: u64,
}

struct WatchReceiverState {
    version: u64,
    waker: Option<Waker>,
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
            receivers: BTreeMap::new(),
            next_receiver_id: 1,
        }));

        let sender = WatchSender {
            state: state.clone(),
        };

        let receiver = WatchReceiver {
            state: state.clone(),
            id: 0,
            version: 0,
        };

        // Register first receiver
        {
            let mut state_guard = state.lock().unwrap();
            state_guard.receivers.insert(
                0,
                WatchReceiverState {
                    version: 0,
                    waker: None,
                },
            );
        }

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
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return Err(WatchError::Closed);
        }
        state.value = value;
        state.version += 1;
        let current_version = state.version;

        // Wake all receivers that are waiting for changes
        for receiver in state.receivers.values_mut() {
            if receiver.version < current_version {
                if let Some(waker) = receiver.waker.take() {
                    waker.wake();
                }
            }
        }

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
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return Err(WatchError::Closed);
        }
        modify(&mut state.value);
        state.version += 1;

        // Wake all receivers
        for receiver in state.receivers.values_mut() {
            if let Some(waker) = receiver.waker.take() {
                waker.wake();
            }
        }

        Ok(())
    }

    /// Get the number of active receivers
    pub fn receiver_count(&self) -> usize {
        self.state.lock().unwrap().receivers.len()
    }
}

impl<T> Drop for WatchSender<T> {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.closed = true;
        for receiver in state.receivers.values_mut() {
            if let Some(waker) = receiver.waker.take() {
                waker.wake();
            }
        }
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
            if let Some(receiver_state) = state.receivers.get_mut(&self.id) {
                receiver_state.version = current_version;
            }
        }
        changed
    }
}

impl<T> Clone for WatchReceiver<T> {
    fn clone(&self) -> Self {
        let mut state = self.state.lock().unwrap();
        let new_id = state.next_receiver_id;
        state.next_receiver_id += 1;
        let current_version = state.version;

        state.receivers.insert(
            new_id,
            WatchReceiverState {
                version: current_version,
                waker: None,
            },
        );

        WatchReceiver {
            state: self.state.clone(),
            id: new_id,
            version: current_version,
        }
    }
}

impl<T> Drop for WatchReceiver<T> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.receivers.remove(&self.id);
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
            if let Some(receiver_state) = state.receivers.get_mut(&receiver.id) {
                receiver_state.version = current_version;
            }
            return Poll::Ready(Ok(()));
        }

        if let Some(receiver_state) = state.receivers.get_mut(&receiver.id) {
            receiver_state.waker = Some(cx.waker().clone());
        }

        Poll::Pending
    }
}

impl<'a, T> Drop for WatchChanged<'a, T> {
    fn drop(&mut self) {
        // If this future is dropped while pending, the waker stored in
        // `receiver_state.waker` would be called by the next `send()` on a
        // now-deallocated task allocation — a use-after-free of the waker.
        // Clear it here so the sender only wakes live futures.
        if let Ok(mut state) = self.receiver.state.lock() {
            if let Some(receiver_state) = state.receivers.get_mut(&self.receiver.id) {
                receiver_state.waker = None;
            }
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
