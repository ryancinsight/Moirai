//! Watch channel for state monitoring with change notifications
//!
//! Provides watch channel implementation that allows monitoring state changes
//! with async notifications, following SLAP principle design.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Watch channel for state monitoring with change notifications
#[allow(dead_code)] // Fields used for future watch functionality per ADR requirements
pub struct Watch<T> {
    state: Arc<Mutex<WatchState<T>>>,
}

struct WatchState<T> {
    value: T,
    version: u64,
    receivers: Vec<WatchReceiverState>,
}

struct WatchReceiverState {
    id: u64,
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
            receivers: Vec::new(),
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
            state_guard.receivers.push(WatchReceiverState {
                id: 0,
                version: 0,
                waker: None,
            });
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
        state.value = value;
        state.version += 1;
        let current_version = state.version;

        // Wake all receivers that are waiting for changes
        for receiver in &mut state.receivers {
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
        modify(&mut state.value);
        state.version += 1;

        // Wake all receivers
        for receiver in &mut state.receivers {
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
        let _ = self.version.max(state.version);
        state.value.clone()
    }

    /// Wait for the value to change
    pub fn changed(&mut self) -> WatchChanged<'_, T> {
        WatchChanged {
            receiver: self,
            registered: false,
        }
    }

    /// Check if the value has changed since last check
    pub fn has_changed(&mut self) -> bool {
        let state = self.state.lock().unwrap();
        let changed = state.version > self.version;
        if changed {
            self.version = state.version;
        }
        changed
    }
}

impl<T> Clone for WatchReceiver<T> {
    fn clone(&self) -> Self {
        let mut state = self.state.lock().unwrap();
        let new_id = state.receivers.len() as u64;
        let current_version = state.version;
        
        state.receivers.push(WatchReceiverState {
            id: new_id,
            version: current_version,
            waker: None,
        });

        WatchReceiver {
            state: self.state.clone(),
            id: new_id,
            version: state.version,
        }
    }
}

impl<T> Drop for WatchReceiver<T> {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.receivers.retain(|r| r.id != self.id);
    }
}

/// Future for waiting for watch value changes
pub struct WatchChanged<'a, T> {
    receiver: &'a mut WatchReceiver<T>,
    registered: bool,
}

impl<'a, T: Clone> Future for WatchChanged<'a, T> {
    type Output = Result<(), WatchError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut state = self.receiver.state.lock().unwrap();
            
            if state.version > self.receiver.version {
                let current_version = state.version;
                drop(state);  // Release lock before modifying self
                self.receiver.version = current_version;
                return Poll::Ready(Ok(()));
            } else if !self.registered {
                if let Some(receiver_state) = state.receivers.iter_mut().find(|r| r.id == self.receiver.id) {
                    receiver_state.waker = Some(cx.waker().clone());
                }
            }
        }
        
        if !self.registered {
            self.registered = true;
        }
        
        Poll::Pending
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