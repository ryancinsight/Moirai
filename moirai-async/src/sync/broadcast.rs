//! Broadcast channel for one-to-many async communication
//!
//! Provides broadcast channel implementation that allows one sender to
//! broadcast messages to multiple receivers with SLAP-compliant design.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Broadcast channel for one-to-many communication
pub struct Broadcast<T> {
    _phantom: std::marker::PhantomData<T>,
}

struct BroadcastState<T> {
    messages: VecDeque<(u64, T)>,
    sequence: u64,
    closed: bool,
    receivers: Vec<BroadcastReceiverState>,
    capacity: usize,
    next_receiver_id: u64,
}

struct BroadcastReceiverState {
    id: u64,
    waker: Option<Waker>,
}

impl<T: Clone + Send + 'static> Broadcast<T> {
    /// Create a new broadcast channel with the given capacity
    /// Returns (sender, receiver) tuple per channel pattern conventions
    #[allow(clippy::new_ret_no_self)] // Standard channel pattern per Rust Book Ch.16
    pub fn new(capacity: usize) -> (BroadcastSender<T>, BroadcastReceiver<T>) {
        let state = Arc::new(Mutex::new(BroadcastState {
            messages: VecDeque::new(),
            sequence: 0,
            closed: false,
            receivers: Vec::new(),
            capacity,
            next_receiver_id: 1,
        }));

        let sender = BroadcastSender {
            state: state.clone(),
        };

        let receiver = BroadcastReceiver {
            state: state.clone(),
            id: 0,
            position: 0,
        };

        // Register the first receiver
        {
            let mut state_guard = state.lock().unwrap();
            state_guard
                .receivers
                .push(BroadcastReceiverState { id: 0, waker: None });
        }

        (sender, receiver)
    }
}

/// Sender half of broadcast channel
pub struct BroadcastSender<T> {
    state: Arc<Mutex<BroadcastState<T>>>,
}

impl<T: Clone> BroadcastSender<T> {
    /// Send a message to all receivers
    pub fn send(&self, message: T) -> Result<usize, BroadcastError> {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return Err(BroadcastError::Closed);
        }

        // Remove old messages if at capacity
        while state.messages.len() >= state.capacity {
            state.messages.pop_front();
        }

        // Add new message
        state.sequence += 1;
        let sequence = state.sequence;
        state.messages.push_back((sequence, message));

        // Wake all receivers
        let receiver_count = state.receivers.len();
        for receiver in &mut state.receivers {
            if let Some(waker) = receiver.waker.take() {
                waker.wake();
            }
        }

        Ok(receiver_count)
    }

    /// Get the number of active receivers
    pub fn receiver_count(&self) -> usize {
        self.state.lock().unwrap().receivers.len()
    }
}

impl<T> Drop for BroadcastSender<T> {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.closed = true;
        for receiver in &mut state.receivers {
            if let Some(waker) = receiver.waker.take() {
                waker.wake();
            }
        }
    }
}

/// Receiver half of broadcast channel
pub struct BroadcastReceiver<T> {
    state: Arc<Mutex<BroadcastState<T>>>,
    id: u64,
    position: u64,
}

impl<T: Clone> BroadcastReceiver<T> {
    /// Receive the next message
    pub fn recv(&mut self) -> BroadcastRecv<'_, T> {
        BroadcastRecv { receiver: self }
    }

    /// Try to receive a message immediately
    pub fn try_recv(&mut self) -> Result<T, BroadcastError> {
        let state_arc = self.state.clone();
        let state = state_arc.lock().unwrap();

        if state.messages.is_empty() {
            if state.closed {
                return Err(BroadcastError::Closed);
            }
            return Err(BroadcastError::Empty);
        }

        // Lagging check
        let oldest_seq = state.messages.front().unwrap().0;
        if self.position + 1 < oldest_seq {
            self.position = oldest_seq - 1;
            return Err(BroadcastError::Lagged);
        }

        // Find message at our position
        for (seq, message) in &state.messages {
            if *seq > self.position {
                self.position = *seq;
                return Ok(message.clone());
            }
        }

        if state.closed {
            Err(BroadcastError::Closed)
        } else {
            Err(BroadcastError::Empty)
        }
    }

    /// Clone this receiver to create a new independent receiver
    pub fn resubscribe(&self) -> BroadcastReceiver<T> {
        let mut state = self.state.lock().unwrap();
        let new_id = state.next_receiver_id;
        state.next_receiver_id += 1;
        let current_sequence = state.sequence;

        state.receivers.push(BroadcastReceiverState {
            id: new_id,
            waker: None,
        });

        BroadcastReceiver {
            state: self.state.clone(),
            id: new_id,
            position: current_sequence,
        }
    }
}

impl<T: Clone> Clone for BroadcastReceiver<T> {
    fn clone(&self) -> Self {
        self.resubscribe()
    }
}

impl<T> Drop for BroadcastReceiver<T> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.receivers.retain(|r| r.id != self.id);
        }
    }
}

/// Future for receiving from broadcast channel
pub struct BroadcastRecv<'a, T> {
    receiver: &'a mut BroadcastReceiver<T>,
}

impl<'a, T: Clone> Future for BroadcastRecv<'a, T> {
    type Output = Result<T, BroadcastError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.receiver.try_recv() {
            Ok(message) => Poll::Ready(Ok(message)),
            Err(BroadcastError::Lagged) => Poll::Ready(Err(BroadcastError::Lagged)),
            Err(BroadcastError::Closed) => Poll::Ready(Err(BroadcastError::Closed)),
            Err(BroadcastError::Empty) => {
                let state_arc = self.receiver.state.clone();
                let mut state = state_arc.lock().unwrap();
                if state.closed {
                    Poll::Ready(Err(BroadcastError::Closed))
                } else {
                    if let Some(receiver_state) = state
                        .receivers
                        .iter_mut()
                        .find(|r| r.id == self.receiver.id)
                    {
                        receiver_state.waker = Some(cx.waker().clone());
                    }
                    Poll::Pending
                }
            }
        }
    }
}

/// Error types for broadcast channel operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BroadcastError {
    /// Channel is empty
    Empty,
    /// Channel has been closed
    Closed,
    /// Message was lost due to channel overflow
    Lagged,
}

impl std::fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BroadcastError::Empty => write!(f, "broadcast channel is empty"),
            BroadcastError::Closed => write!(f, "broadcast channel is closed"),
            BroadcastError::Lagged => write!(f, "broadcast channel lagged"),
        }
    }
}

impl std::error::Error for BroadcastError {}
