//! Broadcast channel for one-to-many async communication
//!
//! Provides broadcast channel implementation that allows one sender to
//! broadcast messages to multiple receivers with SLAP-compliant design.

use std::collections::{BTreeMap, VecDeque};
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
    /// Receiver state keyed by receiver id. Keyed (rather than a linear `Vec`)
    /// so the per-`poll_recv` and per-drop lookup of a receiver's waker slot is
    /// O(log n) instead of O(n), shortening lock-hold when many receivers
    /// subscribe to one channel. The send/sender-drop fan-out iterates all
    /// receivers regardless, which is inherently O(n).
    receivers: BTreeMap<u64, BroadcastReceiverState>,
    capacity: usize,
    next_receiver_id: u64,
}

struct BroadcastReceiverState {
    waker: Option<Waker>,
    /// Last sequence consumed by this receiver.  Used to compute the retention
    /// boundary so messages read by every receiver can be reclaimed.
    position: u64,
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
            receivers: BTreeMap::new(),
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
            state_guard.receivers.insert(
                0,
                BroadcastReceiverState {
                    waker: None,
                    position: 0,
                },
            );
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

        // Add new message first, then reclaim messages already read by every
        // receiver.  The retention boundary is the minimum position across all
        // live receivers; messages with sequence <= that boundary have been
        // consumed by everyone and can be dropped.
        state.sequence += 1;
        let sequence = state.sequence;
        state.messages.push_back((sequence, message));

        // Wake registered receivers after publication. They observe either the
        // appended message or the explicit lag state retained by the capacity
        // contract.
        let receiver_count = state.receivers.len();
        for receiver in state.receivers.values_mut() {
            if let Some(waker) = receiver.waker.take() {
                waker.wake();
            }
        }

        // Reclaim memory from messages read by every receiver, while respecting
        // the configured capacity window.
        let min_position = state
            .receivers
            .values()
            .map(|r| r.position)
            .min()
            .unwrap_or(sequence);
        while state.messages.len() > state.capacity
            || state
                .messages
                .front()
                .is_some_and(|(seq, _)| *seq <= min_position)
        {
            state.messages.pop_front();
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
        for receiver in state.receivers.values_mut() {
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
        let mut state = self.state.lock().unwrap();

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
            if let Some(rx_state) = state.receivers.get_mut(&self.id) {
                rx_state.position = self.position;
            }
            return Err(BroadcastError::Lagged);
        }

        // Sequences are dense (each `send` appends `sequence + 1` and only
        // `pop_front` removes), so the next unread message sits at a directly
        // computable offset from the front — no scan. After the lag check,
        // `position + 1 >= oldest_seq`, and `position <= sequence` bounds the
        // offset by the queue length, so the conversion cannot truncate.
        let offset = usize::try_from(self.position + 1 - oldest_seq)
            .expect("invariant: unread offset is bounded by the message queue length");
        let found = state.messages.get(offset).map(|(seq, message)| {
            debug_assert_eq!(*seq, self.position + 1, "broadcast sequences must be dense");
            self.position = *seq;
            message.clone()
        });
        if let Some(message) = found {
            if let Some(rx_state) = state.receivers.get_mut(&self.id) {
                rx_state.position = self.position;
            }
            return Ok(message);
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

        state.receivers.insert(
            new_id,
            BroadcastReceiverState {
                waker: None,
                position: current_sequence,
            },
        );

        BroadcastReceiver {
            state: self.state.clone(),
            id: new_id,
            position: current_sequence,
        }
    }

    /// Poll to receive a message, registering waker if empty.
    pub fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Result<T, BroadcastError>> {
        let mut state = self.state.lock().unwrap();

        if state.messages.is_empty() {
            if state.closed {
                return Poll::Ready(Err(BroadcastError::Closed));
            }
            if let Some(receiver_state) = state.receivers.get_mut(&self.id) {
                receiver_state.waker = Some(cx.waker().clone());
            }
            return Poll::Pending;
        }

        // Lagging check
        let oldest_seq = state.messages.front().unwrap().0;
        if self.position + 1 < oldest_seq {
            self.position = oldest_seq - 1;
            if let Some(rx_state) = state.receivers.get_mut(&self.id) {
                rx_state.position = self.position;
            }
            return Poll::Ready(Err(BroadcastError::Lagged));
        }

        // Dense-sequence direct index; see `try_recv` for the derivation.
        let offset = usize::try_from(self.position + 1 - oldest_seq)
            .expect("invariant: unread offset is bounded by the message queue length");
        let found_msg = state.messages.get(offset).map(|(seq, message)| {
            debug_assert_eq!(*seq, self.position + 1, "broadcast sequences must be dense");
            self.position = *seq;
            (*seq, message.clone())
        });

        if let Some((_, message)) = found_msg {
            if let Some(rx_state) = state.receivers.get_mut(&self.id) {
                rx_state.position = self.position;
            }
            Poll::Ready(Ok(message))
        } else if state.closed {
            Poll::Ready(Err(BroadcastError::Closed))
        } else {
            if let Some(receiver_state) = state.receivers.get_mut(&self.id) {
                receiver_state.waker = Some(cx.waker().clone());
            }
            Poll::Pending
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
            state.receivers.remove(&self.id);
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
        self.receiver.poll_recv(cx)
    }
}

impl<'a, T> Drop for BroadcastRecv<'a, T> {
    fn drop(&mut self) {
        // Clear the waker registered by `poll_recv` so a cancelled recv future
        // does not leave a stale waker that the next `send` would spuriously wake
        // (and retain until overwritten). This mirrors `WatchChanged::drop`. The
        // future holds `&mut BroadcastReceiver`, so it is the unique registrant
        // for this receiver id — clearing here cannot drop another future's waker.
        if let Ok(mut state) = self.receiver.state.lock() {
            let id = self.receiver.id;
            if let Some(receiver_state) = state.receivers.get_mut(&id) {
                receiver_state.waker = None;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

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
    fn cancelled_recv_clears_waker_and_is_not_spuriously_woken() {
        let (tx, mut rx) = Broadcast::<u32>::new(8);
        let count = Arc::new(AtomicUsize::new(0));
        let waker = Waker::from(Arc::new(CountingWake(Arc::clone(&count))));
        let mut cx = Context::from_waker(&waker);

        {
            let mut fut = rx.recv();
            assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());
            // `fut` is dropped here; its Drop must clear the registered waker.
        }

        // The cancelled future's waker must not fire on the next send.
        tx.send(42).expect("send must succeed");
        assert_eq!(
            count.load(Ordering::Acquire),
            0,
            "a cancelled recv future must not be spuriously woken"
        );

        // The message is still deliverable to a fresh recv on the same receiver.
        assert_eq!(rx.try_recv(), Ok(42));
    }

    #[test]
    fn live_recv_is_woken_on_send() {
        // Control: a still-live registered waker IS woken by send.
        let (tx, mut rx) = Broadcast::<u32>::new(8);
        let count = Arc::new(AtomicUsize::new(0));
        let waker = Waker::from(Arc::new(CountingWake(Arc::clone(&count))));
        let mut cx = Context::from_waker(&waker);

        let mut fut = rx.recv();
        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());
        tx.send(7).expect("send must succeed");
        assert_eq!(
            count.load(Ordering::Acquire),
            1,
            "a live recv future must be woken by send"
        );
        // Keep `fut` alive across the send so its waker stays registered.
        drop(fut);
    }

    #[test]
    fn messages_read_by_every_receiver_are_reclaimed_on_next_send() {
        let (tx, mut first) = Broadcast::<u32>::new(8);
        let mut second = first.resubscribe();

        tx.send(10).expect("first send must succeed");
        tx.send(20).expect("second send must succeed");
        assert_eq!(first.try_recv(), Ok(10));
        assert_eq!(second.try_recv(), Ok(10));
        assert_eq!(first.try_recv(), Ok(20));
        assert_eq!(second.try_recv(), Ok(20));

        tx.send(30).expect("third send must succeed");
        let state = tx.state.lock().expect("broadcast state must not poison");
        assert_eq!(
            state.messages.iter().copied().collect::<Vec<_>>(),
            vec![(3, 30)],
            "the next send must retain only the new unread message"
        );
    }
}
