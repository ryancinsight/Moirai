//! Advanced async synchronization primitives for Moirai
//! 
//! Provides async-aware synchronization that integrates with Moirai's unified runtime

use moirai_core::{TaskId, Priority};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Async-aware semaphore for resource limiting
pub struct Semaphore {
    permits: Arc<Mutex<SemaphoreState>>,
}

struct SemaphoreState {
    available: usize,
    waiters: VecDeque<Waker>,
}

impl Semaphore {
    /// Create a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Self {
        Self {
            permits: Arc::new(Mutex::new(SemaphoreState {
                available: permits,
                waiters: VecDeque::new(),
            })),
        }
    }

    /// Acquire a permit asynchronously
    pub fn acquire(&self) -> SemaphoreAcquire<'_> {
        SemaphoreAcquire {
            semaphore: self,
            registered: false,
        }
    }

    /// Try to acquire a permit immediately
    pub fn try_acquire(&self) -> Option<SemaphorePermit<'_>> {
        let mut state = self.permits.lock().unwrap();
        if state.available > 0 {
            state.available -= 1;
            Some(SemaphorePermit { semaphore: self })
        } else {
            None
        }
    }

    /// Get the number of available permits
    pub fn available_permits(&self) -> usize {
        self.permits.lock().unwrap().available
    }

    fn release(&self) {
        let mut state = self.permits.lock().unwrap();
        state.available += 1;
        if let Some(waker) = state.waiters.pop_front() {
            drop(state);
            waker.wake();
        }
    }
}

/// Future for acquiring a semaphore permit
pub struct SemaphoreAcquire<'a> {
    semaphore: &'a Semaphore,
    registered: bool,
}

impl<'a> Future for SemaphoreAcquire<'a> {
    type Output = SemaphorePermit<'a>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.semaphore.permits.lock().unwrap();
        
        if state.available > 0 {
            state.available -= 1;
            Poll::Ready(SemaphorePermit {
                semaphore: self.semaphore,
            })
        } else {
            if !self.registered {
                state.waiters.push_back(cx.waker().clone());
                self.registered = true;
            }
            Poll::Pending
        }
    }
}

/// RAII guard for semaphore permit
pub struct SemaphorePermit<'a> {
    semaphore: &'a Semaphore,
}

impl<'a> Drop for SemaphorePermit<'a> {
    fn drop(&mut self) {
        self.semaphore.release();
    }
}

/// Broadcast channel for one-to-many communication
pub struct Broadcast<T> {
    state: Arc<Mutex<BroadcastState<T>>>,
}

struct BroadcastState<T> {
    messages: VecDeque<(u64, T)>,
    sequence: u64,
    receivers: Vec<BroadcastReceiverState>,
    capacity: usize,
}

struct BroadcastReceiverState {
    id: u64,
    position: u64,
    waker: Option<Waker>,
}

impl<T: Clone + Send + 'static> Broadcast<T> {
    /// Create a new broadcast channel with the given capacity
    pub fn new(capacity: usize) -> (BroadcastSender<T>, BroadcastReceiver<T>) {
        let state = Arc::new(Mutex::new(BroadcastState {
            messages: VecDeque::new(),
            sequence: 0,
            receivers: Vec::new(),
            capacity,
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
            state_guard.receivers.push(BroadcastReceiverState {
                id: 0,
                position: 0,
                waker: None,
            });
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
        
        // Remove old messages if at capacity
        while state.messages.len() >= state.capacity {
            state.messages.pop_front();
        }

        // Add new message
        state.sequence += 1;
        state.messages.push_back((state.sequence, message));

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

/// Receiver half of broadcast channel
pub struct BroadcastReceiver<T> {
    state: Arc<Mutex<BroadcastState<T>>>,
    id: u64,
    position: u64,
}

impl<T: Clone> BroadcastReceiver<T> {
    /// Receive the next message
    pub fn recv(&mut self) -> BroadcastRecv<'_, T> {
        BroadcastRecv {
            receiver: self,
            registered: false,
        }
    }

    /// Try to receive a message immediately
    pub fn try_recv(&mut self) -> Result<T, BroadcastError> {
        let state = self.state.lock().unwrap();
        
        // Find message at our position
        for (seq, message) in &state.messages {
            if *seq > self.position {
                self.position = *seq;
                return Ok(message.clone());
            }
        }

        Err(BroadcastError::Empty)
    }

    /// Clone this receiver to create a new independent receiver
    pub fn resubscribe(&self) -> BroadcastReceiver<T> {
        let mut state = self.state.lock().unwrap();
        let new_id = state.receivers.len() as u64;
        
        state.receivers.push(BroadcastReceiverState {
            id: new_id,
            position: state.sequence,
            waker: None,
        });

        BroadcastReceiver {
            state: self.state.clone(),
            id: new_id,
            position: state.sequence,
        }
    }
}

impl<T> Clone for BroadcastReceiver<T> {
    fn clone(&self) -> Self {
        self.resubscribe()
    }
}

impl<T> Drop for BroadcastReceiver<T> {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.receivers.retain(|r| r.id != self.id);
    }
}

/// Future for receiving from broadcast channel
pub struct BroadcastRecv<'a, T> {
    receiver: &'a mut BroadcastReceiver<T>,
    registered: bool,
}

impl<'a, T: Clone> Future for BroadcastRecv<'a, T> {
    type Output = Result<T, BroadcastError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.receiver.try_recv() {
            Ok(message) => Poll::Ready(Ok(message)),
            Err(BroadcastError::Empty) => {
                if !self.registered {
                    let mut state = self.receiver.state.lock().unwrap();
                    if let Some(receiver_state) = state.receivers.iter_mut().find(|r| r.id == self.receiver.id) {
                        receiver_state.waker = Some(cx.waker().clone());
                    }
                    self.registered = true;
                }
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

/// Watch channel for state monitoring with change notifications
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

        // Wake all receivers that are waiting for changes
        for receiver in &mut state.receivers {
            if receiver.version < state.version {
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
        self.version.max(state.version);
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
        
        state.receivers.push(WatchReceiverState {
            id: new_id,
            version: state.version,
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
        let mut state = self.receiver.state.lock().unwrap();
        
        if state.version > self.receiver.version {
            self.receiver.version = state.version;
            Poll::Ready(Ok(()))
        } else {
            if !self.registered {
                if let Some(receiver_state) = state.receivers.iter_mut().find(|r| r.id == self.receiver.id) {
                    receiver_state.waker = Some(cx.waker().clone());
                }
                self.registered = true;
            }
            Poll::Pending
        }
    }
}

/// Async-aware RwLock
pub struct RwLock<T> {
    inner: std::sync::RwLock<T>,
    read_waiters: Arc<Mutex<VecDeque<Waker>>>,
    write_waiters: Arc<Mutex<VecDeque<Waker>>>,
}

impl<T> RwLock<T> {
    /// Create a new async RwLock
    pub fn new(data: T) -> Self {
        Self {
            inner: std::sync::RwLock::new(data),
            read_waiters: Arc::new(Mutex::new(VecDeque::new())),
            write_waiters: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Acquire a read lock asynchronously
    pub fn read(&self) -> RwLockReadFuture<'_, T> {
        RwLockReadFuture {
            lock: self,
            registered: false,
        }
    }

    /// Acquire a write lock asynchronously
    pub fn write(&self) -> RwLockWriteFuture<'_, T> {
        RwLockWriteFuture {
            lock: self,
            registered: false,
        }
    }

    /// Try to acquire a read lock immediately
    pub fn try_read(&self) -> Option<std::sync::RwLockReadGuard<'_, T>> {
        self.inner.try_read().ok()
    }

    /// Try to acquire a write lock immediately
    pub fn try_write(&self) -> Option<std::sync::RwLockWriteGuard<'_, T>> {
        self.inner.try_write().ok()
    }
}

/// Future for async read lock acquisition
pub struct RwLockReadFuture<'a, T> {
    lock: &'a RwLock<T>,
    registered: bool,
}

impl<'a, T> Future for RwLockReadFuture<'a, T> {
    type Output = std::sync::RwLockReadGuard<'a, T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Ok(guard) = self.lock.inner.try_read() {
            Poll::Ready(guard)
        } else {
            if !self.registered {
                let mut waiters = self.lock.read_waiters.lock().unwrap();
                waiters.push_back(cx.waker().clone());
                self.registered = true;
            }
            Poll::Pending
        }
    }
}

/// Future for async write lock acquisition
pub struct RwLockWriteFuture<'a, T> {
    lock: &'a RwLock<T>,
    registered: false,
}

impl<'a, T> Future for RwLockWriteFuture<'a, T> {
    type Output = std::sync::RwLockWriteGuard<'a, T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Ok(guard) = self.lock.inner.try_write() {
            Poll::Ready(guard)
        } else {
            if !self.registered {
                let mut waiters = self.lock.write_waiters.lock().unwrap();
                waiters.push_back(cx.waker().clone());
                self.registered = true;
            }
            Poll::Pending
        }
    }
}

/// Notification primitive for efficient task coordination
pub struct Notify {
    waiters: Arc<Mutex<VecDeque<Waker>>>,
}

impl Notify {
    /// Create a new notification primitive
    pub fn new() -> Self {
        Self {
            waiters: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Wait for a notification
    pub fn notified(&self) -> NotifyFuture<'_> {
        NotifyFuture {
            notify: self,
            registered: false,
        }
    }

    /// Notify one waiting task
    pub fn notify_one(&self) {
        let mut waiters = self.waiters.lock().unwrap();
        if let Some(waker) = waiters.pop_front() {
            drop(waiters);
            waker.wake();
        }
    }

    /// Notify all waiting tasks
    pub fn notify_waiters(&self) {
        let mut waiters = self.waiters.lock().unwrap();
        let wakers: Vec<_> = waiters.drain(..).collect();
        drop(waiters);
        
        for waker in wakers {
            waker.wake();
        }
    }
}

impl Default for Notify {
    fn default() -> Self {
        Self::new()
    }
}

/// Future for waiting on notifications
pub struct NotifyFuture<'a> {
    notify: &'a Notify,
    registered: bool,
}

impl<'a> Future for NotifyFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.registered {
            let mut waiters = self.notify.waiters.lock().unwrap();
            waiters.push_back(cx.waker().clone());
            self.registered = true;
            Poll::Pending
        } else {
            // This future has already been registered and woken
            Poll::Ready(())
        }
    }
}

/// Error types for async synchronization primitives
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BroadcastError {
    /// Channel is empty
    Empty,
    /// Channel has been closed
    Closed,
    /// Message was lost due to channel overflow
    Lagged,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchError {
    /// Channel has been closed
    Closed,
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

impl std::fmt::Display for WatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatchError::Closed => write!(f, "watch channel is closed"),
        }
    }
}

impl std::error::Error for WatchError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_semaphore_basic() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let sem = Semaphore::new(2);
            
            let permit1 = sem.acquire().await;
            let permit2 = sem.acquire().await;
            
            assert_eq!(sem.available_permits(), 0);
            assert!(sem.try_acquire().is_none());
            
            drop(permit1);
            assert_eq!(sem.available_permits(), 1);
            
            drop(permit2);
            assert_eq!(sem.available_permits(), 2);
        });
    }

    #[test]
    fn test_broadcast_channel() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (tx, mut rx1) = Broadcast::new(10);
            let mut rx2 = rx1.resubscribe();
            
            tx.send("hello").unwrap();
            tx.send("world").unwrap();
            
            assert_eq!(rx1.recv().await.unwrap(), "hello");
            assert_eq!(rx1.recv().await.unwrap(), "world");
            
            assert_eq!(rx2.recv().await.unwrap(), "hello");
            assert_eq!(rx2.recv().await.unwrap(), "world");
        });
    }

    #[test]
    fn test_watch_channel() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (tx, mut rx) = Watch::new(0);
            
            assert_eq!(rx.borrow(), 0);
            
            tx.send(42).unwrap();
            rx.changed().await.unwrap();
            assert_eq!(rx.borrow(), 42);
            
            tx.send_modify(|x| *x += 1).unwrap();
            rx.changed().await.unwrap();
            assert_eq!(rx.borrow(), 43);
        });
    }

    #[test]
    fn test_notify() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let notify = Notify::new();
            
            let mut notified = false;
            let future = async {
                notify.notified().await;
                notified = true;
            };
            
            // Future should not complete immediately
            tokio::select! {
                _ = future => panic!("Should not complete immediately"),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
            
            notify.notify_one();
            
            // Now it should complete
            tokio::select! {
                _ = future => {},
                _ = tokio::time::sleep(Duration::from_millis(100)) => panic!("Should have completed"),
            }
            
            assert!(notified);
        });
    }
}