//! Zero-copy hybrid channel for async/sync interop.
//!
//! Uses a lock-free ring buffer with memory barriers to ensure safe
//! zero-copy communication between async and sync contexts.

use super::error::{ChannelError, Result};
use crate::communication::RingBuffer;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Zero-copy hybrid channel for async/sync interop
///
/// This channel uses a lock-free ring buffer with memory barriers
/// to ensure safe zero-copy communication between async and sync contexts.
pub struct HybridChannel<T> {
    ring: Arc<RingBuffer<T>>,
    async_notifier: Arc<AtomicBool>,
    sync_notifier: Arc<AtomicBool>,
    /// Parking mechanism for efficient blocking
    parker: Arc<Mutex<Vec<std::thread::Thread>>>,
    /// Waker registration for zero-cost async polling
    async_wakers: Arc<Mutex<Vec<(u64, Waker)>>>,
    parked_count: Arc<AtomicUsize>,
    waker_count: Arc<AtomicUsize>,
    closed: Arc<AtomicBool>,
    next_id: Arc<AtomicU64>,
}

impl<T: Send> HybridChannel<T> {
    /// Create a new hybrid channel with specified capacity
    pub fn new(capacity: usize) -> (HybridSender<T>, HybridReceiver<T>) {
        let channel = Self {
            ring: Arc::new(RingBuffer::new(capacity)),
            async_notifier: Arc::new(AtomicBool::new(false)),
            sync_notifier: Arc::new(AtomicBool::new(false)),
            parker: Arc::new(Mutex::new(Vec::new())),
            async_wakers: Arc::new(Mutex::new(Vec::new())),
            parked_count: Arc::new(AtomicUsize::new(0)),
            waker_count: Arc::new(AtomicUsize::new(0)),
            closed: Arc::new(AtomicBool::new(false)),
            next_id: Arc::new(AtomicU64::new(0)),
        };

        channel.split()
    }

    /// Split the channel into sender and receiver halves
    fn split(self) -> (HybridSender<T>, HybridReceiver<T>) {
        let sender = HybridSender {
            ring: self.ring.clone(),
            async_notifier: self.async_notifier.clone(),
            sync_notifier: self.sync_notifier.clone(),
            parker: self.parker.clone(),
            async_wakers: self.async_wakers.clone(),
            parked_count: self.parked_count.clone(),
            waker_count: self.waker_count.clone(),
            closed: self.closed.clone(),
            _marker: PhantomData,
        };

        let receiver = HybridReceiver {
            ring: self.ring,
            async_notifier: self.async_notifier,
            sync_notifier: self.sync_notifier,
            parker: self.parker,
            async_wakers: self.async_wakers,
            parked_count: self.parked_count,
            waker_count: self.waker_count,
            closed: self.closed,
            next_id: self.next_id,
            _marker: PhantomData,
        };

        (sender, receiver)
    }

    /// Get the capacity of the channel
    pub fn capacity(&self) -> usize {
        self.ring.capacity()
    }

    /// Check if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Check if the channel is full
    pub fn is_full(&self) -> bool {
        self.ring.is_full()
    }

    /// Get the number of items currently in the channel
    pub fn len(&self) -> usize {
        self.ring.len()
    }
}

// ---------------------------------------------------------------------------
// HybridSender
// ---------------------------------------------------------------------------

/// Sender half of hybrid channel
pub struct HybridSender<T> {
    ring: Arc<RingBuffer<T>>,
    async_notifier: Arc<AtomicBool>,
    sync_notifier: Arc<AtomicBool>,
    parker: Arc<Mutex<Vec<std::thread::Thread>>>,
    async_wakers: Arc<Mutex<Vec<(u64, Waker)>>>,
    parked_count: Arc<AtomicUsize>,
    waker_count: Arc<AtomicUsize>,
    closed: Arc<AtomicBool>,
    _marker: PhantomData<std::cell::Cell<()>>,
}

impl<T: Send> HybridSender<T> {
    /// Send value with zero-copy when possible
    pub fn send(&self, value: T) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(ChannelError::Closed);
        }

        self.ring
            .try_produce(value)
            .map_err(|_| ChannelError::Full)?;

        // Notify both async and sync waiters
        self.async_notifier.store(true, Ordering::Release);
        self.sync_notifier.store(true, Ordering::Release);

        // Unpark any waiting threads if there are any
        if self.parked_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut parked) = self.parker.lock() {
                for thread in parked.drain(..) {
                    thread.unpark();
                }
                self.parked_count.store(0, Ordering::Release);
            }
        }

        // Wake any waiting async tasks if there are any
        if self.waker_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut wakers) = self.async_wakers.lock() {
                for (_, waker) in wakers.drain(..) {
                    waker.wake();
                }
                self.waker_count.store(0, Ordering::Release);
            }
        }

        Ok(())
    }

    /// Try to send without blocking
    pub fn try_send(&self, value: T) -> Result<()> {
        self.send(value)
    }

    /// Send with timeout
    pub fn send_timeout(&self, mut value: T, timeout: std::time::Duration) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::Closed);
            }

            match self.ring.try_produce(value) {
                Ok(()) => {
                    // Notify waiters
                    self.async_notifier.store(true, Ordering::Release);
                    self.sync_notifier.store(true, Ordering::Release);

                    // Unpark any waiting threads if there are any
                    if self.parked_count.load(Ordering::Relaxed) > 0 {
                        if let Ok(mut parked) = self.parker.lock() {
                            for thread in parked.drain(..) {
                                thread.unpark();
                            }
                            self.parked_count.store(0, Ordering::Release);
                        }
                    }

                    // Wake any waiting async tasks if there are any
                    if self.waker_count.load(Ordering::Relaxed) > 0 {
                        if let Ok(mut wakers) = self.async_wakers.lock() {
                            for (_, waker) in wakers.drain(..) {
                                waker.wake();
                            }
                            self.waker_count.store(0, Ordering::Release);
                        }
                    }

                    return Ok(());
                }
                Err(v) => {
                    value = v; // Get the value back
                    if start.elapsed() >= timeout {
                        return Err(ChannelError::Full);
                    }
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Check if the sender can send without blocking
    pub fn can_send(&self) -> bool {
        !self.ring.is_full() && !self.closed.load(Ordering::Acquire)
    }

    /// Get the number of items that can be sent without blocking
    pub fn available_capacity(&self) -> usize {
        if self.closed.load(Ordering::Acquire) {
            0
        } else {
            self.ring.capacity() - self.ring.len()
        }
    }
}

impl<T> Drop for HybridSender<T> {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        // Unpark any waiting threads if there are any
        if self.parked_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut parked) = self.parker.lock() {
                for thread in parked.drain(..) {
                    thread.unpark();
                }
                self.parked_count.store(0, Ordering::Release);
            }
        }
        // Wake any waiting async tasks if there are any
        if self.waker_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut wakers) = self.async_wakers.lock() {
                for (_, waker) in wakers.drain(..) {
                    waker.wake();
                }
                self.waker_count.store(0, Ordering::Release);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HybridReceiver
// ---------------------------------------------------------------------------

/// Receiver half of hybrid channel
pub struct HybridReceiver<T> {
    ring: Arc<RingBuffer<T>>,
    #[allow(dead_code)]
    async_notifier: Arc<AtomicBool>,
    #[allow(dead_code)]
    sync_notifier: Arc<AtomicBool>,
    parker: Arc<Mutex<Vec<std::thread::Thread>>>,
    async_wakers: Arc<Mutex<Vec<(u64, Waker)>>>,
    parked_count: Arc<AtomicUsize>,
    waker_count: Arc<AtomicUsize>,
    closed: Arc<AtomicBool>,
    next_id: Arc<AtomicU64>,
    _marker: PhantomData<std::cell::Cell<()>>,
}

impl<T: Send> HybridReceiver<T> {
    /// Receive value with zero-copy
    pub fn recv(&self) -> Result<T> {
        // Fast path: try to receive without blocking
        if let Some(value) = self.ring.try_consume() {
            return Ok(value);
        }

        // Slow path: park the thread and wait for notification
        let current_thread = std::thread::current();

        loop {
            if self.closed.load(Ordering::Acquire) && self.ring.is_empty() {
                return Err(ChannelError::Closed);
            }

            // Register this thread for unparking if not already present
            if let Ok(mut parked) = self.parker.lock() {
                if !parked.iter().any(|t| t.id() == current_thread.id()) {
                    parked.push(current_thread.clone());
                    self.parked_count.fetch_add(1, Ordering::Release);
                }
            }

            // Check again after registering (to avoid race)
            if let Some(value) = self.ring.try_consume() {
                // Remove ourselves from the parker list
                if let Ok(mut parked) = self.parker.lock() {
                    let old_len = parked.len();
                    parked.retain(|t| !t.id().eq(&current_thread.id()));
                    let removed = old_len - parked.len();
                    if removed > 0 {
                        self.parked_count.fetch_sub(removed, Ordering::Release);
                    }
                }
                return Ok(value);
            }

            if self.closed.load(Ordering::Acquire) && self.ring.is_empty() {
                if let Ok(mut parked) = self.parker.lock() {
                    let old_len = parked.len();
                    parked.retain(|t| !t.id().eq(&current_thread.id()));
                    let removed = old_len - parked.len();
                    if removed > 0 {
                        self.parked_count.fetch_sub(removed, Ordering::Release);
                    }
                }
                return Err(ChannelError::Closed);
            }

            // Park until unparked by sender
            std::thread::park();
        }
    }

    /// Try to receive without blocking
    pub fn try_recv(&self) -> Result<T> {
        if let Some(value) = self.ring.try_consume() {
            Ok(value)
        } else if self.closed.load(Ordering::Acquire) {
            Err(ChannelError::Closed)
        } else {
            Err(ChannelError::Empty)
        }
    }

    /// Receive with timeout
    pub fn recv_timeout(&self, timeout: std::time::Duration) -> Result<T> {
        let start = std::time::Instant::now();

        loop {
            match self.try_recv() {
                Ok(value) => return Ok(value),
                Err(ChannelError::Empty) => {
                    if start.elapsed() >= timeout {
                        return Err(ChannelError::Empty);
                    }

                    // Register for wake-up before checking again
                    let current_thread = std::thread::current();
                    if let Ok(mut parked) = self.parker.lock() {
                        if !parked.iter().any(|t| t.id() == current_thread.id()) {
                            parked.push(current_thread.clone());
                            self.parked_count.fetch_add(1, Ordering::Release);
                        }
                    }

                    // Park for the remaining timeout budget.
                    if let Some(remaining) = timeout.checked_sub(start.elapsed()) {
                        std::thread::park_timeout(remaining);
                    }

                    // Remove from parker list
                    if let Ok(mut parked) = self.parker.lock() {
                        let old_len = parked.len();
                        parked.retain(|t| !t.id().eq(&current_thread.id()));
                        let removed = old_len - parked.len();
                        if removed > 0 {
                            self.parked_count.fetch_sub(removed, Ordering::Release);
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Check if there are messages available
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Get the number of messages available
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Drain all available messages
    pub fn drain(&self) -> Vec<T> {
        let mut messages = Vec::new();
        while let Ok(msg) = self.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// Async receive for use in async contexts (zero-cost waker-based Future)
    #[cfg(feature = "std")]
    pub fn recv_async(&self) -> RecvFuture<'_, T> {
        RecvFuture {
            receiver: self,
            id: None,
        }
    }
}

impl<T> Drop for HybridReceiver<T> {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        // Unpark any waiting threads (just in case) if there are any
        if self.parked_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut parked) = self.parker.lock() {
                for thread in parked.drain(..) {
                    thread.unpark();
                }
                self.parked_count.store(0, Ordering::Release);
            }
        }
        // Wake any waiting async tasks (just in case) if there are any
        if self.waker_count.load(Ordering::Relaxed) > 0 {
            if let Ok(mut wakers) = self.async_wakers.lock() {
                for (_, waker) in wakers.drain(..) {
                    waker.wake();
                }
                self.waker_count.store(0, Ordering::Release);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RecvFuture
// ---------------------------------------------------------------------------

/// Future implementation for zero-cost async receive
pub struct RecvFuture<'a, T> {
    receiver: &'a HybridReceiver<T>,
    id: Option<u64>,
}

impl<T: Send> Future for RecvFuture<'_, T> {
    type Output = Result<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Fast path check
        if let Some(value) = self.receiver.ring.try_consume() {
            return Poll::Ready(Ok(value));
        }

        if self.receiver.closed.load(Ordering::Acquire) {
            return Poll::Ready(Err(ChannelError::Closed));
        }

        let this = self.get_mut();
        // Lock wakers to register this task
        if let Ok(mut wakers) = this.receiver.async_wakers.lock() {
            // Check again after locking to prevent race condition
            if let Some(value) = this.receiver.ring.try_consume() {
                return Poll::Ready(Ok(value));
            }
            if this.receiver.closed.load(Ordering::Acquire) {
                return Poll::Ready(Err(ChannelError::Closed));
            }

            let waker = cx.waker();
            if let Some(id) = this.id {
                if let Some(pos) = wakers.iter().position(|(w_id, _)| *w_id == id) {
                    wakers[pos].1.clone_from(waker);
                } else {
                    wakers.push((id, waker.clone()));
                    this.receiver.waker_count.fetch_add(1, Ordering::Release);
                }
            } else {
                let id = this.receiver.next_id.fetch_add(1, Ordering::Relaxed);
                wakers.push((id, waker.clone()));
                this.id = Some(id);
                this.receiver.waker_count.fetch_add(1, Ordering::Release);
            }
        }

        Poll::Pending
    }
}

impl<T> Drop for RecvFuture<'_, T> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut wakers) = self.receiver.async_wakers.lock() {
                if let Some(pos) = wakers.iter().position(|(w_id, _)| *w_id == id) {
                    wakers.remove(pos);
                    self.receiver.waker_count.fetch_sub(1, Ordering::Release);
                }
            }
        }
    }
}

unsafe impl<T: Send> Send for HybridSender<T> {}
unsafe impl<T: Send> Send for HybridReceiver<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    #[test]
    fn test_recv_future_waker_cleanup_on_drop() {
        use std::task::{RawWaker, RawWakerVTable, Waker};

        fn dummy_raw_waker() -> RawWaker {
            fn clone_raw(_: *const ()) -> RawWaker {
                dummy_raw_waker()
            }
            fn wake_raw(_: *const ()) {}
            fn wake_by_ref_raw(_: *const ()) {}
            fn drop_raw(_: *const ()) {}
            static VTABLE: RawWakerVTable =
                RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);
            RawWaker::new(std::ptr::null(), &VTABLE)
        }

        let (_tx, rx) = HybridChannel::<i32>::new(4);
        let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
        let mut cx = Context::from_waker(&waker);

        {
            let mut fut = rx.recv_async();
            let mut pinned = Pin::new(&mut fut);
            assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);

            // Waker should be registered
            let wakers = rx.async_wakers.lock().unwrap();
            assert_eq!(wakers.len(), 1);
        }

        // After dropping the future, the waker list must be cleaned up
        let wakers = rx.async_wakers.lock().unwrap();
        assert_eq!(wakers.len(), 0);
    }

    #[test]
    fn test_hybrid_channel_lost_wakeup() {
        use std::task::{RawWaker, RawWakerVTable, Waker};

        fn dummy_raw_waker() -> RawWaker {
            fn clone_raw(_: *const ()) -> RawWaker {
                dummy_raw_waker()
            }
            fn wake_raw(_: *const ()) {}
            fn wake_by_ref_raw(_: *const ()) {}
            fn drop_raw(_: *const ()) {}
            static VTABLE: RawWakerVTable =
                RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);
            RawWaker::new(std::ptr::null(), &VTABLE)
        }

        let (tx, rx) = HybridChannel::<i32>::new(4);
        let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
        let mut cx = Context::from_waker(&waker);

        let mut fut = rx.recv_async();
        let mut pinned = Pin::new(&mut fut);

        // 1. Initial poll registers the waker. waker_count becomes 1.
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);
        assert_eq!(rx.waker_count.load(Ordering::Relaxed), 1);

        // 2. Sender sends an item, which drains the wakers list and wakes the future.
        // waker_count becomes 0.
        tx.send(42).unwrap();
        assert_eq!(rx.waker_count.load(Ordering::Relaxed), 0);

        // 3. Consume the item externally, so the ring buffer is empty again.
        assert_eq!(rx.try_recv().unwrap(), 42);

        // 4. Poll the future again. This should re-register the waker under the same ID.
        // If the bug is present, waker_count will remain 0. If fixed, it becomes 1.
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);
        assert_eq!(rx.waker_count.load(Ordering::Relaxed), 1);
    }
}
