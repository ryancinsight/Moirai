//! Lock-free queues and ring buffers for high-performance data structures.
//!
//! This module provides efficient concurrent data structures optimized for
//! different usage patterns, from single-producer/single-consumer to
//! multi-producer/multi-consumer scenarios.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(feature = "std")]
use std::boxed::Box;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// A power-of-two sized ring buffer optimized for single-producer, single-consumer scenarios.
#[repr(align(64))]
pub struct RingBuffer<T> {
    data: Box<[UnsafeCell<Option<T>>]>,
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity (must be a power of 2).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be a power of 2");
        assert!(capacity > 0, "Capacity must be greater than 0");

        #[cfg(feature = "std")]
        let data = (0..capacity)
            .map(|_| UnsafeCell::new(None))
            .collect::<std::vec::Vec<_>>()
            .into_boxed_slice();

        #[cfg(not(feature = "std"))]
        let data = (0..capacity)
            .map(|_| UnsafeCell::new(None))
            .collect::<alloc::vec::Vec<_>>()
            .into_boxed_slice();

        Self {
            data,
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Get the capacity of the ring buffer.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if the ring buffer is empty.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }

    /// Check if the ring buffer is full.
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (tail + 1) & self.mask == head
    }

    /// Get the current size of the ring buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (tail.wrapping_sub(head)) & self.mask
    }

    /// Try to push an item to the ring buffer.
    /// Returns `Err(item)` if the buffer is full.
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Acquire);
        let next_tail = (tail + 1) & self.mask;
        let head = self.head.load(Ordering::Acquire);

        if next_tail == head {
            return Err(item); // Buffer is full
        }

        // Safety: We've checked that the slot is available.
        // Accessing the interior of UnsafeCell under shared reference is safe.
        unsafe {
            let slot = &mut *self.data[tail].get();
            *slot = Some(item);
        }

        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    /// Try to pop an item from the ring buffer.
    /// Returns `None` if the buffer is empty.
    pub fn try_pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        if head == tail {
            return None; // Buffer is empty
        }

        // Safety: We've checked that there's an item available.
        // Accessing the interior of UnsafeCell under shared reference is safe.
        let item = unsafe {
            let slot = &mut *self.data[head].get();
            slot.take()
        };

        let next_head = (head + 1) & self.mask;
        self.head.store(next_head, Ordering::Release);

        item
    }

    /// Clear all items from the ring buffer.
    #[allow(clippy::redundant_pattern_matching)]
    pub fn clear(&self) {
        while let Some(_) = self.try_pop() {
            // Items are dropped automatically
        }
    }
}

// Safety: RingBuffer is safe to send between threads
unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

#[cfg(feature = "std")]
use std::collections::VecDeque;

#[cfg(not(feature = "std"))]
use alloc::collections::VecDeque;

#[repr(align(64))]
struct SpinLock<T> {
    lock: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
    const fn new(data: T) -> Self {
        Self {
            lock: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    fn lock(&self) -> SpinLockGuard<'_, T> {
        let mut backoff: usize = 1;
        loop {
            // Read-before-CAS: check first without writing to avoid cache line bouncing
            if !self.lock.load(Ordering::Relaxed)
                && self
                    .lock
                    .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
            {
                return SpinLockGuard { lock: self };
            }

            for _ in 0..backoff {
                core::hint::spin_loop();
            }

            if backoff < 64 {
                backoff = backoff.saturating_mul(2);
            }

            #[cfg(feature = "std")]
            {
                if backoff >= 64 {
                    std::thread::yield_now();
                    backoff = 1; // Reset backoff after yielding
                }
            }
        }
    }
}

struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> core::ops::Deref for SpinLockGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> core::ops::DerefMut for SpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.lock.store(false, Ordering::Release);
    }
}

/// A thread-safe, multi-producer, multi-consumer queue.
pub struct LockFreeQueue<T> {
    inner: SpinLock<VecDeque<T>>,
}

impl<T> LockFreeQueue<T> {
    /// Create a new queue.
    pub fn new() -> Self {
        Self {
            inner: SpinLock::new(VecDeque::new()),
        }
    }

    /// Enqueue an item to the back of the queue.
    pub fn enqueue(&self, item: T) {
        self.inner.lock().push_back(item);
    }

    /// Try to dequeue an item from the front of the queue.
    /// Returns `None` if the queue is empty.
    pub fn try_dequeue(&self) -> Option<T> {
        self.inner.lock().pop_front()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let buffer = RingBuffer::<i32>::new(4);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        buffer.try_push(1).unwrap();
        buffer.try_push(2).unwrap();
        assert_eq!(buffer.len(), 2);

        assert_eq!(buffer.try_pop(), Some(1));
        assert_eq!(buffer.try_pop(), Some(2));
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_full() {
        let buffer = RingBuffer::<i32>::new(2);

        buffer.try_push(1).unwrap();
        assert!(buffer.try_push(2).is_err()); // Should be full after 1 item
    }

    #[test]
    fn test_lock_free_queue_basic() {
        let queue = LockFreeQueue::<i32>::new();
        assert!(queue.is_empty());

        queue.enqueue(1);
        queue.enqueue(2);
        assert!(!queue.is_empty());

        assert_eq!(queue.try_dequeue(), Some(1));
        assert_eq!(queue.try_dequeue(), Some(2));
        assert_eq!(queue.try_dequeue(), None);
        assert!(queue.is_empty());
    }
}
