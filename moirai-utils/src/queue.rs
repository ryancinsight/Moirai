//! Lock-free queues and ring buffers for high-performance data structures.
//!
//! This module provides efficient concurrent data structures optimized for
//! different usage patterns, from single-producer/single-consumer to
//! multi-producer/multi-consumer scenarios.

use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

#[cfg(feature = "std")]
use std::boxed::Box;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// A power-of-two sized ring buffer optimized for single-producer, single-consumer scenarios.
#[repr(align(64))]
pub struct RingBuffer<T> {
    data: Box<[Option<T>]>,
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
            .map(|_| None)
            .collect::<std::vec::Vec<_>>()
            .into_boxed_slice();

        #[cfg(not(feature = "std"))]
        let data = (0..capacity)
            .map(|_| None)
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

        // Safety: We've checked that the slot is available
        unsafe {
            let slot = &mut *(self.data.as_ptr().add(tail) as *mut Option<T>);
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

        // Safety: We've checked that there's an item available
        let item = unsafe {
            let slot = &mut *(self.data.as_ptr().add(head) as *mut Option<T>);
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

/// A lock-free, multi-producer, multi-consumer queue using a linked list structure.
pub struct LockFreeQueue<T> {
    head: AtomicPtr<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}

struct Node<T> {
    data: Option<T>,
    next: AtomicPtr<Node<T>>,
}

impl<T> Node<T> {
    fn new(data: Option<T>) -> Box<Self> {
        Box::new(Self {
            data,
            next: AtomicPtr::new(core::ptr::null_mut()),
        })
    }
}

impl<T> LockFreeQueue<T> {
    /// Create a new lock-free queue.
    pub fn new() -> Self {
        let dummy = Node::new(None);
        let dummy_ptr = Box::into_raw(dummy);

        Self {
            head: AtomicPtr::new(dummy_ptr),
            tail: AtomicPtr::new(dummy_ptr),
        }
    }

    /// Enqueue an item to the back of the queue.
    pub fn enqueue(&self, item: T) {
        let new_node = Box::into_raw(Node::new(Some(item)));

        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*tail).next.load(Ordering::Acquire) };

            if tail == self.tail.load(Ordering::Acquire) {
                if next.is_null() {
                    if unsafe {
                        (*tail).next.compare_exchange_weak(
                            next,
                            new_node,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                    }
                    .is_ok()
                    {
                        break;
                    }
                } else {
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
            }
        }

        let _ = self.tail.compare_exchange_weak(
            self.tail.load(Ordering::Acquire),
            new_node,
            Ordering::Release,
            Ordering::Relaxed,
        );
    }

    /// Try to dequeue an item from the front of the queue.
    /// Returns `None` if the queue is empty.
    pub fn try_dequeue(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*head).next.load(Ordering::Acquire) };

            if head == self.head.load(Ordering::Acquire) {
                if head == tail {
                    if next.is_null() {
                        return None; // Queue is empty
                    }
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                } else {
                    if next.is_null() {
                        continue;
                    }

                    let data = unsafe { (*next).data.take() };

                    if self
                        .head
                        .compare_exchange_weak(
                            head,
                            next,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        unsafe {
                            drop(Box::from_raw(head));
                        }
                        return data;
                    }
                }
            }
        }
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let next = unsafe { (*head).next.load(Ordering::Acquire) };

        head == tail && next.is_null()
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        while self.try_dequeue().is_some() {}

        // Clean up the dummy node
        let head = self.head.load(Ordering::Acquire);
        if !head.is_null() {
            unsafe {
                drop(Box::from_raw(head));
            }
        }
    }
}

// Safety: LockFreeQueue is safe to send between threads
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

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