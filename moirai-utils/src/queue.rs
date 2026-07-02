//! Lock-free queues and ring buffers for high-performance data structures.
//!
//! This module provides efficient concurrent data structures optimized for
//! different usage patterns, from single-producer/single-consumer to
//! multi-producer/multi-consumer scenarios.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

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

/// Default capacity for [`LockFreeQueue`]. Large enough to avoid backpressure
/// under normal scheduling load while bounding memory under adversarial
/// producer rates per the bounded-resource policy.
const DEFAULT_QUEUE_CAPACITY: usize = 65536;

/// A single slot in the bounded MPMC queue.
struct Slot<T> {
    /// Monotonic sequence number that distinguishes empty, full, and stale
    /// states without an ABA hazard.
    sequence: AtomicUsize,
    /// The slot's data. `None` when empty, `Some(item)` when filled.
    data: UnsafeCell<Option<T>>,
}

/// A bounded, genuinely lock-free multi-producer multi-consumer queue.
///
/// This is an array-based MPMC queue using per-slot sequence numbers (the
/// Vyukov algorithm). Producers and consumers operate through independent
/// atomic head/tail cursors and never acquire a mutex or spinlock. The
/// sequence-number protocol eliminates the ABA problem without tagged
/// pointers or epoch-based reclamation: slots are reused in place, so no
/// node allocation or deallocation occurs during enqueue/dequeue.
///
/// # Capacity
///
/// The queue is bounded. [`LockFreeQueue::new`] creates a queue with
/// `DEFAULT_QUEUE_CAPACITY` slots. [`LockFreeQueue::with_capacity`] allows
/// a custom power-of-two capacity. When the queue is full, [`enqueue`]
/// retries with exponential backoff (preserving the unblocked-sender
/// contract of the previous API), while [`try_enqueue`] returns `Err(item)`
/// immediately for callers that prefer explicit backpressure.
///
/// # Memory safety
///
/// Each slot's `Option<T>` is written by the producer (replacing `None` with
/// `Some(item)`) and taken by the consumer (replacing `Some(item)` with
/// `None`). The sequence-number protocol guarantees that only one thread
/// accesses a slot's data at a time: the producer writes between `sequence ==
/// pos` and `sequence == pos+1`, the consumer reads between `sequence ==
/// pos+1` and `sequence == pos+capacity`.
///
/// [`enqueue`]: LockFreeQueue::enqueue
/// [`try_enqueue`]: LockFreeQueue::try_enqueue
#[repr(align(64))]
pub struct LockFreeQueue<T> {
    buffer: Box<[Slot<T>]>,
    mask: usize,
    capacity: usize,
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
}

/// Cache-line padded wrapper to prevent false sharing between head and tail.
#[repr(align(64))]
struct CachePadded<X>(X);

impl<X> CachePadded<X> {
    const fn new(val: X) -> Self {
        Self(val)
    }
}

impl<X> core::ops::Deref for CachePadded<X> {
    type Target = X;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Safety: The sequence-number protocol ensures that each slot's data is
// accessed by at most one thread at a time: a producer writes between
// sequence == pos and sequence == pos+1; a consumer takes between
// sequence == pos+1 and sequence == pos+capacity. The head and tail atomics
// are independently advanced via CAS, so no global lock is needed. T: Send
// is sufficient because ownership of the value transfers between threads
// through the slot, never shared concurrently.
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T> LockFreeQueue<T> {
    /// Create a new queue with the default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_QUEUE_CAPACITY)
    }

    /// Create a new queue with a custom capacity (must be a power of 2).
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0 or not a power of 2.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");
        assert!(capacity.is_power_of_two(), "Capacity must be a power of 2");

        #[cfg(feature = "std")]
        let buffer: Box<[Slot<T>]> = (0..capacity)
            .map(|i| Slot {
                sequence: AtomicUsize::new(i),
                data: UnsafeCell::new(None),
            })
            .collect::<std::vec::Vec<_>>()
            .into_boxed_slice();

        #[cfg(not(feature = "std"))]
        let buffer: Box<[Slot<T>]> = (0..capacity)
            .map(|i| Slot {
                sequence: AtomicUsize::new(i),
                data: UnsafeCell::new(None),
            })
            .collect::<alloc::vec::Vec<_>>()
            .into_boxed_slice();

        Self {
            buffer,
            mask: capacity - 1,
            capacity,
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Try to enqueue an item without blocking.
    ///
    /// Returns `Ok(())` if the item was enqueued, or `Err(item)` if the
    /// queue is full. This is the lock-free fast path: no spinlock, no
    /// mutex, no retry loop.
    #[inline]
    pub fn try_enqueue(&self, item: T) -> Result<(), T> {
        let mut pos = self.tail.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq.wrapping_sub(pos) as isize;

            if diff == 0 {
                // Slot is empty: try to claim it by advancing tail.
                match self.tail.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: we successfully claimed this slot via CAS on
                        // tail. No other producer can claim the same slot
                        // position because tail has advanced past it. The
                        // sequence == pos invariant guarantees no consumer is
                        // reading this slot.
                        unsafe {
                            *slot.data.get() = Some(item);
                        }
                        slot.sequence.store(pos.wrapping_add(1), Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                // Queue is full: sequence lags behind tail, meaning all slots
                // between head and tail are occupied.
                return Err(item);
            } else {
                // Another producer advanced tail before us: reload and retry.
                pos = self.tail.load(Ordering::Relaxed);
            }
        }
    }

    /// Enqueue an item, retrying with exponential backoff if the queue is full.
    ///
    /// This preserves the unblocked-sender contract of the previous API: the
    /// call always eventually succeeds (assuming consumers make progress).
    /// The backoff path uses `core::hint::spin_loop` and, on std targets,
    /// `std::thread::yield_now` after heavy contention, but never acquires a
    /// global lock, so multiple producers can enqueue concurrently.
    #[inline]
    pub fn enqueue(&self, item: T) {
        let mut backoff: usize = 1;
        let mut item = Some(item);
        loop {
            match self.try_enqueue(item.take().expect("invariant: item present")) {
                Ok(()) => return,
                Err(returned) => {
                    item = Some(returned);
                    for _ in 0..backoff {
                        core::hint::spin_loop();
                    }
                    if backoff < 64 {
                        backoff = backoff.saturating_mul(2);
                    } else {
                        #[cfg(feature = "std")]
                        {
                            std::thread::yield_now();
                        }
                        backoff = 1;
                    }
                }
            }
        }
    }

    /// Try to dequeue an item from the front of the queue.
    /// Returns `None` if the queue is empty.
    ///
    /// This is the lock-free fast path: no spinlock, no mutex.
    #[inline]
    pub fn try_dequeue(&self) -> Option<T> {
        let mut pos = self.head.load(Ordering::Relaxed);
        loop {
            let slot = &self.buffer[pos & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq.wrapping_sub(pos.wrapping_add(1)) as isize;

            if diff == 0 {
                // Slot has data: try to claim it by advancing head.
                match self.head.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: we successfully claimed this slot via CAS on
                        // head. No other consumer can claim the same slot
                        // because head has advanced past it. The
                        // sequence == pos+1 invariant guarantees the producer
                        // has finished writing and no producer will write
                        // again until we set sequence to pos+capacity.
                        let item = unsafe { (*slot.data.get()).take() };
                        slot.sequence
                            .store(pos.wrapping_add(self.capacity), Ordering::Release);
                        return item;
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                // Queue is empty: sequence has not advanced past pos+1.
                return None;
            } else {
                // Another consumer advanced head before us: reload and retry.
                pos = self.head.load(Ordering::Relaxed);
            }
        }
    }

    /// Check if the queue is empty.
    ///
    /// This is a best-effort check: the queue may have items added or removed
    /// between this call and the next operation. It is safe to call
    /// concurrently with enqueue/dequeue.
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head == tail
    }

    /// Returns the queue capacity.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // Drain remaining items so their destructors run.
        while self.try_dequeue().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    #[cfg(feature = "std")]
    use std::sync::Arc;

    #[cfg(not(feature = "std"))]
    use alloc::sync::Arc;

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
        let queue = LockFreeQueue::<i32>::with_capacity(4);
        assert!(queue.is_empty());

        queue.enqueue(1);
        queue.enqueue(2);
        assert!(!queue.is_empty());

        assert_eq!(queue.try_dequeue(), Some(1));
        assert_eq!(queue.try_dequeue(), Some(2));
        assert_eq!(queue.try_dequeue(), None);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_lock_free_queue_wrap_around() {
        // Fill and drain multiple times to exercise slot reuse.
        let queue = LockFreeQueue::<i32>::with_capacity(4);
        for round in 0..16 {
            for i in 0..3 {
                queue.enqueue(round * 3 + i);
            }
            for i in 0..3 {
                assert_eq!(
                    queue.try_dequeue(),
                    Some(round * 3 + i),
                    "round {round}, item {i}"
                );
            }
            assert!(queue.try_dequeue().is_none(), "round {round} not empty");
        }
    }

    #[test]
    fn test_lock_free_queue_full_try_enqueue() {
        let queue = LockFreeQueue::<i32>::with_capacity(4);
        for i in 0..4 {
            queue.try_enqueue(i).unwrap();
        }
        // Now full.
        assert!(queue.try_enqueue(99).is_err());
        assert_eq!(queue.try_dequeue(), Some(0));
        // One slot freed.
        queue.try_enqueue(99).unwrap();
    }

    #[test]
    fn test_lock_free_queue_drop_runs_destructors() {
        struct DropCounter {
            counter: Arc<AtomicUsize>,
        }
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        {
            let queue = LockFreeQueue::<DropCounter>::with_capacity(4);
            for _ in 0..3 {
                queue.enqueue(DropCounter {
                    counter: Arc::clone(&counter),
                });
            }
            // Drop the queue without draining: destructors must run.
        }
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_lock_free_queue_concurrent_mpmc() {
        use std::thread;

        let queue = Arc::new(LockFreeQueue::<i32>::with_capacity(1024));
        let num_producers = 4;
        let num_consumers = 4;
        let items_per_producer = 1000;
        let total_items = num_producers * items_per_producer;

        let mut handles = Vec::new();

        for p in 0..num_producers {
            let q = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for i in 0..items_per_producer {
                    q.enqueue((p * items_per_producer + i) as i32);
                }
            }));
        }

        let consumed = Arc::new(AtomicUsize::new(0));
        for _ in 0..num_consumers {
            let q = Arc::clone(&queue);
            let c = Arc::clone(&consumed);
            handles.push(thread::spawn(move || {
                while c.load(Ordering::Relaxed) < total_items {
                    if q.try_dequeue().is_some() {
                        c.fetch_add(1, Ordering::Relaxed);
                    } else {
                        std::thread::yield_now();
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(consumed.load(Ordering::Relaxed), total_items);
    }
}
