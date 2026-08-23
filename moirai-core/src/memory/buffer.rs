#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::allocator::CacheAlignedAllocator;

/// Zero-copy ring buffer with unified memory management.
///
/// Overflow beyond the ring capacity is the caller's concern:
/// `UnifiedChannel` layers its own overflow queue on top of this buffer.
pub struct UnifiedRingBuffer<T> {
    /// Cache-aligned buffer storage
    buffer: NonNull<MaybeUninit<T>>,
    /// Buffer capacity (always power of 2)
    capacity: usize,
    /// Mask for fast modulo operations
    mask: usize,
    /// Producer position
    head: AtomicUsize,
    /// Consumer position
    tail: AtomicUsize,
    write_lock: Mutex<()>,
    read_lock: Mutex<()>,
}

impl<T> UnifiedRingBuffer<T> {
    /// Create a new unified ring buffer with specified capacity
    pub fn new(capacity: usize) -> Option<Self> {
        let capacity = capacity.next_power_of_two().max(2);
        let buffer = CacheAlignedAllocator::allocate::<MaybeUninit<T>>(capacity)?;

        Some(Self {
            buffer,
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            write_lock: Mutex::new(()),
            read_lock: Mutex::new(()),
        })
    }

    /// Try to push an item using zero-copy semantics
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let _guard = self.write_lock.lock().unwrap();
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) & self.mask;
        let tail = self.tail.load(Ordering::Acquire);

        if next_head == tail {
            // Buffer is full
            return Err(item);
        }

        // SAFETY: the write lock makes this thread the sole producer; the
        // fullness check (`next_head != tail`) guarantees the masked slot is
        // outside the pending-consumer window, and the slot is
        // uninitialized until this `ptr::write`.
        unsafe {
            let slot = self.buffer.as_ptr().add(head & self.mask);
            ptr::write((*slot).as_mut_ptr(), item);
        }

        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Try to pop an item using zero-copy semantics
    pub fn try_pop(&self) -> Option<T> {
        let _guard = self.read_lock.lock().unwrap();
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            // Buffer is empty
            return None;
        }

        // SAFETY: the read lock makes this thread the sole consumer; the
        // emptiness check (`tail != head`) guarantees the masked slot was
        // published by a producer and holds an initialized value that this
        // `ptr::read` moves out exactly once before `tail` advances.
        let item = unsafe {
            let slot = self.buffer.as_ptr().add(tail & self.mask);
            ptr::read((*slot).as_ptr())
        };

        self.tail.store((tail + 1) & self.mask, Ordering::Release);
        Some(item)
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head.wrapping_sub(tail)) & self.mask
    }
}

impl<T> Drop for UnifiedRingBuffer<T> {
    fn drop(&mut self) {
        // Clean up remaining items
        #[allow(clippy::redundant_pattern_matching)]
        while let Some(_) = self.try_pop() {
            // Items are dropped automatically
        }

        // SAFETY: remaining items were drained above, so no live values are
        // leaked; the pointer, type, and count match the allocating call
        // exactly and no reference survives into `drop`.
        unsafe {
            CacheAlignedAllocator::deallocate(self.buffer, self.capacity);
        }
    }
}

// SAFETY: values transfer between threads through the ring, so `T` must be
// `Send`; the raw pointer never yields references.
unsafe impl<T: Send> Send for UnifiedRingBuffer<T> {}
// SAFETY: shared access is serialized by the per-side mutexes and
// head/tail atomics; no `&T` aliasing is ever exposed, so `T: Send`
// suffices.
unsafe impl<T: Send> Sync for UnifiedRingBuffer<T> {}
