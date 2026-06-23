//! Chase-Lev work-stealing deque and zero-copy variant.

use super::buffer::Buffer;
use crate::platform::*;
use moirai_utils::cache::CachePadded;

/// Chase-Lev work-stealing deque implementation (inspired by Rayon)
///
/// This is a highly optimized deque that allows:
/// - Single owner pushing/popping from one end
/// - Multiple stealers taking from the other end
/// - Minimal synchronization overhead
pub struct WorkStealingDeque<T> {
    /// Bottom index - owned by worker
    bottom: CachePadded<AtomicUsize>,
    /// Top index - accessed by stealers
    top: CachePadded<AtomicUsize>,
    /// Ring buffer for tasks
    buffer: CachePadded<AtomicPtr<Buffer<T>>>,
    /// Retired buffers to prevent use-after-free by concurrent stealers
    retired_buffers: Mutex<Vec<*mut Buffer<T>>>,
    _phantom: PhantomData<T>,
}

impl<T: Send> WorkStealingDeque<T> {
    /// Create a new work-stealing deque
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let buffer = Box::into_raw(Box::new(Buffer::new(capacity)));

        Self {
            bottom: CachePadded {
                value: AtomicUsize::new(0),
            },
            top: CachePadded {
                value: AtomicUsize::new(0),
            },
            buffer: CachePadded {
                value: AtomicPtr::new(buffer),
            },
            retired_buffers: Mutex::new(Vec::new()),
            _phantom: PhantomData,
        }
    }

    /// Push a task (owner only)
    pub fn push(&self, task: T) {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let top = self.top.value.load(Ordering::Acquire);
        let size = bottom.wrapping_sub(top);

        let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };

        // Check if resize needed
        if size >= buffer.mask {
            // Grow buffer by allocating a new one and copying existing elements
            let old_buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
            let new_capacity = old_buffer.capacity() * 2;
            let new_buffer = Box::into_raw(Box::new(Buffer::new(new_capacity)));
            let old_top = self.top.value.load(Ordering::Acquire);
            let old_bottom = bottom;
            // Copy existing range [old_top, old_bottom)
            for i in old_top..old_bottom {
                unsafe {
                    (*new_buffer).put(i, old_buffer.get(i));
                }
            }
            // Swap buffer pointer
            let old_ptr = self.buffer.value.swap(new_buffer, Ordering::Release);
            // Retire old buffer instead of dropping immediately to avoid Use-After-Free
            self.retired_buffers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(old_ptr);
        }

        let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
        unsafe {
            buffer.put(bottom, task);
        }

        // Release store to make task visible to stealers
        self.bottom
            .value
            .store(bottom.wrapping_add(1), Ordering::Release);
        fence(Ordering::SeqCst);
    }

    /// Pop a task (owner only)
    pub fn pop(&self) -> Option<T> {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let new_bottom = bottom.wrapping_sub(1);

        self.bottom.value.store(new_bottom, Ordering::Relaxed);

        // Synchronize with stealers
        fence(Ordering::SeqCst);

        let top = self.top.value.load(Ordering::Relaxed);

        if top <= new_bottom {
            // Non-empty
            let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
            // The buffer uses wrapping index arithmetic (get/put both apply a
            // power-of-two mask), so new_bottom is always a valid logical index.
            // A raw capacity check here is incorrect and would spuriously return
            // None when bottom wraps past the buffer size.
            let task = unsafe { buffer.get(new_bottom) };

            if top == new_bottom {
                // Last task - race with stealers
                if self
                    .top
                    .value
                    .compare_exchange(
                        top,
                        top.wrapping_add(1),
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    )
                    .is_err()
                {
                    // Lost race
                    self.bottom.value.store(bottom, Ordering::Relaxed);
                    return None;
                }
                self.bottom.value.store(bottom, Ordering::Relaxed);
            }

            Some(task)
        } else {
            // Empty
            self.bottom.value.store(bottom, Ordering::Relaxed);
            None
        }
    }

    /// Steal a task (can be called by multiple threads)
    pub fn steal(&self) -> Option<T> {
        loop {
            let top = self.top.value.load(Ordering::Acquire);

            // Synchronize with owner
            fence(Ordering::SeqCst);

            let bottom = self.bottom.value.load(Ordering::Acquire);

            if top >= bottom {
                return None; // Empty
            }

            let buffer = unsafe { &*self.buffer.value.load(Ordering::Acquire) };

            // Try to increment top
            if self
                .top
                .value
                .compare_exchange(
                    top,
                    top.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                // SAFETY: We won the CAS, establishing exclusive ownership of this slot.
                // No other stealer or owner can access it, making this read safe and data-race-free.
                let task = unsafe { buffer.get(top) };
                return Some(task);
            }

            // CAS failed, retry
        }
    }

    /// Get the current size estimate
    pub fn len(&self) -> usize {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let top = self.top.value.load(Ordering::Relaxed);
        bottom.wrapping_sub(top)
    }
}

impl<T> Drop for WorkStealingDeque<T> {
    fn drop(&mut self) {
        let bottom = *self.bottom.value.get_mut();
        let top = *self.top.value.get_mut();
        let buffer_ptr = *self.buffer.value.get_mut();
        if !buffer_ptr.is_null() {
            let buffer = unsafe { Box::from_raw(buffer_ptr) };
            for i in top..bottom {
                unsafe {
                    drop(buffer.get(i));
                }
            }
        }

        // H-8 fix: use `get_mut()` (available on `&mut self`) to bypass the
        // `Mutex` entirely rather than `lock()` which returns Err on poisoning
        // — a poisoned lock would silently skip the drain and leak all retired
        // buffer pointers.
        for ptr in self
            .retired_buffers
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}

// Safety: Tasks are Send
unsafe impl<T: Send> Send for WorkStealingDeque<T> {}
unsafe impl<T: Send> Sync for WorkStealingDeque<T> {}

/// Improved zero-copy work-stealing deque
///
/// This implementation minimizes allocations and uses atomic operations
/// for lock-free stealing between workers.
pub struct ZeroCopyWorkStealingDeque<T> {
    /// Bottom of the deque (owner's end)
    bottom: CachePadded<AtomicUsize>,
    /// Top of the deque (stealer's end)
    top: CachePadded<AtomicUsize>,
    /// Current buffer
    buffer: CachePadded<AtomicPtr<Buffer<T>>>,
    /// Retired buffers to prevent use-after-free by concurrent stealers
    retired_buffers: Mutex<Vec<*mut Buffer<T>>>,
}

impl<T> ZeroCopyWorkStealingDeque<T> {
    /// Create a new deque with initial capacity
    pub fn new(capacity: usize) -> Self {
        let buffer = Box::new(Buffer::new(capacity));
        Self {
            bottom: CachePadded {
                value: AtomicUsize::new(0),
            },
            top: CachePadded {
                value: AtomicUsize::new(0),
            },
            buffer: CachePadded {
                value: AtomicPtr::new(Box::into_raw(buffer)),
            },
            retired_buffers: Mutex::new(Vec::new()),
        }
    }

    /// Push a task (owner only)
    pub fn push(&self, task: T) {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let top = self.top.value.load(Ordering::Acquire);
        let size = bottom.wrapping_sub(top);

        let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };

        // Check if resize is needed
        if size >= buffer.capacity() {
            // Grow buffer with zero-copy transfer
            self.grow(bottom, top, buffer);
        }

        let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
        unsafe {
            buffer.put(bottom, task);
        }

        // Release store to make the push visible to stealers
        self.bottom
            .value
            .store(bottom.wrapping_add(1), Ordering::Release);
    }

    /// Pop a task (owner only)
    pub fn pop(&self) -> Option<T> {
        let bottom = self.bottom.value.load(Ordering::Relaxed);
        let new_bottom = bottom.wrapping_sub(1);

        // Relaxed store is safe - only owner modifies bottom
        self.bottom.value.store(new_bottom, Ordering::Relaxed);

        // Synchronize with stealers
        std::sync::atomic::fence(Ordering::SeqCst);

        let top = self.top.value.load(Ordering::Relaxed);

        if top <= new_bottom {
            // Non-empty queue
            let buffer = unsafe { &*self.buffer.value.load(Ordering::Relaxed) };
            let task = unsafe { buffer.get(new_bottom) };

            if top == new_bottom {
                // Last element - race with stealers
                if self
                    .top
                    .value
                    .compare_exchange(
                        top,
                        top.wrapping_add(1),
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    )
                    .is_err()
                {
                    // Lost the race
                    self.bottom.value.store(bottom, Ordering::Relaxed);
                    return None;
                }
                self.bottom.value.store(bottom, Ordering::Relaxed);
            }

            Some(task)
        } else {
            // Empty queue
            self.bottom.value.store(bottom, Ordering::Relaxed);
            None
        }
    }

    /// Steal a task (stealers)
    pub fn steal(&self) -> Option<T> {
        let top = self.top.value.load(Ordering::Acquire);

        // Synchronize with owner
        std::sync::atomic::fence(Ordering::SeqCst);

        let bottom = self.bottom.value.load(Ordering::Acquire);

        if top < bottom {
            let buffer = unsafe { &*self.buffer.value.load(Ordering::Acquire) };

            // Try to increment top
            if self
                .top
                .value
                .compare_exchange(
                    top,
                    top.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                // SAFETY: We won the CAS, establishing exclusive ownership of this slot.
                // No other stealer or owner can access it, making this read safe and data-race-free.
                let task = unsafe { buffer.get(top) };
                Some(task)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Grow the buffer with zero-copy transfer
    fn grow(&self, bottom: usize, top: usize, old_buffer: &Buffer<T>) {
        let new_capacity = old_buffer.capacity() * 2;
        let new_buffer = Box::new(Buffer::new(new_capacity));

        // Zero-copy transfer of elements
        for i in top..bottom {
            unsafe {
                let value = old_buffer.get(i);
                new_buffer.put(i, value);
            }
        }

        let new_buffer_ptr = Box::into_raw(new_buffer);
        let old_buffer_ptr = self.buffer.value.swap(new_buffer_ptr, Ordering::Release);

        // Retire the old buffer pointer safely to avoid Use-After-Free
        self.retired_buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(old_buffer_ptr);
    }
}

impl<T> Drop for ZeroCopyWorkStealingDeque<T> {
    fn drop(&mut self) {
        let bottom = *self.bottom.value.get_mut();
        let top = *self.top.value.get_mut();
        let buffer_ptr = *self.buffer.value.get_mut();
        if !buffer_ptr.is_null() {
            let buffer = unsafe { Box::from_raw(buffer_ptr) };
            for i in top..bottom {
                unsafe {
                    drop(buffer.get(i));
                }
            }
        }
        // H-8 fix: use `get_mut()` to bypass the poisoned-mutex leak path.
        for ptr in self
            .retired_buffers
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}
