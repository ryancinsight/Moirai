//! Memory-mapped ring buffer for zero-copy operations.

use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

use super::error::{ZeroCopyError, ZeroCopyResult};

/// Memory-mapped ring buffer for zero-copy operations.
///
/// Safety: uses raw allocation for performance while maintaining
/// acquire/release ordering and bounds checks.
pub struct MemoryMappedRing<T> {
    buffer: AtomicPtr<T>,
    capacity: usize,
    producer_cursor: AtomicUsize,
    consumer_cursor: AtomicUsize,
    producer_lock: AtomicBool,
    consumer_lock: AtomicBool,
    buffer_size: usize,
    _element_size: usize,
    closed: AtomicBool,
}

impl<T> MemoryMappedRing<T> {
    /// Creates a new memory-mapped ring buffer with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - The maximum number of elements the ring buffer can hold
    ///
    /// # Returns
    /// A new `MemoryMappedRing` instance or a `ZeroCopyError` if allocation fails
    pub fn new(capacity: usize) -> ZeroCopyResult<Self> {
        if !capacity.is_power_of_two() || capacity == 0 {
            return Err(ZeroCopyError::InvalidBufferSize);
        }

        let element_size = mem::size_of::<T>();
        let alignment = mem::align_of::<T>();
        let buffer_size = capacity
            .checked_mul(element_size)
            .ok_or(ZeroCopyError::InvalidBufferSize)?;

        let layout = std::alloc::Layout::from_size_align(buffer_size, alignment)
            .map_err(|_| ZeroCopyError::AlignmentError)?;

        let buffer = unsafe {
            #[cfg(feature = "mnemosyne")]
            {
                use core::alloc::GlobalAlloc;
                let ptr = mnemosyne::Mnemosyne.alloc(layout) as *mut T;
                if ptr.is_null() {
                    return Err(ZeroCopyError::MemoryMapFailed);
                }
                ptr
            }
            #[cfg(not(feature = "mnemosyne"))]
            {
                let ptr = std::alloc::alloc(layout) as *mut T;
                if ptr.is_null() {
                    return Err(ZeroCopyError::MemoryMapFailed);
                }
                ptr
            }
        };

        Ok(Self {
            buffer: AtomicPtr::new(buffer),
            capacity,
            producer_cursor: AtomicUsize::new(0),
            consumer_cursor: AtomicUsize::new(0),
            producer_lock: AtomicBool::new(false),
            consumer_lock: AtomicBool::new(false),
            buffer_size,
            _element_size: element_size,
            closed: AtomicBool::new(false),
        })
    }

    /// Sends a value through the ring buffer using zero-copy semantics.
    ///
    /// # Arguments
    /// * `value` - The value to send
    ///
    /// # Returns
    /// `Ok(())` on success, or `Err((value, error))` if the send fails
    pub fn send_zero_copy(&self, value: T) -> Result<(), (T, ZeroCopyError)> {
        while self
            .producer_lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }

        if self.closed.load(Ordering::Acquire) {
            self.producer_lock.store(false, Ordering::Release);
            return Err((value, ZeroCopyError::Closed));
        }
        let p = self.producer_cursor.load(Ordering::Relaxed);
        let c = self.consumer_cursor.load(Ordering::Acquire);
        if p.wrapping_sub(c) >= self.capacity {
            self.producer_lock.store(false, Ordering::Release);
            return Err((value, ZeroCopyError::Full));
        }

        let ptr = self.buffer.load(Ordering::Relaxed);
        let idx = p & (self.capacity - 1);
        unsafe {
            ptr::write(ptr.add(idx), value);
        }
        self.producer_cursor
            .store(p.wrapping_add(1), Ordering::Release);

        self.producer_lock.store(false, Ordering::Release);
        Ok(())
    }

    /// Receives a value from the ring buffer using zero-copy semantics.
    ///
    /// # Returns
    /// The received value or a `ZeroCopyError` if no value is available
    pub fn recv_zero_copy(&self) -> ZeroCopyResult<T> {
        while self
            .consumer_lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }

        let c = self.consumer_cursor.load(Ordering::Relaxed);
        let p = self.producer_cursor.load(Ordering::Acquire);
        if c == p {
            let err = if self.closed.load(Ordering::Acquire) {
                ZeroCopyError::Closed
            } else {
                ZeroCopyError::Empty
            };
            self.consumer_lock.store(false, Ordering::Release);
            return Err(err);
        }
        let ptr = self.buffer.load(Ordering::Relaxed);
        let idx = c & (self.capacity - 1);
        let value = unsafe { ptr::read(ptr.add(idx)) };
        self.consumer_cursor
            .store(c.wrapping_add(1), Ordering::Release);

        self.consumer_lock.store(false, Ordering::Release);
        Ok(value)
    }

    /// Attempts to send a value without blocking.
    ///
    /// # Arguments
    /// * `value` - The value to send
    ///
    /// # Returns
    /// `Ok(())` on success, or `Err((value, error))` if the send fails
    pub fn try_send(&self, value: T) -> Result<(), (T, ZeroCopyError)> {
        self.send_zero_copy(value)
    }

    /// Attempts to receive a value without blocking.
    pub fn try_recv(&self) -> ZeroCopyResult<T> {
        self.recv_zero_copy()
    }

    /// Closes the ring buffer, preventing further operations.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    /// Returns true if the ring buffer has been closed.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Returns the current number of elements in the ring buffer.
    pub fn len(&self) -> usize {
        let p = self.producer_cursor.load(Ordering::Relaxed);
        let c = self.consumer_cursor.load(Ordering::Relaxed);
        p.wrapping_sub(c)
    }

    /// Check if the ring buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if the ring buffer is full
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Get the capacity of the ring buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for MemoryMappedRing<T> {
    fn drop(&mut self) {
        let ptr = self.buffer.load(Ordering::Relaxed);
        if !ptr.is_null() {
            let layout =
                std::alloc::Layout::from_size_align(self.buffer_size, mem::align_of::<T>())
                    .unwrap();
            unsafe {
                let c = self.consumer_cursor.load(Ordering::Relaxed);
                let p = self.producer_cursor.load(Ordering::Relaxed);
                let len = p.wrapping_sub(c);
                for i in 0..len {
                    let idx = (c.wrapping_add(i)) & (self.capacity - 1);
                    ptr::drop_in_place(ptr.add(idx));
                }
                #[cfg(feature = "mnemosyne")]
                {
                    use core::alloc::GlobalAlloc;
                    mnemosyne::Mnemosyne.dealloc(ptr as *mut u8, layout);
                }
                #[cfg(not(feature = "mnemosyne"))]
                {
                    std::alloc::dealloc(ptr as *mut u8, layout);
                }
            }
        }
    }
}

unsafe impl<T: Send> Send for MemoryMappedRing<T> {}
unsafe impl<T: Send> Sync for MemoryMappedRing<T> {}
