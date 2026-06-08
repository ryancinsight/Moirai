//! Cache-padded ring-buffer backing store for work-stealing deques.

use crate::platform::*;
use core::cell::UnsafeCell;

/// Padding helper to ensure cache line alignment
#[repr(align(64))]
pub(super) struct CachePadded<T> {
    pub(super) value: T,
}

pub(super) struct Buffer<T> {
    /// Capacity mask (capacity - 1 for fast modulo)
    pub(super) mask: usize,
    /// Storage for tasks
    pub(super) storage: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

impl<T> Buffer<T> {
    pub(super) fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        let storage = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            mask: capacity - 1,
            storage,
        }
    }

    pub(super) unsafe fn get(&self, index: usize) -> T {
        let slot = &*self.storage[index & self.mask].get();
        slot.assume_init_read()
    }

    pub(super) unsafe fn put(&self, index: usize, value: T) {
        let slot = &mut *self.storage[index & self.mask].get();
        slot.write(value);
    }

    pub(super) fn capacity(&self) -> usize {
        self.storage.len()
    }
}
