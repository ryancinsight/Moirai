//! Cache-padded ring-buffer backing store for work-stealing deques.

use crate::platform::*;
use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::ptr::NonNull;

#[allow(unused_imports)]
#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc, handle_alloc_error};
#[allow(unused_imports)]
#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc, handle_alloc_error};

pub(super) struct Buffer<T> {
    /// Capacity mask (capacity - 1 for fast modulo)
    pub(super) mask: usize,
    /// Capacity
    pub(super) capacity: usize,
    /// Storage for tasks
    pub(super) storage: NonNull<UnsafeCell<MaybeUninit<T>>>,
}

unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}

impl<T> Buffer<T> {
    pub(super) fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        let layout = Layout::array::<UnsafeCell<MaybeUninit<T>>>(capacity)
            .expect("Invalid layout for Buffer");

        let raw_ptr = unsafe {
            #[cfg(feature = "mnemosyne")]
            {
                use core::alloc::GlobalAlloc;
                mnemosyne::Mnemosyne.alloc(layout)
            }
            #[cfg(not(feature = "mnemosyne"))]
            {
                alloc(layout)
            }
        };

        if raw_ptr.is_null() {
            handle_alloc_error(layout);
        }

        let storage = NonNull::new(raw_ptr as *mut UnsafeCell<MaybeUninit<T>>).unwrap();

        Self {
            mask: capacity - 1,
            capacity,
            storage,
        }
    }

    pub(super) unsafe fn get(&self, index: usize) -> T {
        let ptr = self.storage.as_ptr().add(index & self.mask) as *mut MaybeUninit<T>;
        (*ptr).assume_init_read()
    }

    pub(super) unsafe fn put(&self, index: usize, value: T) {
        let ptr = self.storage.as_ptr().add(index & self.mask) as *mut MaybeUninit<T>;
        (*ptr).write(value);
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<UnsafeCell<MaybeUninit<T>>>(self.capacity)
            .expect("Invalid layout for Buffer");
        unsafe {
            #[cfg(feature = "mnemosyne")]
            {
                use core::alloc::GlobalAlloc;
                mnemosyne::Mnemosyne.dealloc(self.storage.as_ptr() as *mut u8, layout);
            }
            #[cfg(not(feature = "mnemosyne"))]
            {
                dealloc(self.storage.as_ptr() as *mut u8, layout);
            }
        }
    }
}
