//! Result slots for scoped indexed reductions.

use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, Ordering},
};

pub(crate) struct ReduceSlots<T> {
    slots: Box<[ReduceSlot<T>]>,
}

struct ReduceSlot<T> {
    value: UnsafeCell<MaybeUninit<T>>,
    initialized: AtomicBool,
}

// Safety: each slot is written by exactly one scheduled chunk and read by the
// parent after the scoped completion counter reaches zero. `T: Send` is
// required because values cross worker-thread boundaries.
unsafe impl<T: Send> Sync for ReduceSlot<T> {}

impl<T> ReduceSlots<T> {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            slots: (0..len)
                .map(|_| ReduceSlot::new())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub(crate) fn write(&self, index: usize, value: T) {
        self.slots[index].write(value);
    }

    pub(crate) fn reduce<F>(&self, identity: T, reduce: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self.slots.iter().fold(identity, |accumulator, slot| {
            if let Some(value) = slot.take() {
                reduce(accumulator, value)
            } else {
                accumulator
            }
        })
    }
}

impl<T> Drop for ReduceSlots<T> {
    fn drop(&mut self) {
        for slot in &self.slots {
            let _ = slot.take();
        }
    }
}

impl<T> ReduceSlot<T> {
    fn new() -> Self {
        Self {
            value: UnsafeCell::new(MaybeUninit::uninit()),
            initialized: AtomicBool::new(false),
        }
    }

    fn write(&self, value: T) {
        // Safety: a chunk owns exactly one slot index. No other worker writes
        // the same slot, and the parent reads only after scoped completion.
        unsafe {
            (*self.value.get()).write(value);
        }
        self.initialized.store(true, Ordering::Release);
    }

    fn take(&self) -> Option<T> {
        if self.initialized.swap(false, Ordering::AcqRel) {
            // Safety: initialized=true is stored only after `write` initializes
            // the slot. The swap gives the caller unique read ownership.
            Some(unsafe { (*self.value.get()).assume_init_read() })
        } else {
            None
        }
    }
}
