use std::{
    alloc::Layout,
    cell::UnsafeCell,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    sync::atomic::{AtomicIsize, Ordering},
};

pub(super) struct Array<T> {
    capacity: usize,
    mask: usize,
    ptr: NonNull<UnsafeCell<MaybeUninit<T>>>,
    states: Box<[AtomicIsize]>,
}

// SAFETY: allocation ownership moves with `Array`; slots are accessed only by
// the deque protocol, which transfers values rather than exposing references.
unsafe impl<T: Send> Send for Array<T> {}
// SAFETY: slot mutation is synchronized by the owner/stealer index protocol.
unsafe impl<T: Sync> Sync for Array<T> {}

impl<T> Array<T> {
    pub(super) fn new(capacity: usize, first_index: isize) -> Self {
        assert!(capacity.is_power_of_two());
        let layout = Layout::array::<UnsafeCell<MaybeUninit<T>>>(capacity)
            .expect("invariant: deque capacity has a valid allocation layout");

        // SAFETY: `layout` is non-zero and valid for the slot array; allocation
        // failure is handled before constructing `NonNull`.
        let raw_ptr = unsafe {
            #[cfg(feature = "mnemosyne")]
            {
                use std::alloc::GlobalAlloc;
                mnemosyne::Mnemosyne.alloc(layout)
            }
            #[cfg(not(feature = "mnemosyne"))]
            {
                std::alloc::alloc(layout)
            }
        };
        if raw_ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let ptr = NonNull::new(raw_ptr.cast::<UnsafeCell<MaybeUninit<T>>>())
            .expect("invariant: allocation failure was handled above");

        let offset = (first_index as usize) & (capacity - 1);
        let base = first_index.wrapping_sub(offset as isize);
        let states = (0..capacity)
            .map(|slot| {
                let slot = slot as isize;
                let index = if slot as usize >= offset {
                    base.wrapping_add(slot)
                } else {
                    base.wrapping_add(capacity as isize).wrapping_add(slot)
                };
                AtomicIsize::new(index)
            })
            .collect();

        Self {
            capacity,
            mask: capacity - 1,
            ptr,
            states,
        }
    }

    pub(super) fn claim(&self, index: isize) -> bool {
        let slot = (index as usize) & self.mask;
        self.states[slot]
            .compare_exchange(index, !index, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    pub(super) fn publish(&self, index: isize) {
        let slot = (index as usize) & self.mask;
        self.states[slot].store(index, Ordering::Release);
    }

    pub(super) fn release(&self, index: isize) {
        let slot = (index as usize) & self.mask;
        self.states[slot].store(
            index.wrapping_add(self.capacity as isize),
            Ordering::Release,
        );
    }

    pub(super) fn claim_for_write(&self, index: isize) {
        while !self.claim(index) {
            std::hint::spin_loop();
        }
    }

    #[cfg(test)]
    pub(super) fn reset_states(&self, first_index: isize) {
        let offset = (first_index as usize) & self.mask;
        let base = first_index.wrapping_sub(offset as isize);
        for (slot, state) in self.states.iter().enumerate() {
            let slot = slot as isize;
            let index = if slot as usize >= offset {
                base.wrapping_add(slot)
            } else {
                base.wrapping_add(self.capacity as isize).wrapping_add(slot)
            };
            state.store(index, Ordering::Relaxed);
        }
    }

    pub(super) unsafe fn write(&self, index: isize, item: T) {
        let index = (index as usize) & self.mask;
        // SAFETY: masking keeps the pointer within the allocation; the unique
        // owner writes this unpublished slot exactly once.
        let cell = unsafe { self.ptr.as_ptr().add(index) };
        // SAFETY: the caller proves this slot is uninitialized and owner-only.
        unsafe { (*(*cell).get()).write(item) };
    }

    pub(super) unsafe fn read(&self, index: isize) -> T {
        let index = (index as usize) & self.mask;
        // SAFETY: masking keeps the pointer within the allocation.
        let cell = unsafe { self.ptr.as_ptr().add(index) };
        // SAFETY: the caller proves ownership of this initialized slot.
        unsafe { (*(*cell).get()).assume_init_read() }
    }

    pub(super) unsafe fn copy_slot_to(&self, target: &Self, index: isize) {
        let source_index = (index as usize) & self.mask;
        let target_index = (index as usize) & target.mask;
        // SAFETY: both masked indices are within their respective allocations.
        let source = unsafe { self.ptr.as_ptr().add(source_index) };
        // SAFETY: both masked indices are within their respective allocations.
        let target_cell = unsafe { target.ptr.as_ptr().add(target_index) };
        // SAFETY: source and target allocations are distinct and the target
        // slot is uninitialized during owner-only resize.
        unsafe {
            ptr::copy_nonoverlapping(
                (*(*source).get()).as_ptr(),
                (*(*target_cell).get()).as_mut_ptr(),
                1,
            )
        };
        let state = self.states[source_index].load(Ordering::Acquire);
        target.states[target_index].store(state, Ordering::Release);
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for Array<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<UnsafeCell<MaybeUninit<T>>>(self.capacity)
            .expect("invariant: allocated deque capacity retains a valid layout");
        // SAFETY: this `Array` uniquely owns the allocation and uses its
        // original layout. Element destruction is owned by the deque indices.
        unsafe {
            #[cfg(feature = "mnemosyne")]
            {
                use std::alloc::GlobalAlloc;
                mnemosyne::Mnemosyne.dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
            }
            #[cfg(not(feature = "mnemosyne"))]
            {
                std::alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), layout);
            }
        }
    }
}
