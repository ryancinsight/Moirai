//! Chase-Lev lock-free work-stealing deque.
//!
//! Provides [`ChaseLevDeque<T, P>`] — a dynamically-resizing, lock-free deque
//! parameterised by a [`DequeReclaimPolicy`] — and the associated
//! [`StealResult<T>`] enum.

use crate::reclaim::{DequeReclaimPolicy, DequeReclaimState, QuiescentReclaim, SharedEpochReclaim};
use std::{
    alloc::Layout,
    cell::UnsafeCell,
    marker::PhantomData,
    mem::MaybeUninit,
    ptr,
    ptr::NonNull,
    sync::{
        atomic::{AtomicIsize, AtomicPtr, Ordering},
        Mutex,
    },
};

/// Minimum capacity for Chase-Lev deque to ensure efficient operations.
pub(crate) const MIN_DEQUE_CAPACITY: usize = 16;
const MAX_BATCH_STEAL: usize = 16;

// ── Array ─────────────────────────────────────────────────────────────────────

/// Contiguous inline storage backing a [`ChaseLevDeque`].
pub(crate) struct Array<T> {
    /// Capacity of this array (always power of 2)
    capacity: usize,
    /// Mask for fast modulo operations
    mask: usize,
    /// Raw pointer to the allocated memory block
    ptr: NonNull<UnsafeCell<MaybeUninit<T>>>,
}

unsafe impl<T: Send> Send for Array<T> {}
unsafe impl<T: Sync> Sync for Array<T> {}

impl<T> Array<T> {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        let layout = Layout::array::<UnsafeCell<MaybeUninit<T>>>(capacity)
            .expect("Invalid layout for Array");

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

        let ptr = NonNull::new(raw_ptr as *mut UnsafeCell<MaybeUninit<T>>).unwrap();

        Self {
            capacity,
            mask: capacity - 1,
            ptr,
        }
    }

    pub(crate) unsafe fn write(&self, index: isize, item: T) {
        let idx = (index as usize) & self.mask;
        let cell_ptr = self.ptr.as_ptr().add(idx);
        (*(*cell_ptr).get()).write(item);
    }

    pub(crate) unsafe fn read(&self, index: isize) -> T {
        let idx = (index as usize) & self.mask;
        let cell_ptr = self.ptr.as_ptr().add(idx);
        (*(*cell_ptr).get()).assume_init_read()
    }

    pub(crate) unsafe fn copy_slot_to(&self, target: &Self, index: isize) {
        let source_idx = (index as usize) & self.mask;
        let target_idx = (index as usize) & target.mask;
        let source_cell = self.ptr.as_ptr().add(source_idx);
        let target_cell = target.ptr.as_ptr().add(target_idx);
        ptr::copy_nonoverlapping(
            (*(*source_cell).get()).as_ptr(),
            (*(*target_cell).get()).as_mut_ptr(),
            1,
        );
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for Array<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<UnsafeCell<MaybeUninit<T>>>(self.capacity)
            .expect("Invalid layout for Array");
        unsafe {
            #[cfg(feature = "mnemosyne")]
            {
                use std::alloc::GlobalAlloc;
                mnemosyne::Mnemosyne.dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
            #[cfg(not(feature = "mnemosyne"))]
            {
                std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

// ── StealResult ───────────────────────────────────────────────────────────────

/// Result of a steal operation.
#[derive(Debug, Clone, PartialEq)]
pub enum StealResult<T> {
    /// Successfully stole an item
    Success(T),
    /// Queue was empty
    Empty,
    /// Race condition occurred, should retry
    Retry,
}

// ── ChaseLevDeque ─────────────────────────────────────────────────────────────

/// A lock-free work-stealing deque implementation based on the Chase-Lev algorithm.
pub struct ChaseLevDeque<T, P = QuiescentReclaim>
where
    P: DequeReclaimPolicy,
{
    /// Bottom index (only modified by owner)
    bottom: AtomicIsize,
    /// Top index (modified by thieves)
    top: AtomicIsize,
    /// Array of task pointers
    array: AtomicPtr<Array<T>>,
    /// Retired arrays pending deallocation after quiescence.
    pub(crate) retired_arrays: Mutex<Vec<*mut Array<T>>>,
    /// Policy-specific reclamation state.
    pub(crate) reclaim: P::State,
    policy: PhantomData<P>,
}

impl<T, P> ChaseLevDeque<T, P>
where
    P: DequeReclaimPolicy,
{
    /// Create a new Chase-Lev deque with the specified initial capacity.
    pub fn new(initial_capacity: usize) -> Self {
        let capacity = initial_capacity.next_power_of_two().max(MIN_DEQUE_CAPACITY);
        let array = Box::new(Array::new(capacity));

        Self {
            bottom: AtomicIsize::new(0),
            top: AtomicIsize::new(0),
            array: AtomicPtr::new(Box::into_raw(array)),
            retired_arrays: Mutex::new(Vec::new()),
            reclaim: P::State::default(),
            policy: PhantomData,
        }
    }

    /// Push an item to the bottom of the deque (owner operation).
    pub fn push(&self, item: T) {
        let _guard = self.reclaim.enter();
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Acquire);

        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };

        // Check if we need to resize
        if b - t >= array.capacity() as isize - 1 {
            self.resize();
        }

        // Re-load array pointer after potential resize
        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };

        // Store the item inline before publishing the updated bottom index.
        unsafe {
            array.write(b, item);
        }

        // Release the item to thieves
        self.bottom.store(b + 1, Ordering::Release);
    }

    /// Pop an item from the bottom of the deque (owner operation).
    pub fn pop(&self) -> Option<T> {
        let _guard = self.reclaim.enter();
        let b = self.bottom.load(Ordering::Relaxed) - 1;
        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };

        self.bottom.store(b, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);

        let t = self.top.load(Ordering::Relaxed);

        if t < b {
            // More than one item: thieves can only claim from the top, so the
            // owner can read bottom directly.
            return Some(unsafe { array.read(b) });
        }

        if t == b {
            // Single last element: claim the index before moving the value.
            if self
                .top
                .compare_exchange_weak(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                self.bottom.store(b + 1, Ordering::Relaxed);
                return Some(unsafe { array.read(b) });
            }

            self.bottom.store(b + 1, Ordering::Relaxed);
            return None;
        }

        // Empty queue, restore bottom.
        self.bottom.store(b + 1, Ordering::Relaxed);
        None
    }

    /// Steal an item from the top of the deque (thief operation).
    pub fn steal(&self) -> StealResult<T> {
        let _guard = self.reclaim.enter();
        let t = self.top.load(Ordering::Acquire);
        // SeqCst fence between the top and bottom loads so a thief observes a
        // consistent (top, bottom) snapshot relative to a concurrent owner
        // `pop`. It pairs with the store-plus-SeqCst-fence in `pop` to impose one
        // total order over the two indices (Lê, Pop, Cohen & Nardelli,
        // "Correct and Efficient Work-Stealing for Weak Memory Models",
        // PPoPP 2013).
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if t < b {
            let array_ptr = self.array.load(Ordering::Acquire);
            let array = unsafe { &*array_ptr };

            // Read the value BEFORE claiming the slot, then claim with the CAS.
            // A successful CAS proves `top` never left `t`, hence the owner did
            // not advance `top` to make room for `capacity` further pushes, hence
            // it could not have lapped the ring and overwritten `array[t]`
            // between this read and the claim. Reading *after* the CAS admitted
            // exactly that wraparound race (and the missing fence admitted a torn
            // snapshot), duplicating one item and losing another under
            // multi-thief contention.
            let value = unsafe { array.read(t) };

            if self
                .top
                .compare_exchange_weak(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return StealResult::Success(value);
            }

            // Lost the claim: the slot still logically owns this value and the
            // winning consumer will read the same bytes. `value` is a bitwise
            // `ptr::read` copy, so forget it rather than drop it here — dropping
            // would double-free the slot's contents.
            std::mem::forget(value);
            return StealResult::Retry;
        }

        StealResult::Empty
    }

    /// Steal multiple items from this deque, passing all but the first one to the closure
    /// and returning the first one.
    pub fn steal_batch_with<F>(&self, mut f: F) -> StealResult<T>
    where
        F: FnMut(T),
    {
        let _guard = self.reclaim.enter();
        let t = self.top.load(Ordering::Acquire);
        // SeqCst fence for a consistent (top, bottom) snapshot — see `steal`.
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        let len = b - t;
        if len <= 0 {
            return StealResult::Empty;
        }

        let n = ((len / 2).max(1) as usize).min(MAX_BATCH_STEAL);

        let array_ptr = self.array.load(Ordering::Acquire);
        let array = unsafe { &*array_ptr };

        // Read the whole batch BEFORE claiming it (read-before-CAS; see `steal`
        // for the wraparound-safety argument). The batch is capped and buffered
        // on the stack so the steal path remains allocation-free.
        // Safety: an uninitialized `[MaybeUninit<T>; N]` is valid because
        // `MaybeUninit<T>` may hold uninitialized bytes.
        let mut items: [MaybeUninit<T>; MAX_BATCH_STEAL] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for (i, slot) in items.iter_mut().enumerate().take(n) {
            slot.write(unsafe { array.read(t + i as isize) });
        }

        if self
            .top
            .compare_exchange_weak(t, t + n as isize, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            // Safety: slots `0..n` were initialized by the loop above and are
            // each read exactly once after the CAS transfers logical ownership.
            let first_item = unsafe { items[0].assume_init_read() };
            for slot in items.iter().take(n).skip(1) {
                f(unsafe { slot.assume_init_read() });
            }
            return StealResult::Success(first_item);
        }

        // Lost the claim: the slots still own these values. Discard the
        // speculative `ptr::read` copies without running their destructors,
        // avoiding a double-free of the slot contents.
        StealResult::Retry
    }

    /// Get the current size of the deque.
    pub fn len(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        (b - t).max(0) as usize
    }

    /// Check if the deque is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resize the underlying array when it becomes full.
    fn resize(&self) {
        let old_array_ptr = self.array.load(Ordering::Relaxed);
        let old_array = unsafe { &*old_array_ptr };
        let new_capacity = old_array.capacity() * 2;
        let new_array = Box::new(Array::new(new_capacity));

        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);

        // Copy live elements to the new array. Retired arrays do not drop their
        // copied elements; global top/bottom ownership decides which copy is
        // later read or dropped from the current array.
        for i in t..b {
            unsafe {
                old_array.copy_slot_to(&new_array, i);
            }
        }

        // Atomically replace the array
        let new_array_ptr = Box::into_raw(new_array);
        self.array.store(new_array_ptr, Ordering::Release);

        // Push the old array into the list of arrays pending deallocation
        let mut retired_arrays = self.retired_arrays.lock().unwrap();
        retired_arrays.push(old_array_ptr);

        // Note: memory reclamation is deferred to an explicit safe point.
    }

    /// Deallocate retired backing arrays through an exclusive quiescent access path.
    pub fn reclaim_memory(&mut self, _policy: P) {
        self.deallocate_retired_arrays();
    }

    fn deallocate_retired_arrays(&self) {
        let mut retired_arrays = self.retired_arrays.lock().unwrap();
        for array_ptr in retired_arrays.drain(..) {
            unsafe {
                // Retired arrays may contain duplicated bytes copied into a
                // newer current array, so only the backing allocation is freed.
                drop(Box::from_raw(array_ptr));
            }
        }
    }
}

impl<T> ChaseLevDeque<T, SharedEpochReclaim> {
    /// Try to deallocate retired backing arrays while the deque remains shared.
    ///
    /// This succeeds only when no active push, pop, or steal operation is inside
    /// an array-access section.
    pub fn try_reclaim_shared(&self, _policy: SharedEpochReclaim) -> bool {
        if !self.reclaim.can_reclaim_shared() {
            return false;
        }

        self.deallocate_retired_arrays();
        true
    }
}

impl<T, P> Drop for ChaseLevDeque<T, P>
where
    P: DequeReclaimPolicy,
{
    fn drop(&mut self) {
        let top = *self.top.get_mut();
        let bottom = *self.bottom.get_mut();
        let array_ptr = *self.array.get_mut();

        if !array_ptr.is_null() {
            let array = unsafe { Box::from_raw(array_ptr) };
            for index in top..bottom {
                unsafe {
                    drop(array.read(index));
                }
            }
        }

        let retired_arrays = self
            .retired_arrays
            .get_mut()
            .expect("retired array mutex poisoned during deque drop");
        for array_ptr in retired_arrays.drain(..) {
            unsafe {
                drop(Box::from_raw(array_ptr));
            }
        }
    }
}

// Safety: ChaseLevDeque is thread-safe by design
unsafe impl<T, P> Send for ChaseLevDeque<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Send,
{
}

unsafe impl<T, P> Sync for ChaseLevDeque<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Sync,
{
}
