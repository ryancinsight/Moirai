use super::reclaim::{DequeReclaimPolicy, DequeReclaimState, QuiescentReclaim, SharedEpochReclaim};
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

pub(crate) const MIN_DEQUE_CAPACITY: usize = 16;
const MAX_BATCH_STEAL: usize = 16;

// ── Array ─────────────────────────────────────────────────────────────────────

pub(crate) struct Array<T> {
    capacity: usize,
    mask: usize,
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

#[derive(Debug, Clone, PartialEq)]
pub enum StealResult<T> {
    Success(T),
    Empty,
    Retry,
}

// ── ChaseLevDeque ─────────────────────────────────────────────────────────────

/// A lock-free work-stealing deque implementation based on the Chase-Lev algorithm.
pub struct ChaseLevDeque<T, P = QuiescentReclaim>
where
    P: DequeReclaimPolicy,
{
    pub(crate) bottom: AtomicIsize,
    pub(crate) top: AtomicIsize,
    array: AtomicPtr<Array<T>>,
    pub(crate) retired_arrays: Mutex<Vec<*mut Array<T>>>,
    pub(crate) reclaim: P::State,
    policy: PhantomData<P>,
}

impl<T, P> ChaseLevDeque<T, P>
where
    P: DequeReclaimPolicy,
{
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

    pub fn push(&self, item: T) {
        let _guard = self.reclaim.enter();
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Acquire);

        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };

        if b.wrapping_sub(t) >= array.capacity() as isize - 1 {
            self.resize();
        }

        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };

        unsafe {
            array.write(b, item);
        }

        self.bottom.store(b.wrapping_add(1), Ordering::Release);
    }

    pub fn pop(&self) -> Option<T> {
        let _guard = self.reclaim.enter();
        let b = self.bottom.load(Ordering::Relaxed).wrapping_sub(1);
        let array_ptr = self.array.load(Ordering::Relaxed);
        let array = unsafe { &*array_ptr };

        self.bottom.store(b, Ordering::Relaxed);

        // Morrison-Afek fence-free pop optimization on TSO (x86/x86_64)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let t = self.top.load(Ordering::Relaxed);
            if b.wrapping_sub(t) >= MAX_BATCH_STEAL as isize {
                return Some(unsafe { array.read(b) });
            }
        }

        std::sync::atomic::fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Relaxed);

        if b.wrapping_sub(t) > 0 {
            return Some(unsafe { array.read(b) });
        }

        if b.wrapping_sub(t) == 0 {
            if self
                .top
                .compare_exchange_weak(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
                return Some(unsafe { array.read(b) });
            }

            self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
            return None;
        }

        self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
        None
    }

    pub fn steal(&self) -> StealResult<T> {
        let _guard = self.reclaim.enter();
        let t = self.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if b.wrapping_sub(t) > 0 {
            let array_ptr = self.array.load(Ordering::Acquire);
            let array = unsafe { &*array_ptr };

            // SAFETY: We read before the CAS; if the CAS fails a concurrent
            // stealer or the owner claimed this slot, so we must not use
            // `value`.  `mem::forget` below prevents a double-free.
            let value = unsafe { array.read(t) };

            if self
                .top
                .compare_exchange_weak(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return StealResult::Success(value);
            }

            std::mem::forget(value);
            return StealResult::Retry;
        }

        StealResult::Empty
    }

    pub fn steal_batch_with<F>(&self, mut f: F) -> StealResult<T>
    where
        F: FnMut(T),
    {
        let _guard = self.reclaim.enter();
        let t = self.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        let len = b.wrapping_sub(t);
        if len <= 0 {
            return StealResult::Empty;
        }

        let n = ((len / 2).max(1) as usize).min(MAX_BATCH_STEAL);

        let array_ptr = self.array.load(Ordering::Acquire);
        let array = unsafe { &*array_ptr };

        let mut items: [MaybeUninit<T>; MAX_BATCH_STEAL] =
            [const { MaybeUninit::uninit() }; MAX_BATCH_STEAL];
        for (i, slot) in items.iter_mut().enumerate().take(n) {
            slot.write(unsafe { array.read(t.wrapping_add(i as isize)) });
        }

        if self
            .top
            .compare_exchange_weak(t, t.wrapping_add(n as isize), Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            let first_item = unsafe { items[0].assume_init_read() };
            for slot in items.iter().take(n).skip(1) {
                f(unsafe { slot.assume_init_read() });
            }
            return StealResult::Success(first_item);
        }

        // CAS lost: the CAS winner also performed assume_init_read on the same
        // slots (they observed the same `t`); both hold bitwise copies.  The
        // winner legitimately owns theirs.  `MaybeUninit<T>` does NOT invoke
        // `T::drop()` on scope exit, so `items` goes out of scope here with no
        // destructor calls — correct, because any destructor call would
        // double-decrement Arc ref-counts / double-free alongside the winner.
        StealResult::Retry
    }

    pub fn len(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        b.wrapping_sub(t).max(0) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn resize(&self) {
        let old_array_ptr = self.array.load(Ordering::Relaxed);
        let old_array = unsafe { &*old_array_ptr };
        let new_capacity = old_array.capacity() * 2;
        let new_array = Box::new(Array::new(new_capacity));

        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);

        let len = b.wrapping_sub(t);
        for i in 0..len {
            unsafe {
                old_array.copy_slot_to(&new_array, t.wrapping_add(i));
            }
        }

        let new_array_ptr = Box::into_raw(new_array);
        self.array.store(new_array_ptr, Ordering::Release);

        let mut retired_arrays = self.retired_arrays.lock().unwrap();
        retired_arrays.push(old_array_ptr);
    }

    pub fn reclaim_memory(&mut self, _policy: P) {
        self.deallocate_retired_arrays();
    }

    fn deallocate_retired_arrays(&self) {
        let mut retired_arrays = self.retired_arrays.lock().unwrap();
        for array_ptr in retired_arrays.drain(..) {
            unsafe {
                drop(Box::from_raw(array_ptr));
            }
        }
    }
}

impl<T> ChaseLevDeque<T, SharedEpochReclaim> {
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
            let len = bottom.wrapping_sub(top);
            for i in 0..len {
                let index = top.wrapping_add(i);
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

unsafe impl<T, P> Send for ChaseLevDeque<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Send,
{}

unsafe impl<T, P> Sync for ChaseLevDeque<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Sync,
{}
