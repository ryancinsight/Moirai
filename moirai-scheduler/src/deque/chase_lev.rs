use super::reclaim::{DeferredReclaim, DequeReclaimPolicy, DequeReclaimState, SharedEpochReclaim};
use moirai_core::CacheAligned;
use std::{
    cell::Cell,
    marker::PhantomData,
    mem::MaybeUninit,
    sync::{
        atomic::{AtomicIsize, AtomicPtr, Ordering},
        Arc, Mutex,
    },
};

mod storage;
use storage::Array;

pub(crate) const MIN_DEQUE_CAPACITY: usize = 16;
const MAX_BATCH_STEAL: usize = 16;

// Compile-time guarantee of the false-sharing fix: wrapping `bottom`/`top` in
// `CacheAligned` forces the whole deque to ≥64-byte alignment, so two deques in
// a priority array (`[ChaseLevDeque<_>; N]`) can never share a cache line.
// Alignment is independent of `T`, so `u8` is a representative witness.
const _: () = assert!(core::mem::align_of::<ChaseLevInner<u8, DeferredReclaim>>() >= 64);

// ── StealResult ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum StealResult<T> {
    Success(T),
    Empty,
    Retry,
}

/// Allocation-free ownership container returned by a batch steal.
pub struct StolenBatch<T> {
    items: [MaybeUninit<T>; MAX_BATCH_STEAL],
    next: usize,
    len: usize,
}

impl<T> Iterator for StolenBatch<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }
        let index = self.next;
        self.next += 1;
        // SAFETY: `[next, len)` is initialized and advancing `next` transfers
        // this slot exactly once.
        Some(unsafe { self.items[index].assume_init_read() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for StolenBatch<T> {}

impl<T> Drop for StolenBatch<T> {
    fn drop(&mut self) {
        for item in &mut self.items[self.next..self.len] {
            // SAFETY: `[next, len)` is the initialized, unconsumed tail.
            unsafe { item.assume_init_drop() };
        }
    }
}

// ── ChaseLevDeque ─────────────────────────────────────────────────────────────

/// The unique bottom-side endpoint of a Chase-Lev work-stealing deque.
///
/// This endpoint is `Send`, but neither `Sync` nor `Clone`; therefore safe code
/// cannot create two concurrent push/pop owners. Use [`Self::stealer`] to
/// create cloneable top-side endpoints.
///
/// ```compile_fail
/// use moirai_scheduler::ChaseLevDeque;
/// let owner = ChaseLevDeque::<usize>::new(16);
/// owner.steal();
/// ```
pub struct ChaseLevDeque<T, P = DeferredReclaim>
where
    P: DequeReclaimPolicy,
{
    pub(crate) inner: Arc<ChaseLevInner<T, P>>,
    not_sync: PhantomData<Cell<()>>,
}

/// A cloneable top-side endpoint of a Chase-Lev work-stealing deque.
///
/// ```compile_fail
/// use moirai_scheduler::ChaseLevDeque;
/// let owner = ChaseLevDeque::<usize>::new(16);
/// let mut stealer = owner.stealer();
/// stealer.push(1);
/// ```
pub struct ChaseLevStealer<T, P = DeferredReclaim>
where
    P: DequeReclaimPolicy,
{
    pub(crate) inner: Arc<ChaseLevInner<T, P>>,
}

pub(crate) struct ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    // `bottom` is written only by the owning worker (push/pop); `top` is
    // CAS'd by thieves (steal). Co-locating them on one cache line makes every
    // steal invalidate the owner's `bottom` line and vice versa, so each is
    // isolated to its own 64-byte line. This also forces the whole struct to
    // 64-byte alignment, eliminating false sharing between adjacent deques in
    // the per-worker priority array (`[ChaseLevDeque<_>; PRIORITY_LEVELS]`).
    pub(crate) bottom: CacheAligned<AtomicIsize>,
    pub(crate) top: CacheAligned<AtomicIsize>,
    array: AtomicPtr<Array<T>>,
    retired_arrays: Mutex<Vec<*mut Array<T>>>,
    pub(crate) reclaim: P::State,
    policy: PhantomData<P>,
}

impl<T, P> ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    fn new(initial_capacity: usize) -> Self {
        let capacity = initial_capacity.next_power_of_two().max(MIN_DEQUE_CAPACITY);
        let array = Box::new(Array::new(capacity));

        Self {
            bottom: CacheAligned::new(AtomicIsize::new(0)),
            top: CacheAligned::new(AtomicIsize::new(0)),
            array: AtomicPtr::new(Box::into_raw(array)),
            retired_arrays: Mutex::new(Vec::new()),
            reclaim: P::State::default(),
            policy: PhantomData,
        }
    }

    fn push(&self, item: T) {
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

    fn pop(&self) -> Option<T> {
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

    fn steal(&self) -> StealResult<T> {
        let _guard = self.reclaim.enter();
        let t = self.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if b.wrapping_sub(t) > 0 {
            let array_ptr = self.array.load(Ordering::Acquire);
            let array = unsafe { &*array_ptr };

            // SAFETY: read-before-CAS is the canonical Chase-Lev steal protocol.
            // On CAS failure, `mem::forget(value)` prevents the destructor from
            // running on this speculative copy — the CAS winner will read and
            // own the slot independently.
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

    fn steal_batch(&self) -> StealResult<StolenBatch<T>> {
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
            .compare_exchange_weak(
                t,
                t.wrapping_add(n as isize),
                Ordering::SeqCst,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            return StealResult::Success(StolenBatch {
                items,
                next: 0,
                len: n,
            });
        }

        StealResult::Retry
    }

    fn len(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        b.wrapping_sub(t).max(0) as usize
    }

    fn is_empty(&self) -> bool {
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
}

impl<T> ChaseLevInner<T, SharedEpochReclaim> {
    fn try_reclaim_shared(&self) -> bool {
        if !self.reclaim.can_reclaim_shared() {
            return false;
        }

        let mut retired = self
            .retired_arrays
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if retired.is_empty() {
            return false;
        }

        if self.reclaim.active_accesses() == 0 {
            for array_ptr in retired.drain(..) {
                if !array_ptr.is_null() {
                    unsafe {
                        drop(Box::from_raw(array_ptr));
                    }
                }
            }
            true
        } else {
            false
        }
    }
}

impl<T, P> Drop for ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    fn drop(&mut self) {
        // `.0.get_mut()` reaches the inner atomic: `CacheAligned` has its own
        // inherent `get_mut` that would otherwise shadow `AtomicIsize::get_mut`.
        let top = *self.top.0.get_mut();
        let bottom = *self.bottom.0.get_mut();
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

unsafe impl<T, P> Send for ChaseLevInner<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Send,
{
}

unsafe impl<T, P> Sync for ChaseLevInner<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Sync,
{
}

impl<T, P> ChaseLevDeque<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
{
    /// Creates an empty deque with at least `initial_capacity` slots.
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            inner: Arc::new(ChaseLevInner::new(initial_capacity)),
            not_sync: PhantomData,
        }
    }

    /// Creates a cloneable top-side stealing endpoint.
    pub fn stealer(&self) -> ChaseLevStealer<T, P> {
        ChaseLevStealer {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Pushes an item at the owner-only bottom side.
    pub fn push(&mut self, item: T) {
        self.inner.push(item);
    }

    /// Pops an item from the owner-only bottom side.
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// Returns the current advisory length.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the deque is observably empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn retired_array_count(&self) -> usize {
        self.inner.retired_arrays.lock().unwrap().len()
    }
}

impl<T> ChaseLevDeque<T, SharedEpochReclaim>
where
    T: Send,
{
    /// Reclaims retired arrays if no endpoint operation is active.
    pub fn try_reclaim_shared(&self, _policy: SharedEpochReclaim) -> bool {
        self.inner.try_reclaim_shared()
    }
}

impl<T, P> Clone for ChaseLevStealer<T, P>
where
    P: DequeReclaimPolicy,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T, P> ChaseLevStealer<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
{
    /// Steals one item from the top side.
    pub fn steal(&self) -> StealResult<T> {
        self.inner.steal()
    }

    /// Steals an allocation-free, panic-safe batch from the top side.
    pub fn steal_batch(&self) -> StealResult<StolenBatch<T>> {
        self.inner.steal_batch()
    }

    /// Returns the current advisory length.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the deque is observably empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
