//! Shared state of a Chase-Lev deque: indices, storage pointer, and lifecycle.

use super::super::reclaim::{DeferredReclaim, DequeReclaimPolicy, DequeReclaimState};
use super::capacity::DequeCapacity;
use super::{
    gate::{ResizeGate, StealAccessGuard},
    storage::Array,
};
use moirai_core::CacheAligned;
use std::{
    marker::PhantomData,
    sync::{
        Mutex,
        atomic::{AtomicIsize, AtomicPtr, Ordering},
    },
};

// Compile-time guarantee of the false-sharing fix: wrapping `bottom`/`top` in
// `CacheAligned` forces the whole deque to ≥64-byte alignment, so two deques in
// a priority array (`[ChaseLevDeque<_>; N]`) can never share a cache line.
// Alignment is independent of `T`, so `u8` is a representative witness.
const _: () = assert!(core::mem::align_of::<ChaseLevInner<u8, DeferredReclaim>>() >= 64);

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
    pub(super) array: AtomicPtr<Array<T>>,
    pub(super) retired_arrays: Mutex<Vec<*mut Array<T>>>,
    pub(super) resize_gate: ResizeGate,
    pub(crate) reclaim: P::State,
    policy: PhantomData<P>,
}

impl<T, P> ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    pub(super) fn enter_steal_access(&self) -> StealAccessGuard<'_> {
        self.resize_gate.enter(|| {}, || {})
    }

    pub(super) fn new(capacity: DequeCapacity<T>) -> Self {
        let capacity = capacity.get();
        let array = Box::new(Array::new(capacity, 0));

        Self {
            bottom: CacheAligned::new(AtomicIsize::new(0)),
            top: CacheAligned::new(AtomicIsize::new(0)),
            array: AtomicPtr::new(Box::into_raw(array)),
            retired_arrays: Mutex::new(Vec::new()),
            resize_gate: ResizeGate::new(),
            reclaim: P::State::default(),
            policy: PhantomData,
        }
    }

    pub(super) fn len(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        b.wrapping_sub(t).max(0) as usize
    }

    /// Slot count of the currently published array.
    pub(super) fn capacity(&self) -> usize {
        // A stealer reads this from any thread. Entering the resize gate and the
        // reclaim guard, as a steal does, keeps the array it loads from being
        // retired and then freed by `try_reclaim_shared` before the read.
        let _access = self.enter_steal_access();
        let _guard = self.reclaim.enter();
        // SAFETY: `array` always points at a live `Array<T>` published by the
        // owner. The gate excludes a concurrent resize and the guard excludes a
        // concurrent reclaim, so the loaded array is neither retired nor freed
        // until this read ends.
        unsafe { &*self.array.load(Ordering::Acquire) }.capacity()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
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
            // SAFETY: exclusive &mut self in drop; array_ptr originated from
            // Box::into_raw and no concurrent access can exist during drop.
            let array = unsafe { Box::from_raw(array_ptr) };
            let len = bottom.wrapping_sub(top);
            for i in 0..len {
                let index = top.wrapping_add(i);
                // SAFETY: indices span exactly the live window top..bottom;
                // read() moves each stored value out for its one true drop.
                unsafe {
                    drop(array.read(index));
                }
            }
        }

        let retired_arrays = self
            .retired_arrays
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for array_ptr in retired_arrays.drain(..) {
            // SAFETY: exclusive &mut self in drop outranks any quiescent
            // stealer references; retired pointers are Box::into_raw origins
            // freed exactly once here.
            unsafe {
                drop(Box::from_raw(array_ptr));
            }
        }
    }
}

// SAFETY: deque endpoints transfer `T` by ownership (owner push/pop,
// stealer take), so values never yield cross-thread references.
unsafe impl<T, P> Send for ChaseLevInner<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Send,
{
}

// SAFETY: all shared access is arbitrated by the top/bottom index protocol
// and the reclamation policy's epoch accounting; slot contents are touched
// only by the thread that won the claiming CAS.
unsafe impl<T, P> Sync for ChaseLevInner<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
    P::State: Sync,
{
}
