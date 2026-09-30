//! Buffer replacement under the resize gate: growth, shrink, and reclamation.

use super::super::reclaim::{DequeReclaimPolicy, DequeReclaimState, SharedEpochReclaim};
use super::capacity::DequeCapacity;
use super::inner::ChaseLevInner;
use super::storage::Array;
use std::sync::atomic::Ordering;

impl<T, P> ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    pub(super) fn resize(&self) {
        let _resize_gate = self.resize_gate.claim(|| {});

        let old_array_ptr = self.array.load(Ordering::Relaxed);
        // SAFETY: non-null and owner-only (`resize` is reached only from `push`);
        // the reclaim guard and resize gate keep the buffer live and free of
        // in-flight thief accesses.
        let old_array = unsafe { &*old_array_ptr };
        // Growth has no rejection channel. Once doubling can no longer form a
        // valid allocation layout, follow the allocator's unrecoverable
        // resource-exhaustion policy instead of panicking through safe `push`.
        let new_capacity = old_array
            .capacity()
            .checked_mul(2)
            .and_then(|capacity| DequeCapacity::<T>::try_from(capacity).ok())
            .unwrap_or_else(|| std::process::abort())
            .get();

        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        let new_array = Box::new(Array::new(new_capacity, b));

        let len = b.wrapping_sub(t);
        for i in 0..len {
            // SAFETY: `i < len = bottom - top`, so slot `t + i` is an initialized,
            // live element being relocated into the fresh (distinct) buffer; the
            // bitwise copy moves ownership without running a destructor.
            unsafe {
                old_array.copy_slot_to(&new_array, t.wrapping_add(i));
            }
        }

        let new_array_ptr = Box::into_raw(new_array);
        self.array.store(new_array_ptr, Ordering::Release);

        let mut retired_arrays = self
            .retired_arrays
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        retired_arrays.push(old_array_ptr);
    }

    /// Replaces a grown buffer with one of `capacity` slots once the deque is
    /// small enough to fit, freeing the old buffer and every retired one.
    ///
    /// Owner-only, reached through the `&mut` owner endpoint, so no `push` or
    /// `pop` runs concurrently. Claiming the resize gate drains every thief:
    /// each steal and each `capacity` read holds an admission for the whole
    /// span in which it can load the array pointer or a retired buffer, so once
    /// the claim is held no thread other than the owner can reach any buffer,
    /// and the old ones are freed here rather than retired. A thief admitted
    /// after the claim drops loads the replacement. The gate protocol is the
    /// one `tests/loom_chase_lev_resize_gate.rs` model-checks for `resize`.
    ///
    /// Returns whether storage was replaced. The deque keeps one slot of
    /// headroom beyond its live length (`push` grows at `capacity - 1`), so the
    /// replacement never grows again on the next push.
    pub(super) fn shrink_to(&self, capacity: DequeCapacity<T>) -> bool {
        let target = capacity.get();
        // SAFETY: only the owner replaces the array and the caller is the
        // owner, so the pointer is live and stable until this call replaces it.
        if unsafe { &*self.array.load(Ordering::Relaxed) }.capacity() <= target {
            return false;
        }

        let _resize_gate = self.resize_gate.claim(|| {});

        let old_array_ptr = self.array.load(Ordering::Relaxed);
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        let len = b.wrapping_sub(t);
        let fits = usize::try_from(len).is_ok_and(|len| len < target - 1);
        if !fits {
            return false;
        }

        let new_array = Box::new(Array::new(target, b));
        // SAFETY: non-null and owner-replaced only; the gate claim excludes
        // every other reader.
        let old_array = unsafe { &*old_array_ptr };
        for i in 0..len {
            // SAFETY: `i < len = bottom - top`, so slot `t + i` is an initialized,
            // live element relocated into the fresh (distinct) buffer; the
            // bitwise copy moves ownership without running a destructor.
            unsafe {
                old_array.copy_slot_to(&new_array, t.wrapping_add(i));
            }
        }
        self.array
            .store(Box::into_raw(new_array), Ordering::Release);

        let mut retired_arrays = self
            .retired_arrays
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        retired_arrays.push(old_array_ptr);
        for array_ptr in retired_arrays.drain(..) {
            // SAFETY: each pointer is a `Box::into_raw` origin, retired exactly
            // once; the gate claim proves no thread holds a reference into it,
            // and element ownership moved with the bitwise copies, so dropping
            // the `Array` releases only storage.
            unsafe {
                drop(Box::from_raw(array_ptr));
            }
        }
        true
    }
}

impl<T> ChaseLevInner<T, SharedEpochReclaim> {
    pub(super) fn try_reclaim_shared(&self) -> bool {
        if !self.reclaim.can_reclaim_shared() {
            return false;
        }

        // Closing the same gate used by resize prevents a thief from entering
        // after the zero-access observation and then loading a retired pointer.
        // Once the claim is held, the repeated reclaim-state check below covers
        // owner-side guards while this gate covers every thief-side guard.
        let _resize_gate = self.resize_gate.claim(|| {});

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
                    // SAFETY: active_accesses()==0 proves no thread holds a
                    // reference into this retired array; the pointer came
                    // from Box::into_raw at retirement.
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
