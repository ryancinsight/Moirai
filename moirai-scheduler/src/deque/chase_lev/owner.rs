//! Owner-side operations: `push` and `pop` at the bottom of the deque.

use super::super::reclaim::{DequeReclaimPolicy, DequeReclaimState};
use super::inner::ChaseLevInner;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::steal_outcome::MAX_BATCH_STEAL;
use std::sync::atomic::Ordering;

impl<T, P> ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    pub(super) fn push(&self, item: T) {
        let _guard = self.reclaim.enter();
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Acquire);

        let array_ptr = self.array.load(Ordering::Relaxed);
        // SAFETY: the array pointer is never null after construction, and this
        // owner holds a reclaim guard (`_guard`), so `resize` cannot free the
        // buffer while it is borrowed here. Owner-only access needs no acquire.
        let array = unsafe { &*array_ptr };

        if b.wrapping_sub(t) >= array.capacity() as isize - 1 {
            self.resize();
        }

        // Re-load: `resize` may have installed a larger buffer above.
        let array_ptr = self.array.load(Ordering::Relaxed);
        // SAFETY: as above — non-null, guard-protected, owner-only.
        let array = unsafe { &*array_ptr };

        // The generation claim waits for any thief that still owns the previous
        // occupant of this wrapped slot. It makes the following write disjoint
        // from every in-flight read without allocating per-item nodes.
        array.claim_for_write(b);

        // SAFETY: the generation claim makes this slot owner-exclusive and the
        // slot is uninitialized for generation `b` — `Array::write`'s
        // precondition.
        unsafe {
            array.write(b, item);
        }
        array.publish(b);

        self.bottom.store(b.wrapping_add(1), Ordering::Release);
    }

    pub(super) fn pop(&self) -> Option<T> {
        let _guard = self.reclaim.enter();
        let b = self.bottom.load(Ordering::Relaxed).wrapping_sub(1);
        let array_ptr = self.array.load(Ordering::Relaxed);
        // SAFETY: non-null after construction and guard-protected against a
        // concurrent `resize` free; `pop` is owner-only.
        let array = unsafe { &*array_ptr };

        self.bottom.store(b, Ordering::Relaxed);

        // Morrison-Afek fence-free pop optimization on TSO (x86/x86_64)
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let t = self.top.load(Ordering::Relaxed);
            if b.wrapping_sub(t) >= MAX_BATCH_STEAL as isize {
                if array.claim(b) {
                    // SAFETY: the generation claim makes this initialized slot
                    // owner-exclusive.
                    let item = unsafe { array.read(b) };
                    array.publish(b);
                    return Some(item);
                }
                self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
                return None;
            }
        }

        std::sync::atomic::fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Relaxed);

        if b.wrapping_sub(t) > 0 {
            if array.claim(b) {
                // SAFETY: the generation claim makes this initialized slot
                // owner-exclusive.
                let item = unsafe { array.read(b) };
                array.publish(b);
                return Some(item);
            }
            self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
            return None;
        }

        if b.wrapping_sub(t) == 0 {
            if !array.claim(t) {
                self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
                return None;
            }
            if self
                .top
                .compare_exchange(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
                // SAFETY: the generation claim and last-element CAS make this
                // initialized slot owner-exclusive.
                let item = unsafe { array.read(b) };
                array.release(b);
                return Some(item);
            }

            array.publish(t);
            self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
            return None;
        }

        self.bottom.store(b.wrapping_add(1), Ordering::Relaxed);
        None
    }
}
