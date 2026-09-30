//! Thief-side operations: single and batch `steal` at the top of the deque.

use super::super::reclaim::{DequeReclaimPolicy, DequeReclaimState};
use super::inner::ChaseLevInner;
use super::steal_outcome::{MAX_BATCH_STEAL, StealResult, StolenBatch};
use std::{mem::MaybeUninit, sync::atomic::Ordering};

impl<T, P> ChaseLevInner<T, P>
where
    P: DequeReclaimPolicy,
{
    pub(super) fn steal(&self) -> StealResult<T> {
        let _access = self.enter_steal_access();
        let _guard = self.reclaim.enter();
        self.steal_within_access()
    }

    /// One steal, with the resize gate and the reclaim guard already held by the
    /// caller.
    ///
    /// The gate and the guard are per-*access*, not per-item: the gate makes
    /// `resize` wait, and the reclaim guard keeps the buffer live. Both hold for
    /// the whole call, so a caller taking them once and stealing repeatedly
    /// observes exactly the state one `steal` would — the array pointer cannot
    /// change under it, because `resize` publishes a new buffer only after the
    /// gate is empty. The `Acquire` load of the pointer is kept per item so this
    /// body remains identical to the single-item path.
    fn steal_within_access(&self) -> StealResult<T> {
        let t = self.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if b.wrapping_sub(t) > 0 {
            let array_ptr = self.array.load(Ordering::Acquire);
            // SAFETY: the `Acquire` load pairs with `resize`'s `Release` store, so
            // this is a live buffer; the caller's reclaim guard keeps it from
            // being freed while borrowed.
            let array = unsafe { &*array_ptr };

            if !array.claim(t) {
                return StealResult::Retry;
            }

            // A slot's state equals its index both while it holds an item and
            // while it is free, so this claim can succeed on a slot the owner's
            // fence-free pop already emptied and republished. The owner's
            // `bottom` store precedes that publish, and the claim acquired it,
            // so this load sees a `bottom` that no longer covers `t`.
            if self.bottom.load(Ordering::Acquire).wrapping_sub(t) <= 0 {
                array.publish(t);
                return StealResult::Retry;
            }

            if self
                .top
                .compare_exchange(t, t.wrapping_add(1), Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: the generation claim and successful CAS claim this
                // index against every other thief and the owner; the caller's
                // reclaim guard keeps the allocation live while the value moves.
                let value = unsafe { array.read(t) };
                array.release(t);
                return StealResult::Success(value);
            }

            array.publish(t);
            return StealResult::Retry;
        }

        StealResult::Empty
    }

    pub(super) fn steal_batch(&self) -> StealResult<StolenBatch<T>> {
        let mut items: [MaybeUninit<T>; MAX_BATCH_STEAL] =
            [const { MaybeUninit::uninit() }; MAX_BATCH_STEAL];
        let mut count = 0;
        let mut retry = false;

        // One gate entry and one reclaim guard for the batch, not one per item.
        // `enter_steal_access` is a sequentially-consistent increment and
        // decrement of a counter every thief shares, so per-item entry put up to 2 ×
        // MAX_BATCH_STEAL contended RMWs on one line to move at most
        // MAX_BATCH_STEAL elements.
        //
        // The cost is on the other side: `resize` spins until the counter
        // reaches zero, so it now waits behind a whole batch instead of a
        // single steal. The batch cannot block — every step below is a bounded
        // sequence of atomics with no wait on the owner — so the wait is
        // bounded by MAX_BATCH_STEAL steal attempts, and in exchange a batch no
        // longer stalls mid-flight when a resize opens between two of its items.
        let _access = self.enter_steal_access();
        let _guard = self.reclaim.enter();

        // Claim each element through the single-item protocol before moving it
        // out of storage. A single atomic range claim can overlap owner pops
        // that advance `bottom` while leaving `top` unchanged, so batching the
        // reads must not bypass the last-item arbitration in `steal`.
        while count < MAX_BATCH_STEAL {
            match self.steal_within_access() {
                StealResult::Success(item) => {
                    items[count].write(item);
                    count += 1;
                }
                StealResult::Empty => break,
                StealResult::Retry => {
                    retry = true;
                    break;
                }
            }
        }

        if count == 0 {
            return if retry {
                StealResult::Retry
            } else {
                StealResult::Empty
            };
        }

        StealResult::Success(StolenBatch {
            items,
            next: 0,
            len: count,
        })
    }
}
