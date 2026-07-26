//! Split work-stealing deque: a private owner stack over a shared deque.
//!
//! The owner pushes and pops against an in-line `PrivateStack` that thieves
//! cannot see, and only spills to the shared [`ChaseLevDeque`] when that stack
//! fills. Most owner traffic therefore costs no atomics at all, and the shared
//! deque carries only the work old enough to be worth stealing.
//!
//! # Which end each operation takes
//!
//! `push` and `pop` work the newest end of the private stack, while the spill
//! hands over the *oldest* half — so an owner keeps the freshest work, where its
//! cache is warm, and thieves get the coldest, where contention is least. `pop`
//! falls through to the shared deque only once the private stack is empty.
//!
//! # Initialization invariant
//!
//! `PrivateStack` stores `[MaybeUninit<T>; N]` with a `len`, and the invariant
//! is that exactly `items[0..len]` are initialized. Every unsafe access here is
//! justified by that one statement: `pop` reads at `len - 1` after decrementing,
//! `Drop` drops precisely the live prefix, and the spill reads `items[0..count]`
//! having checked `count <= len`.
//!
//! Keeping the invariant true across a panic is what [`OffloadGuard`] is for.
//! The spill moves items out of the front one at a time and pushes each onto the
//! shared deque, so between iterations the prefix `items[0..moved]` is already
//! moved out while `items[moved..len]` is still live — a state no plain `len`
//! can describe. If a push unwinds there, the guard's `Drop` shifts the live
//! remainder down over the vacated slots and shortens `len` to match, restoring
//! the invariant before any other code observes the stack. That also matters for
//! the poison-tolerant `Mutex` below: `unwrap_or_else(into_inner)` continues
//! after a panic, and it can only do so safely because the guard has already
//! repaired the state by the time the lock is released.
//!
//! The guard's normal path and its panic path converge on the same shift, which
//! is why a panic in the *last* push — where `moved` has reached `count` — is
//! handled correctly by the branch nominally labelled "no panic".

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use super::chase_lev::{ChaseLevDeque, ChaseLevStealer, StealResult, StolenBatch};
use super::reclaim::{DeferredReclaim, DequeReclaimPolicy};

/// A split work-stealing deque inspired by the Lace runtime.
pub struct SplitDeque<T, const N: usize = 32, P = DeferredReclaim>
where
    P: DequeReclaimPolicy,
{
    owner: Mutex<SplitOwner<T, N, P>>,
    stealer: ChaseLevStealer<T, P>,
    policy: PhantomData<P>,
}

struct SplitOwner<T, const N: usize, P>
where
    P: DequeReclaimPolicy,
{
    private: PrivateStack<T, N>,
    shared: ChaseLevDeque<T, P>,
}

struct PrivateStack<T, const N: usize> {
    items: [MaybeUninit<T>; N],
    len: usize,
}

impl<T, const N: usize> PrivateStack<T, N> {
    fn new() -> Self {
        Self {
            items: std::array::from_fn(|_| MaybeUninit::uninit()),
            len: 0,
        }
    }

    fn push(&mut self, item: T) {
        debug_assert!(self.len < N, "private stack capacity is exhausted");
        self.items[self.len].write(item);
        self.len += 1;
    }

    fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        // SAFETY: `len` was non-zero and has just been decremented, so it now
        // indexes the topmost live element. Reading it out leaves `items[0..len]`
        // initialized, which is the invariant.
        Some(unsafe { self.items[self.len].assume_init_read() })
    }

    fn offload_oldest_half<P>(&mut self, shared: &mut ChaseLevDeque<T, P>)
    where
        T: Send,
        P: DequeReclaimPolicy,
    {
        let count = self.len / 2;
        let mut guard = OffloadGuard {
            stack: self,
            moved: 0,
            count,
        };

        for i in 0..count {
            // SAFETY: `i < count <= len / 2 <= len`, so this slot is live. The
            // read leaves it moved-out, which `guard.moved` records; the guard's
            // `Drop` reconciles that with `len` whether or not `push` unwinds.
            let item = unsafe { guard.stack.items[i].assume_init_read() };
            guard.moved += 1;
            shared.push(item);
        }
    }
}

struct OffloadGuard<'a, T, const N: usize> {
    stack: &'a mut PrivateStack<T, N>,
    moved: usize,
    count: usize,
}

impl<'a, T, const N: usize> Drop for OffloadGuard<'a, T, N> {
    fn drop(&mut self) {
        if self.moved < self.count {
            // A panic occurred during one of the pushes.
            // Shift the remaining initialized elements to the front of the stack.
            let remaining_to_move = self.count - self.moved;
            let retained = self.stack.len - self.count;
            // SAFETY: `items[moved..len]` are still initialized — the loop only
            // moved out the first `moved`. Shifting that whole live range down
            // to index 0 spans `(count - moved) + (len - count) == len - moved`
            // elements, all within `N`. `copy` is a memmove, so the overlap
            // between source and destination is fine.
            unsafe {
                std::ptr::copy(
                    self.stack.items.as_ptr().add(self.moved),
                    self.stack.items.as_mut_ptr(),
                    remaining_to_move + retained,
                );
            }
            self.stack.len -= self.moved;
        } else {
            // Normal execution: all `count` elements successfully offloaded.
            let retained = self.stack.len - self.count;
            if retained > 0 {
                // SAFETY: every one of the first `count` slots was moved out, so
                // the live elements are `items[count..len]`; shifting those
                // `retained` down to index 0 stays in bounds, and `copy`
                // tolerates the overlap.
                unsafe {
                    std::ptr::copy(
                        self.stack.items.as_ptr().add(self.count),
                        self.stack.items.as_mut_ptr(),
                        retained,
                    );
                }
            }
            self.stack.len = retained;
        }
    }
}

impl<T, const N: usize> Drop for PrivateStack<T, N> {
    fn drop(&mut self) {
        for item in self.items.iter_mut().take(self.len) {
            // SAFETY: `items[0..len]` are exactly the initialized elements, and
            // each is dropped once here. Slots past `len` were either never
            // written or already moved out by `pop`/the spill.
            unsafe {
                item.assume_init_drop();
            }
        }
    }
}

impl<T, const N: usize, P> SplitDeque<T, N, P>
where
    T: Send,
    P: DequeReclaimPolicy,
{
    pub fn new() -> Self {
        assert!(N >= 2, "SplitDeque capacity N must be at least 2");
        let shared = ChaseLevDeque::new(N);
        let stealer = shared.stealer();
        Self {
            owner: Mutex::new(SplitOwner {
                private: PrivateStack::new(),
                shared,
            }),
            stealer,
            policy: PhantomData,
        }
    }

    pub fn push(&self, item: T) {
        let mut owner = self.owner.lock().unwrap_or_else(|e| e.into_inner());
        if owner.private.len >= N {
            let SplitOwner { private, shared } = &mut *owner;
            private.offload_oldest_half(shared);
        }
        owner.private.push(item);
    }

    pub fn pop(&self) -> Option<T> {
        let mut owner = self.owner.lock().unwrap_or_else(|e| e.into_inner());
        owner.private.pop().or_else(|| owner.shared.pop())
    }

    pub fn steal(&self) -> StealResult<T> {
        self.stealer.steal()
    }

    pub fn steal_batch(&self) -> StealResult<StolenBatch<T>> {
        self.stealer.steal_batch()
    }

    pub fn len(&self) -> usize {
        let owner = self.owner.lock().unwrap_or_else(|e| e.into_inner());
        owner.private.len + owner.shared.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const N: usize, P> Default for SplitDeque<T, N, P>
where
    T: Send,
    P: DequeReclaimPolicy,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn pop_drains_the_shared_deque_once_the_private_stack_empties() {
        // `pop` falls through to the shared side only when the private stack is
        // exhausted. The existing offload test always drains the shared deque by
        // stealing first, so that fall-through never returns a value there.
        let deque: SplitDeque<i32, 4> = SplitDeque::new();

        for value in 1..=5 {
            deque.push(value);
        }
        // The fifth push spilled the oldest half — 1 and 2 — leaving 3, 4, 5.

        assert_eq!(deque.pop(), Some(5));
        assert_eq!(deque.pop(), Some(4));
        assert_eq!(deque.pop(), Some(3));

        // Private side is empty now, so the rest must come from the shared deque
        // rather than reporting the queue as drained.
        let spilled = [deque.pop(), deque.pop()];
        assert!(
            spilled.iter().all(Option::is_some),
            "both spilled items must still be reachable through pop, got {spilled:?}"
        );
        let mut spilled: Vec<i32> = spilled.into_iter().flatten().collect();
        spilled.sort_unstable();
        assert_eq!(spilled, vec![1, 2]);

        assert_eq!(deque.pop(), None);
        assert!(deque.is_empty());
    }

    #[test]
    fn spilled_items_are_dropped_exactly_once() {
        // The spill moves items between two owners of very different kinds; a
        // slot counted by both — or by neither — shows up here and nowhere else.
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let deque: SplitDeque<DropProbe, 4> = SplitDeque::new();
            for _ in 0..5 {
                deque.push(DropProbe(drops.clone()));
            }
            assert_eq!(drops.load(Ordering::Relaxed), 0, "nothing dropped yet");
        }

        assert_eq!(
            drops.load(Ordering::Relaxed),
            5,
            "every item must be dropped once, whether it ended up private or spilled"
        );
    }

    #[test]
    fn test_private_stack_offload_panic_safety() {
        let drop_count = Arc::new(AtomicUsize::new(0));
        let mut stack: PrivateStack<DropProbe, 8> = PrivateStack::new();

        // Push 6 items
        for _ in 0..6 {
            stack.push(DropProbe(drop_count.clone()));
        }

        // We want to offload 3 items (len/2)
        // Simulate a panic after 1 item is moved
        {
            let mut guard = OffloadGuard {
                stack: &mut stack,
                moved: 0,
                count: 3,
            };

            // Simulating reading/moving the first item
            let item = unsafe { guard.stack.items[0].assume_init_read() };
            guard.moved += 1;
            // The item is successfully pushed/moved. Let's drop it here representing successful move
            drop(item);
            assert_eq!(drop_count.load(Ordering::Relaxed), 1);

            // Now the guard drops, simulating a panic during the second push
        }

        assert_eq!(stack.len, 5);
        assert_eq!(drop_count.load(Ordering::Relaxed), 1);

        // When stack drops, the remaining 5 items should be dropped exactly once
        drop(stack);
        assert_eq!(drop_count.load(Ordering::Relaxed), 6);
    }
}
