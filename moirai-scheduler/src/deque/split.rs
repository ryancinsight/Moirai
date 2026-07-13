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
