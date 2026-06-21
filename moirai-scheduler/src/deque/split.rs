use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use super::chase_lev::{ChaseLevDeque, StealResult};
use super::reclaim::{DequeReclaimPolicy, QuiescentReclaim};

/// A split work-stealing deque inspired by the Lace runtime.
pub struct SplitDeque<T, const N: usize = 32, P = QuiescentReclaim>
where
    P: DequeReclaimPolicy,
{
    private: Mutex<PrivateStack<T, N>>,
    shared: ChaseLevDeque<T, P>,
    policy: PhantomData<P>,
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

    fn offload_oldest_half<P>(&mut self, shared: &ChaseLevDeque<T, P>)
    where
        P: DequeReclaimPolicy,
    {
        let count = self.len / 2;

        for item in self.items.iter_mut().take(count) {
            shared.push(unsafe { item.assume_init_read() });
        }

        let retained = self.len - count;
        if retained > 0 {
            unsafe {
                std::ptr::copy(
                    self.items.as_ptr().add(count),
                    self.items.as_mut_ptr(),
                    retained,
                );
            }
        }
        self.len = retained;
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
    P: DequeReclaimPolicy,
{
    pub fn new() -> Self {
        assert!(N >= 2, "SplitDeque capacity N must be at least 2");
        Self {
            private: Mutex::new(PrivateStack::new()),
            shared: ChaseLevDeque::new(N),
            policy: PhantomData,
        }
    }

    pub fn push(&self, item: T) {
        let mut private = self.private.lock().unwrap_or_else(|e| e.into_inner());
        if private.len >= N {
            private.offload_oldest_half(&self.shared);
        }
        private.push(item);
    }

    pub fn pop(&self) -> Option<T> {
        let mut private = self.private.lock().unwrap_or_else(|e| e.into_inner());
        private.pop().or_else(|| self.shared.pop())
    }

    pub fn steal(&self) -> StealResult<T> {
        self.shared.steal()
    }

    pub fn steal_batch_with<F>(&self, f: F) -> StealResult<T>
    where
        F: FnMut(T),
    {
        self.shared.steal_batch_with(f)
    }

    pub fn len(&self) -> usize {
        self.private.lock().unwrap_or_else(|e| e.into_inner()).len + self.shared.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const N: usize, P> Default for SplitDeque<T, N, P>
where
    P: DequeReclaimPolicy,
{
    fn default() -> Self {
        Self::new()
    }
}
