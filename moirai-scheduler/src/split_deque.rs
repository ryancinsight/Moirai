//! Split work-stealing deque with a const-generic private stack.
//!
//! [`SplitDeque`] separates owner-biased local work from thief-visible work:
//! pushes and owner pops use a bounded inline LIFO stack, while overflow is
//! offloaded to a shared Chase-Lev deque for stealing.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use crate::deque::{ChaseLevDeque, StealResult};
use crate::reclaim::{DequeReclaimPolicy, QuiescentReclaim};

/// A split work-stealing deque inspired by the Lace runtime.
///
/// The private stack has a fixed const-generic capacity `N`. When it fills,
/// the oldest half of the private entries is moved to the shared Chase-Lev
/// deque, making those tasks visible to thieves without allocating in the
/// private storage.
pub struct SplitDeque<T, const N: usize = 32, P = QuiescentReclaim>
where
    P: DequeReclaimPolicy,
{
    /// Owner-biased inline stack for private pushes and pops.
    private: Mutex<PrivateStack<T, N>>,
    /// Shared Chase-Lev deque for thief steals and offloaded owner tasks.
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
        // SAFETY: indices below `self.len` before decrement are initialized.
        // The decrement removes this slot from the initialized prefix before
        // the value is read, so `Drop` will not drop it a second time.
        Some(unsafe { self.items[self.len].assume_init_read() })
    }

    fn offload_oldest_half<P>(&mut self, shared: &ChaseLevDeque<T, P>)
    where
        P: DequeReclaimPolicy,
    {
        let count = self.len / 2;

        for item in self.items.iter_mut().take(count) {
            // SAFETY: `0..count` is inside the initialized prefix because
            // `count <= self.len`. Reading moves each value out exactly once.
            shared.push(unsafe { item.assume_init_read() });
        }

        let retained = self.len - count;
        if retained > 0 {
            // SAFETY: source range `count..self.len` and destination range
            // `0..retained` are both inside the backing array. `copy` permits
            // overlapping regions and performs a bitwise move of MaybeUninit
            // slots. The old tail is excluded by the new length below.
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
            // SAFETY: `0..self.len` is the initialized prefix maintained by
            // `push`, `pop`, and `offload_oldest_half`.
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
    /// Create a new split deque.
    pub fn new() -> Self {
        assert!(N >= 2, "SplitDeque capacity N must be at least 2");
        Self {
            private: Mutex::new(PrivateStack::new()),
            shared: ChaseLevDeque::new(N),
            policy: PhantomData,
        }
    }

    /// Push an item to the private stack.
    ///
    /// If the private stack is full, the oldest half of the entries is moved
    /// to the shared deque before the new item is inserted.
    pub fn push(&self, item: T) {
        let mut private = self.private.lock().unwrap_or_else(|e| e.into_inner());
        if private.len >= N {
            private.offload_oldest_half(&self.shared);
        }
        private.push(item);
    }

    /// Pop an item from the private stack or, if empty, the shared deque.
    pub fn pop(&self) -> Option<T> {
        let mut private = self.private.lock().unwrap_or_else(|e| e.into_inner());
        private.pop().or_else(|| self.shared.pop())
    }

    /// Steal an item from the top of the shared deque.
    pub fn steal(&self) -> StealResult<T> {
        self.shared.steal()
    }

    /// Steal a batch of items from the top of the shared deque.
    pub fn steal_batch_with<F>(&self, f: F) -> StealResult<T>
    where
        F: FnMut(T),
    {
        self.shared.steal_batch_with(f)
    }

    /// Get the total size of the private stack and shared deque.
    pub fn len(&self) -> usize {
        self.private.lock().unwrap_or_else(|e| e.into_inner()).len + self.shared.len()
    }

    /// Check if the deque is empty.
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
