//! Capacity-bounded object pool parameterizing the crate's single
//! slot-freelist implementation, [`crate::pool::LockFreeStack`].

use crate::pool::LockFreeStack;

/// Memory pool for reducing allocation overhead.
///
/// A capacity-bounded, lock-free object pool: [`Self::deallocate`] stores up to
/// `max_size` objects for reuse and [`Self::allocate`] pops a stored object or
/// falls back to `T::default()`. The freelist mechanics live in
/// [`LockFreeStack`]; this type only adds the pool vocabulary
/// (allocate/deallocate) over the stack's fixed slot capacity.
pub struct MemoryPool<T> {
    stack: LockFreeStack<T>,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool retaining at most `max_size` objects.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            stack: LockFreeStack::with_capacity(max_size),
        }
    }

    /// Allocate an object from the pool or create new if pool is empty
    pub fn allocate(&self) -> T
    where
        T: Default,
    {
        self.stack.pop().unwrap_or_default()
    }

    /// Return an object to the pool for reuse.
    ///
    /// When the pool already holds `max_size` objects the item is dropped;
    /// this is the pool's documented retention cap, not an error condition.
    pub fn deallocate(&self, item: T) {
        if let Err(item) = self.stack.push(item) {
            drop(item); // pool at capacity: intentional retention-cap drop
        }
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.stack.len()
    }
}
