//! Public owner and stealer endpoints over the shared deque state.

use super::super::reclaim::{DeferredReclaim, DequeReclaimPolicy, SharedEpochReclaim};
use super::capacity::DequeCapacity;
use super::inner::ChaseLevInner;
use super::steal_outcome::{StealResult, StolenBatch};
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::{cell::Cell, marker::PhantomData, sync::Arc};

/// The unique bottom-side endpoint of a Chase-Lev work-stealing deque.
///
/// This endpoint is `Send`, but neither `Sync` nor `Clone`; therefore safe code
/// cannot create two concurrent push/pop owners. Use [`Self::stealer`] to
/// create cloneable top-side endpoints.
///
/// ```compile_fail
/// use moirai_scheduler::{ChaseLevDeque, DequeCapacity};
/// let capacity = DequeCapacity::<usize>::try_from(16).expect("16 is representable");
/// let owner = ChaseLevDeque::<usize>::new(capacity);
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
/// use moirai_scheduler::{ChaseLevDeque, DequeCapacity};
/// let capacity = DequeCapacity::<usize>::try_from(16).expect("16 is representable");
/// let owner = ChaseLevDeque::<usize>::new(capacity);
/// let mut stealer = owner.stealer();
/// stealer.push(1);
/// ```
pub struct ChaseLevStealer<T, P = DeferredReclaim>
where
    P: DequeReclaimPolicy,
{
    pub(crate) inner: Arc<ChaseLevInner<T, P>>,
}

impl<T, P> ChaseLevDeque<T, P>
where
    T: Send,
    P: DequeReclaimPolicy,
{
    /// Creates an empty deque with the validated allocation capacity.
    pub fn new(capacity: DequeCapacity<T>) -> Self {
        Self {
            inner: Arc::new(ChaseLevInner::new(capacity)),
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

    /// Returns the current allocated slot count.
    ///
    /// This is the initial capacity until the owner's push grows the deque,
    /// and the grown capacity afterwards. It reports the storage actually
    /// held, so a caller sizing retained memory reads the array rather than a
    /// recorded intent.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Replaces a grown buffer with one of `capacity` slots and frees every
    /// displaced buffer, returning whether storage was replaced.
    ///
    /// A deque that grew for a one-off burst otherwise keeps the grown buffer
    /// and, under [`DeferredReclaim`], every earlier one until the last
    /// endpoint drops. The call is a no-op when the current buffer is already
    /// no larger than `capacity`, or when the live items do not fit with one
    /// slot of headroom. It waits for in-flight steals, so the owner calls it
    /// where it has no work, such as before parking.
    pub fn shrink_to(&mut self, capacity: DequeCapacity<T>) -> bool {
        self.inner.shrink_to(capacity)
    }

    #[cfg(test)]
    pub(crate) fn retired_array_count(&self) -> usize {
        self.inner
            .retired_arrays
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    #[cfg(test)]
    pub(crate) fn poison_retired_array_lock_for_test(&self) {
        let _guard = self
            .inner
            .retired_arrays
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        panic!("poison retired-array mutex for recovery regression");
    }

    #[cfg(test)]
    pub(crate) fn set_indices_for_test(&self, index: isize) {
        self.inner.top.store(index, Ordering::Relaxed);
        self.inner.bottom.store(index, Ordering::Relaxed);
        let array_ptr = self.inner.array.load(Ordering::Relaxed);
        // SAFETY: tests call this only while the deque is empty and uniquely
        // owned, so resetting the generation markers is owner-exclusive.
        unsafe { &*array_ptr }.reset_states(index);
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

    /// Returns the current allocated slot count of the deque being stolen from.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}
