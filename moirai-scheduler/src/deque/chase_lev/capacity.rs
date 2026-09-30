//! Validated allocation capacity for the Chase-Lev deque.

use std::{fmt, marker::PhantomData, mem::MaybeUninit, sync::atomic::AtomicIsize};

pub(crate) const MIN_DEQUE_CAPACITY: usize = 16;

/// A validated Chase-Lev allocation capacity for element type `T`.
///
/// Values below the implementation minimum normalize to 16 slots. Other
/// values round upward to the next power of two. Use [`TryFrom<usize>`] so an
/// unrepresentable next power or concrete slot/state layout is reported before
/// allocation.
#[repr(transparent)]
pub struct DequeCapacity<T> {
    slots: usize,
    element: PhantomData<fn() -> T>,
}

impl<T> Copy for DequeCapacity<T> {}

impl<T> Clone for DequeCapacity<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for DequeCapacity<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("DequeCapacity")
            .field(&self.slots)
            .finish()
    }
}

impl<T> DequeCapacity<T> {
    /// Return the normalized slot count.
    #[must_use]
    pub const fn get(self) -> usize {
        self.slots
    }

    /// The smallest capacity this implementation allocates.
    ///
    /// A deque grows on the owner's push, so this is the right initial
    /// capacity for a queue whose use is possible but not expected: it pays
    /// the minimum retained storage and takes owner-only resizes if work
    /// arrives. Callers reach it without naming the slot count or handling a
    /// [`TryFrom`] failure that cannot occur, since the minimum is always
    /// representable.
    #[must_use]
    pub const fn minimum() -> Self {
        Self {
            slots: MIN_DEQUE_CAPACITY,
            element: PhantomData,
        }
    }
}

impl<T> TryFrom<usize> for DequeCapacity<T> {
    type Error = DequeCapacityError;

    fn try_from(requested: usize) -> Result<Self, Self::Error> {
        let slots = requested
            .checked_next_power_of_two()
            .map(|capacity| capacity.max(MIN_DEQUE_CAPACITY))
            .ok_or(DequeCapacityError { requested })?;
        std::alloc::Layout::array::<std::cell::UnsafeCell<MaybeUninit<T>>>(slots)
            .and_then(|_| std::alloc::Layout::array::<AtomicIsize>(slots))
            .map_err(|_| DequeCapacityError { requested })?;
        Ok(Self {
            slots,
            element: PhantomData,
        })
    }
}

/// Failure to represent a requested Chase-Lev allocation capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DequeCapacityError {
    requested: usize,
}

impl DequeCapacityError {
    /// Return the requested, unrepresentable capacity.
    #[must_use]
    pub const fn requested(self) -> usize {
        self.requested
    }
}

impl fmt::Display for DequeCapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "deque capacity {} cannot form a supported allocation",
            self.requested
        )
    }
}

impl std::error::Error for DequeCapacityError {}
