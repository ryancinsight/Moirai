//! Stable pinned cells for retained futures and ordered outputs.
//!
//! # Provenance contract
//!
//! A pending future may hold a `&mut` into its own state across an await
//! point (for example a local flag borrowed by a leaf future). That borrow is
//! a child of the pointer the slab handed to `poll`, so any later shared
//! reference covering the future's bytes is a read that invalidates it under
//! Stacked Borrows. The slab is therefore addressed only through raw
//! pointers: every access projects a single scalar field with `&raw`, and no
//! reference to a whole [`FutureSlot`] or to the slot slice is ever formed
//! while a future may be pending. The only reference that covers a future's
//! bytes is the pinned `&mut Fut` handed to its own `poll`.

use core::future::Future;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::ptr::{self, NonNull};
use core::task::{Context, Poll};

use super::{ORDER_END, VACANT_END};

/// One stable-address future cell inside a pinned contiguous slab.
///
/// `state` distinguishes initialized futures, detached ready values, retained
/// completed outputs, and vacant cells. It transitions away from `Pending`
/// before a future is dropped, so cancellation and unwinding drop each inserted
/// future at most once without moving a pinned value.
///
/// `metadata` stores the next physical slot in input order while pending or
/// completed and the intrusive vacancy link while vacant. Slot state is the
/// discriminant, so both linked structures reuse one full-width word.
pub(super) struct FutureSlot<Fut> {
    storage: MaybeUninit<Fut>,
    state: SlotState,
    metadata: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum SlotState {
    Vacant,
    Pending,
    Detached,
    Completed,
}

impl<Fut> FutureSlot<Fut> {
    const fn empty(vacant_next: usize) -> Self {
        Self {
            storage: MaybeUninit::uninit(),
            state: SlotState::Vacant,
            metadata: vacant_next,
        }
    }

    /// Reads `state` without forming a reference to the slot.
    ///
    /// # Safety
    ///
    /// `slot` must point to a live `FutureSlot` and no write to its `state`
    /// may race this read.
    unsafe fn state(slot: *mut Self) -> SlotState {
        // SAFETY: the caller guarantees `slot` is live; `state` is always
        // initialized, and `&raw const` projects the field without a reference.
        unsafe { (&raw const (*slot).state).read() }
    }

    /// Writes `state` without forming a reference to the slot.
    ///
    /// # Safety
    ///
    /// `slot` must point to a live `FutureSlot` and the caller must have
    /// exclusive access to its scalar fields.
    unsafe fn set_state(slot: *mut Self, state: SlotState) {
        // SAFETY: the caller guarantees liveness and exclusive access.
        unsafe { (&raw mut (*slot).state).write(state) };
    }

    /// Reads `metadata` without forming a reference to the slot.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::state`].
    unsafe fn metadata(slot: *mut Self) -> usize {
        // SAFETY: the caller guarantees `slot` is live; `metadata` is always
        // initialized.
        unsafe { (&raw const (*slot).metadata).read() }
    }

    /// Replaces `metadata`, returning the previous word.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::set_state`].
    unsafe fn replace_metadata(slot: *mut Self, metadata: usize) -> usize {
        // SAFETY: the caller guarantees liveness and exclusive access.
        unsafe { ptr::replace(&raw mut (*slot).metadata, metadata) }
    }

    /// Address of the future storage without forming a reference.
    ///
    /// # Safety
    ///
    /// `slot` must point to a live `FutureSlot`.
    unsafe fn storage(slot: *mut Self) -> *mut Fut {
        // SAFETY: the caller guarantees `slot` is live. `MaybeUninit<Fut>` is
        // `repr(transparent)` over `Fut`, so the cast preserves layout.
        unsafe { (&raw mut (*slot).storage).cast::<Fut>() }
    }
}

impl<Fut> Drop for FutureSlot<Fut> {
    fn drop(&mut self) {
        if self.state == SlotState::Pending {
            self.state = SlotState::Detached;
            // SAFETY: a pending slot contains one initialized future. The
            // slab has not moved it, and clearing first prevents a second drop
            // if the future destructor unwinds.
            unsafe { ptr::drop_in_place(self.storage.as_mut_ptr()) };
        }
    }
}

/// Owned, never-moving array of [`FutureSlot`]s addressed through raw pointers.
///
/// The marker names the ownership (and keeps the slab `Unpin` and dropck-equal
/// to the `Box<[FutureSlot<Fut>]>` it replaces).
///
/// The allocation is created once by `Box::into_raw` and freed once in
/// [`Drop`], so the slab stays at one address for its whole life and moving
/// the owner (a `Vec` push, an `Option` assignment) retags nothing: a `Box`
/// field would be Unique-retagged over the whole slab on every such move,
/// invalidating every pointer already derived from it.
pub(super) struct SlotSlab<Fut> {
    slots: NonNull<[FutureSlot<Fut>]>,
    owns: PhantomData<Box<[FutureSlot<Fut>]>>,
}

// SAFETY: `SlotSlab` uniquely owns its slots exactly as a `Box<[FutureSlot<Fut>]>`
// would, so it may cross threads whenever the owned futures may.
unsafe impl<Fut: Send> Send for SlotSlab<Fut> {}

// SAFETY: every `&self` method only reads scalar slot fields; all writes take
// `&mut self`. Sharing the slab therefore shares nothing beyond `Fut` itself,
// matching `Box<[FutureSlot<Fut>]>`.
unsafe impl<Fut: Sync> Sync for SlotSlab<Fut> {}

impl<Fut> SlotSlab<Fut> {
    /// Allocates `len` vacant slots chained into one vacancy list.
    pub(super) fn new(len: usize) -> Self {
        let slots = (0..len)
            .map(|index| {
                let next = if index + 1 == len {
                    VACANT_END
                } else {
                    index + 1
                };
                FutureSlot::empty(next)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let slots = NonNull::new(Box::into_raw(slots))
            .expect("invariant: a boxed slice allocation is never null");
        Self {
            slots,
            owns: PhantomData,
        }
    }

    pub(super) const fn len(&self) -> usize {
        self.slots.len()
    }

    /// Address of slot `index`, bounds-checked in every build because the
    /// pointer offset below is undefined behavior out of range.
    fn slot(&self, index: usize) -> *mut FutureSlot<Fut> {
        assert!(
            index < self.len(),
            "invariant: retained slot index is in bounds"
        );
        // SAFETY: `index < len`, so the offset stays inside the one allocation
        // created in `new`. The base pointer keeps the provenance of the whole
        // slice and no reference to any element is created here.
        unsafe { self.slots.as_ptr().cast::<FutureSlot<Fut>>().add(index) }
    }

    pub(super) fn is_pollable(&self, index: usize) -> bool {
        let slot = self.slot(index);
        // SAFETY: `slot` is a live slot; `&self` excludes concurrent writes.
        unsafe { FutureSlot::state(slot) == SlotState::Pending }
    }

    pub(super) fn insert(&mut self, index: usize, future: Fut) {
        let slot = self.slot(index);
        // SAFETY: `&mut self` gives exclusive access to the slot scalars.
        unsafe {
            debug_assert_eq!(
                FutureSlot::state(slot),
                SlotState::Vacant,
                "retained future slot must be vacant"
            );
            debug_assert_eq!(
                FutureSlot::metadata(slot),
                VACANT_END,
                "retained future slot must be detached from the vacancy list"
            );
        }
        // SAFETY: the slot is vacant, so its storage holds no value to
        // overwrite, and writing through the raw storage pointer moves nothing
        // that is pinned.
        unsafe { FutureSlot::storage(slot).write(future) };
        // SAFETY: `&mut self` gives exclusive access to the slot scalars. The
        // future is written first so `Pending` never names uninitialized data.
        unsafe {
            FutureSlot::replace_metadata(slot, ORDER_END);
            FutureSlot::set_state(slot, SlotState::Pending);
        }
    }

    pub(super) fn take_vacant_next(&mut self, index: usize) -> usize {
        let slot = self.slot(index);
        // SAFETY: `&mut self` gives exclusive access to the slot scalars, and
        // changing the vacancy link cannot touch the uninitialized storage of
        // a vacant slot.
        unsafe {
            debug_assert_eq!(FutureSlot::state(slot), SlotState::Vacant);
            FutureSlot::replace_metadata(slot, VACANT_END)
        }
    }

    pub(super) fn return_to_vacant(&mut self, index: usize, next: usize) {
        let slot = self.slot(index);
        // SAFETY: `&mut self` gives exclusive access to the slot scalars. The
        // initialized value was dropped before the slot entered `Detached`,
        // so the storage is dead when the slot becomes vacant.
        unsafe {
            debug_assert_eq!(FutureSlot::state(slot), SlotState::Detached);
            debug_assert_eq!(
                FutureSlot::metadata(slot),
                VACANT_END,
                "returned future slot must not already be vacant"
            );
            FutureSlot::replace_metadata(slot, next);
            FutureSlot::set_state(slot, SlotState::Vacant);
        }
    }

    pub(super) fn set_order_next(&mut self, index: usize, next: usize) {
        let slot = self.slot(index);
        // SAFETY: `&mut self` gives exclusive access to the slot scalars;
        // updating them cannot move the pinned future.
        unsafe {
            debug_assert!(matches!(
                FutureSlot::state(slot),
                SlotState::Pending | SlotState::Completed
            ));
            debug_assert_eq!(FutureSlot::metadata(slot), ORDER_END);
            FutureSlot::replace_metadata(slot, next);
        }
    }

    pub(super) fn order_next(&self, index: usize) -> usize {
        let slot = self.slot(index);
        // SAFETY: `slot` is a live slot; `&self` excludes concurrent writes.
        unsafe {
            debug_assert!(matches!(
                FutureSlot::state(slot),
                SlotState::Pending | SlotState::Detached | SlotState::Completed
            ));
            FutureSlot::metadata(slot)
        }
    }

    pub(super) fn mark_completed(&mut self, index: usize) {
        let slot = self.slot(index);
        // SAFETY: `&mut self` gives exclusive access to the slot scalars. The
        // future was already dropped in place, so publishing `Completed`
        // changes only scalar state beside dead storage.
        unsafe {
            debug_assert_eq!(FutureSlot::state(slot), SlotState::Detached);
            FutureSlot::set_state(slot, SlotState::Completed);
        }
    }

    pub(super) fn take_completed_next(&mut self, index: usize) -> Option<usize> {
        let slot = self.slot(index);
        // SAFETY: `&mut self` gives exclusive access to the slot scalars. The
        // future is absent in `Completed`; the transition changes scalar state
        // only and prepares the cell for vacancy reinsertion.
        unsafe {
            if FutureSlot::state(slot) != SlotState::Completed {
                return None;
            }
            FutureSlot::set_state(slot, SlotState::Detached);
            Some(FutureSlot::replace_metadata(slot, ORDER_END))
        }
    }
}

impl<Fut> SlotSlab<Fut>
where
    Fut: Future,
{
    pub(super) fn poll(&mut self, index: usize, context: &mut Context<'_>) -> Poll<Fut::Output> {
        let slot = self.slot(index);
        // SAFETY: `&mut self` gives exclusive access to the slot.
        let future = unsafe {
            debug_assert_eq!(FutureSlot::state(slot), SlotState::Pending);
            FutureSlot::storage(slot)
        };
        // SAFETY: `future` points to an initialized value (`Pending`) in a slab
        // whose allocation never moves, and it stays at that address until it
        // is dropped below.
        let poll = unsafe { Pin::new_unchecked(&mut *future) }.poll(context);
        match poll {
            Poll::Ready(output) => {
                // Detach first so an unwinding destructor cannot be run twice
                // by `FutureSlot::drop`. The ordered link remains live until
                // the caller publishes or consumes the output.
                // SAFETY: `&mut self` gives exclusive access to the slot scalars.
                unsafe { FutureSlot::set_state(slot, SlotState::Detached) };
                // SAFETY: `future` is initialized and has not been moved. The
                // state transition makes this its unique drop.
                unsafe { ptr::drop_in_place(future) };
                Poll::Ready(output)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<Fut> Drop for SlotSlab<Fut> {
    fn drop(&mut self) {
        // SAFETY: `slots` came from `Box::into_raw` in `new`, is uniquely owned
        // by this slab, and is freed only here. `FutureSlot::drop` then drops
        // each pending future in place.
        drop(unsafe { Box::from_raw(self.slots.as_ptr()) });
    }
}
