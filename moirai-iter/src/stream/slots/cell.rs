//! Stable pinned cells for retained futures and ordered outputs.

use core::future::Future;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::ptr;
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
    pub(super) const fn empty(vacant_next: usize) -> Self {
        Self {
            storage: MaybeUninit::uninit(),
            state: SlotState::Vacant,
            metadata: vacant_next,
        }
    }

    pub(super) const fn is_pollable(&self) -> bool {
        matches!(self.state, SlotState::Pending)
    }

    pub(super) fn insert(self: Pin<&mut Self>, future: Fut) {
        // SAFETY: the slot slab is already pinned. This method writes only an
        // empty slot and never moves a pending future out of its address.
        let this = unsafe { self.get_unchecked_mut() };
        debug_assert_eq!(
            this.state,
            SlotState::Vacant,
            "retained future slot must be vacant"
        );
        debug_assert_eq!(
            this.metadata, VACANT_END,
            "retained future slot must be detached from the vacancy list"
        );
        this.storage.write(future);
        this.metadata = ORDER_END;
        this.state = SlotState::Pending;
    }

    pub(super) fn take_vacant_next(self: Pin<&mut Self>) -> usize {
        // SAFETY: changing the intrusive vacancy link cannot move the pinned
        // future storage, which is uninitialized while this slot is vacant.
        let this = unsafe { self.get_unchecked_mut() };
        debug_assert_eq!(this.state, SlotState::Vacant);
        core::mem::replace(&mut this.metadata, VACANT_END)
    }

    pub(super) fn return_to_vacant(self: Pin<&mut Self>, next: usize) {
        // SAFETY: changing the intrusive vacancy link cannot move the pinned
        // future storage, whose initialized value was dropped before this slot
        // entered `Detached`.
        let this = unsafe { self.get_unchecked_mut() };
        debug_assert_eq!(this.state, SlotState::Detached);
        debug_assert_eq!(
            this.metadata, VACANT_END,
            "returned future slot must not already be vacant"
        );
        this.metadata = next;
        this.state = SlotState::Vacant;
    }

    pub(super) fn set_order_next(self: Pin<&mut Self>, next: usize) {
        // SAFETY: updating scalar metadata does not move the pinned future.
        let this = unsafe { self.get_unchecked_mut() };
        debug_assert!(matches!(
            this.state,
            SlotState::Pending | SlotState::Completed
        ));
        debug_assert_eq!(this.metadata, ORDER_END);
        this.metadata = next;
    }

    pub(super) fn order_next(&self) -> usize {
        debug_assert!(matches!(
            self.state,
            SlotState::Pending | SlotState::Detached | SlotState::Completed
        ));
        self.metadata
    }

    pub(super) fn mark_completed(self: Pin<&mut Self>) {
        // SAFETY: the future was already dropped in place. Publishing the
        // completed state changes only scalar metadata beside pinned storage.
        let this = unsafe { self.get_unchecked_mut() };
        debug_assert_eq!(this.state, SlotState::Detached);
        this.state = SlotState::Completed;
    }

    pub(super) fn take_completed_next(self: Pin<&mut Self>) -> Option<usize> {
        // SAFETY: the future is absent in `Completed`; this transition changes
        // scalar state only and prepares the cell for vacancy reinsertion.
        let this = unsafe { self.get_unchecked_mut() };
        if this.state != SlotState::Completed {
            return None;
        }
        this.state = SlotState::Detached;
        Some(core::mem::replace(&mut this.metadata, ORDER_END))
    }
}

impl<Fut> FutureSlot<Fut>
where
    Fut: Future,
{
    pub(super) fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Fut::Output> {
        // SAFETY: the slot remains pinned for this call. `storage` is
        // initialized exactly in `Pending`, and this method never moves the
        // future before dropping it in place after `Ready`.
        let this = unsafe { self.get_unchecked_mut() };
        debug_assert_eq!(this.state, SlotState::Pending);
        let future = this.storage.as_mut_ptr();
        // SAFETY: `future` points to an initialized value in a stable pinned
        // slab and remains at that address until it is dropped below.
        let poll = unsafe { Pin::new_unchecked(&mut *future) }.poll(context);
        match poll {
            Poll::Ready(output) => {
                // Detach first so an unwinding destructor cannot be run twice
                // by `FutureSlot::drop`. The ordered link remains live until
                // the caller publishes or consumes the output.
                this.state = SlotState::Detached;
                // SAFETY: `future` is initialized and has not been moved. The
                // state transition makes this its unique drop.
                unsafe { ptr::drop_in_place(future) };
                Poll::Ready(output)
            }
            Poll::Pending => Poll::Pending,
        }
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
