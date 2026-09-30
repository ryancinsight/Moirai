//! Several waiters on one task, held in the single waker slot of its state.
//!
//! A [`TaskState`](super::state::TaskState) holds one `Option<Waker>`, and its
//! size is pinned because every task retains one (1,024 per block). A second
//! waiter therefore cannot get a slot of its own. Instead, when a waker that
//! does not [`Waker::will_wake`] the held one arrives, the slot's waker is
//! replaced by a *fan-out* waker owning both; its wake wakes every member. The
//! slot keeps its type and its size, and completion keeps taking the slot's
//! waker and waking it exactly as it does for one waiter, so the completion
//! path carries no fan-out code and a task with one waiter never allocates.
//!
//! Alternatives rejected: a registry- or block-level table of spilled wakers
//! (completion reaches the state through a lifecycle token that holds no
//! registry or block, so the state would need a back-pointer, 8 bytes per task
//! against a pinned budget), and an `Empty | One | Many` slot enum (24 bytes
//! for the enum plus the mutex header, 8 more per task).
//!
//! A fan-out is recognised by its vtable, which is a `static` and so has one
//! address; the vtable of an `Arc<impl Wake>` waker is a promoted constant that
//! codegen units may duplicate, so it could not be compared.
//!
//! Every member is owned by the state's slot, so a fan-out is released with the
//! slot: by completion (woken), by [`register`] finding the task already
//! complete (woken), or by the state's drop. Block retirement requires every
//! task in the block to be complete, and completion drains the slot, so
//! retirement never finds a member registered. Members are bounded by the
//! distinct wakers that registered before completion; a waiter that is dropped
//! or that re-polls under a different waker leaves its old member behind until
//! completion, exactly as the single slot kept a dropped waiter's waker.

use std::task::{RawWaker, RawWakerVTable, Waker};

/// Identity and behavior of a fan-out waker whose data is a `Box<Vec<Waker>>`.
static FAN_OUT_VTABLE: RawWakerVTable = RawWakerVTable::new(
    clone_members,
    wake_members,
    wake_members_by_ref,
    drop_members,
);

/// Add `waker` to the wakers held in `slot`.
///
/// An empty slot stores a clone. A held waker that would wake the same target
/// makes the call a no-op, which also skips the clone. Otherwise the slot
/// becomes, or already is, a fan-out that gains `waker` unless one of its
/// members would wake the same target; the scan is linear in the members.
pub(super) fn register(slot: &mut Option<Waker>, waker: &Waker) {
    let Some(held) = slot.as_mut() else {
        *slot = Some(waker.clone());
        return;
    };
    if held.will_wake(waker) {
        return;
    }
    if let Some(members) = members_mut(held) {
        if !members.iter().any(|member| member.will_wake(waker)) {
            members.push(waker.clone());
        }
        return;
    }
    let first = std::mem::replace(held, Waker::noop().clone());
    *held = fan_out(vec![first, waker.clone()]);
}

/// A waker whose wake wakes every waker in `members`.
fn fan_out(members: Vec<Waker>) -> Waker {
    let data = Box::into_raw(Box::new(members)).cast_const().cast::<()>();
    // SAFETY: `data` is a live `Box<Vec<Waker>>` allocation, which is what
    // every function of `FAN_OUT_VTABLE` takes back, and the four functions
    // uphold the `RawWakerVTable` contract: `clone_members` allocates an
    // independent copy, `wake_members` and `drop_members` consume the
    // allocation, and `wake_members_by_ref` leaves it intact.
    unsafe { Waker::from_raw(RawWaker::new(data, &FAN_OUT_VTABLE)) }
}

/// The members of `waker` if it is a fan-out.
fn members_mut(waker: &mut Waker) -> Option<&mut Vec<Waker>> {
    if !std::ptr::eq(waker.vtable(), &FAN_OUT_VTABLE) {
        return None;
    }
    // SAFETY: the vtable is `FAN_OUT_VTABLE`'s, so the data pointer is the
    // `Box<Vec<Waker>>` allocation `fan_out` made, valid for as long as
    // `waker` is. The allocation is owned by this waker alone, because
    // `clone_members` copies it, so `&mut Waker` is exclusive access to it.
    Some(unsafe { &mut *waker.data().cast::<Vec<Waker>>().cast_mut() })
}

/// `RawWakerVTable::clone`: an independent fan-out over clones of the members.
unsafe fn clone_members(data: *const ()) -> RawWaker {
    // SAFETY: `data` is a live `Box<Vec<Waker>>` allocation (`fan_out`), and
    // this call only reads it.
    let members = unsafe { &*data.cast::<Vec<Waker>>() };
    let data = Box::into_raw(Box::new(members.clone()))
        .cast_const()
        .cast::<()>();
    RawWaker::new(data, &FAN_OUT_VTABLE)
}

/// `RawWakerVTable::wake`: wake every member, consuming the allocation.
unsafe fn wake_members(data: *const ()) {
    // SAFETY: `data` is a live `Box<Vec<Waker>>` allocation and `wake` consumes
    // the waker that owned it, so this is the last use.
    let members = unsafe { Box::from_raw(data.cast::<Vec<Waker>>().cast_mut()) };
    for member in *members {
        member.wake();
    }
}

/// `RawWakerVTable::wake_by_ref`: wake every member, leaving the allocation.
unsafe fn wake_members_by_ref(data: *const ()) {
    // SAFETY: `data` is a live `Box<Vec<Waker>>` allocation, and this call only
    // reads it.
    let members = unsafe { &*data.cast::<Vec<Waker>>() };
    for member in members {
        member.wake_by_ref();
    }
}

/// `RawWakerVTable::drop`: release the members and the allocation.
unsafe fn drop_members(data: *const ()) {
    // SAFETY: `data` is a live `Box<Vec<Waker>>` allocation and `drop` is the
    // last use of the waker that owned it.
    drop(unsafe { Box::from_raw(data.cast::<Vec<Waker>>().cast_mut()) });
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test scope")]

    use std::sync::Arc;

    use crate::counting_wake::counting_waker;

    use super::register;

    #[test]
    fn a_single_waiter_stays_unwrapped() {
        let (target, waker) = counting_waker();
        let mut slot = None;

        register(&mut slot, &waker);
        register(&mut slot, &waker);

        let held = slot.take().unwrap();
        assert!(held.will_wake(&waker), "one waiter must not allocate");
        held.wake();
        assert_eq!(target.wakes(), 1);
    }

    #[test]
    fn a_fan_out_wakes_each_member_once_by_ref_then_by_value() {
        let members: Vec<_> = (0..3).map(|_| counting_waker()).collect();
        let mut slot = None;
        for _ in 0..2 {
            for (_, waker) in &members {
                register(&mut slot, waker);
            }
        }
        let held = slot.take().unwrap();

        held.wake_by_ref();
        for (index, (target, _)) in members.iter().enumerate() {
            assert_eq!(target.wakes(), 1, "member {index} by reference");
        }
        held.wake();
        for (index, (target, _)) in members.iter().enumerate() {
            assert_eq!(target.wakes(), 2, "member {index} by value");
        }
    }

    #[test]
    fn a_cloned_fan_out_is_independent_of_its_source() {
        let members: Vec<_> = (0..2).map(|_| counting_waker()).collect();
        let mut slot = None;
        for (_, waker) in &members {
            register(&mut slot, waker);
        }
        let held = slot.take().unwrap();
        let copy = held.clone();

        drop(held);
        copy.wake();

        for (index, (target, _)) in members.iter().enumerate() {
            assert_eq!(target.wakes(), 1, "member {index}");
        }
    }

    #[test]
    fn dropping_a_fan_out_releases_every_member_without_waking() {
        let members: Vec<_> = (0..3).map(|_| counting_waker()).collect();
        let mut slot = None;
        for (_, waker) in &members {
            register(&mut slot, waker);
        }
        // Each target is held by its waker here and by one slot member.
        for (target, _) in &members {
            assert_eq!(Arc::strong_count(target), 3);
        }

        drop(slot);

        for (target, _) in &members {
            assert_eq!(target.wakes(), 0);
            assert_eq!(Arc::strong_count(target), 2);
        }
    }
}
