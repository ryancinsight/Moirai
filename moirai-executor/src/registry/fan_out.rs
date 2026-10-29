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
//! retirement never finds a member registered.
//!
//! Dedup rides [`Waker::will_wake`], which std documents as best-effort: it may
//! report two wakers for one target as different. A missed match costs a member
//! and a spurious wake, never a lost wake. Members are therefore bounded by the
//! distinct waker identities presented before completion, not by live futures:
//! the single slot retained exactly one waker, whereas a waiter that presents a
//! new waker on every poll adds a member per poll (each holding whatever its
//! waker owns) until completion, and each registration scans the members. A
//! dropped waiter leaves its member behind the same way.
//!
//! A panic in one member's wake is retained while every later member is still
//! attempted. The first panic resumes after the fan-out has consumed each
//! member exactly once.

use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    task::{RawWaker, RawWakerVTable, Waker},
};

/// Identity and behavior of a fan-out waker whose data is a `Box<Vec<Waker>>`.
static FAN_OUT_VTABLE: RawWakerVTable = RawWakerVTable::new(
    clone_members,
    wake_members,
    wake_members_by_ref,
    drop_members,
);

/// Merge an owned `waker` with the owned waker already held by a slot.
///
/// A held waker that `will_wake` the new one returns the unused duplicate for
/// its caller to drop. Otherwise the result becomes, or already is, a fan-out
/// that gains `waker` unless one of its members `will_wake` it; the scan is
/// linear in the members. `will_wake` is best-effort, so a repeat can be added
/// again.
pub(super) fn register(mut held: Waker, waker: Waker) -> (Waker, Option<Waker>) {
    if held.will_wake(&waker) {
        return (held, Some(waker));
    }
    if let Some(members) = members_mut(&mut held) {
        if members.iter().any(|member| member.will_wake(&waker)) {
            return (held, Some(waker));
        }
        members.push(waker);
        return (held, None);
    }
    (fan_out(vec![held, waker]), None)
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
///
unsafe fn wake_members(data: *const ()) {
    // SAFETY: `data` is a live `Box<Vec<Waker>>` allocation and `wake` consumes
    // the waker that owned it, so this is the last use.
    let members = unsafe { Box::from_raw(data.cast::<Vec<Waker>>().cast_mut()) };
    let mut panics = Vec::with_capacity(members.len());
    for member in *members {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| member.wake())) {
            panics.push(payload);
        }
    }
    resume_callback_panics(panics.into_iter());
}

/// `RawWakerVTable::wake_by_ref`: wake every member, leaving the allocation.
unsafe fn wake_members_by_ref(data: *const ()) {
    // SAFETY: `data` is a live `Box<Vec<Waker>>` allocation, and this call only
    // reads it.
    let members = unsafe { &*data.cast::<Vec<Waker>>() };
    let mut panics = Vec::with_capacity(members.len());
    for member in members {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| member.wake_by_ref())) {
            panics.push(payload);
        }
    }
    resume_callback_panics(panics.into_iter());
}

/// Resume the first callback panic after safely disposing later payloads.
pub(super) fn resume_callback_panics(mut panics: impl Iterator<Item = Box<dyn Any + Send>>) {
    let Some(first) = panics.next() else {
        return;
    };
    for secondary in panics {
        crate::schedule::job::run_caught(|| drop(secondary));
    }
    resume_unwind(first);
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

    fn register_slot(slot: &mut Option<std::task::Waker>, waker: &std::task::Waker) {
        let incoming = waker.clone();
        *slot = Some(match slot.take() {
            Some(held) => register(held, incoming).0,
            None => incoming,
        });
    }

    #[test]
    fn a_single_waiter_stays_unwrapped() {
        let (target, waker) = counting_waker();
        let mut slot = None;

        register_slot(&mut slot, &waker);
        register_slot(&mut slot, &waker);

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
                register_slot(&mut slot, waker);
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
            register_slot(&mut slot, waker);
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
            register_slot(&mut slot, waker);
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
