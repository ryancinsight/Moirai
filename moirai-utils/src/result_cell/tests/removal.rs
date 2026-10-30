use super::{
    CountingWake, GatedStateWord, PanicDropWaiter, ReentrantCompletion, ReentrantDropWaiter,
};
use crate::result_cell::{RESULT_PENDING, ResultCell, WAITER_IDLE};
use std::sync::{Arc, atomic::Ordering};
use std::task::Waker;
use std::thread;

#[test]
fn removal_returns_the_registered_owner_and_allows_reregistration() {
    let first = Arc::new(CountingWake(std::sync::atomic::AtomicUsize::new(0)));
    let second = Arc::new(CountingWake(std::sync::atomic::AtomicUsize::new(0)));
    let cell = ResultCell::<usize, Waker>::new();

    // SAFETY: this thread is the cell's only consumer.
    unsafe { cell.register(&Waker::from(Arc::clone(&first))) };
    // SAFETY: registration returned and this thread remains the only consumer.
    let removed = unsafe { cell.take_waiter() };
    assert!(removed.is_some());
    assert!(!cell.has_registered_waiter());
    // SAFETY: the preceding removal released its claim.
    assert!(unsafe { cell.take_waiter() }.is_none());
    drop(removed);

    // SAFETY: all earlier consumer operations returned.
    unsafe { cell.register(&Waker::from(Arc::clone(&second))) };
    cell.complete(89);
    assert_eq!(cell.try_take_ready(), Some(89));
    assert_eq!(first.0.load(Ordering::SeqCst), 0);
    assert_eq!(second.0.load(Ordering::SeqCst), 1);
}

#[test]
fn removal_does_not_wait_for_result_publication() {
    let wake = Arc::new(CountingWake(std::sync::atomic::AtomicUsize::new(0)));
    let cell = Arc::new(ResultCell::<usize, Waker, GatedStateWord>::new());
    // SAFETY: this thread is the cell's only consumer.
    unsafe { cell.register(&Waker::from(Arc::clone(&wake))) };
    let ready_gate = Arc::clone(&cell.state.ready_gate);
    let producer_cell = Arc::clone(&cell);
    let producer = thread::spawn(move || producer_cell.complete(97));

    ready_gate.wait_until_reached();
    // SAFETY: registration returned and this thread remains the only consumer.
    let removed = unsafe { cell.take_waiter() };
    assert!(removed.is_some());
    assert!(!cell.has_registered_waiter());
    drop(removed);
    ready_gate.release();
    producer.join().expect("invariant: producer completes");

    assert_eq!(cell.try_take_ready(), Some(97));
    assert_eq!(cell.try_take_ready(), None);
    assert_eq!(wake.0.load(Ordering::SeqCst), 0);
}

#[test]
fn removed_waiter_drop_may_complete_the_same_cell() {
    let cell = Arc::new(ResultCell::<usize, ReentrantDropWaiter>::new());
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wakes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut waiter = ReentrantDropWaiter {
        completion: Some(ReentrantCompletion {
            cell: Arc::downgrade(&cell),
            calls: Arc::clone(&calls),
            value: 101,
        }),
        wakes: Arc::clone(&wakes),
    };
    // SAFETY: this thread is the cell's only consumer.
    unsafe { cell.register(&waiter) };
    waiter.completion = None;

    // SAFETY: registration returned and this thread remains the only consumer.
    let removed = unsafe { cell.take_waiter() };
    assert_eq!(cell.waiter_state.load(Ordering::Acquire), WAITER_IDLE);
    assert!(!cell.waiter_present.load(Ordering::Acquire));
    drop(removed);

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(wakes.load(Ordering::SeqCst), 0);
    assert_eq!(cell.try_take_ready(), Some(101));
}

#[test]
fn removed_waiter_drop_panic_does_not_strand_the_cell() {
    let panic_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wakes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let cell = ResultCell::<usize, PanicDropWaiter>::new();
    let mut waiter = PanicDropWaiter {
        panic_on_drop: true,
        panic_drops: Arc::clone(&panic_drops),
        wakes: Arc::clone(&wakes),
    };
    // SAFETY: this thread is the cell's only consumer.
    unsafe { cell.register(&waiter) };
    waiter.panic_on_drop = false;
    // SAFETY: registration returned and this thread remains the only consumer.
    let removed = unsafe { cell.take_waiter() };

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(removed)))
        .expect_err("removed waiter drop must panic");
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"replaced waiter drop"));
    assert_eq!(cell.state.load(Ordering::Acquire), RESULT_PENDING);
    assert_eq!(cell.waiter_state.load(Ordering::Acquire), WAITER_IDLE);
    assert!(!cell.waiter_present.load(Ordering::Acquire));
    cell.complete(103);
    assert_eq!(cell.try_take_ready(), Some(103));
    assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
    assert_eq!(wakes.load(Ordering::SeqCst), 0);
}
