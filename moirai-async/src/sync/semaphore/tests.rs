//! Cancelling a pending acquire must neither lose a permit nor lose the wake
//! owed to the next waiter.

use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Wake, Waker};

/// Counts wakes delivered to one waiter.
#[derive(Default)]
struct WakeCount(AtomicUsize);

impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl WakeCount {
    fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

fn poll_acquire<'a>(
    acquire: &mut Pin<Box<SemaphoreAcquire<'a>>>,
    wakes: &Arc<WakeCount>,
) -> Poll<SemaphorePermit<'a>> {
    let waker = Waker::from(Arc::clone(wakes));
    acquire.as_mut().poll(&mut Context::from_waker(&waker))
}

#[test]
fn dropped_pending_acquire_passes_the_next_permit_to_the_following_waiter() {
    let semaphore = Semaphore::new(1);
    let held = semaphore.try_acquire().expect("one permit is free");

    let first_wakes = Arc::new(WakeCount::default());
    let mut first = Box::pin(semaphore.acquire());
    assert!(poll_acquire(&mut first, &first_wakes).is_pending());
    let second_wakes = Arc::new(WakeCount::default());
    let mut second = Box::pin(semaphore.acquire());
    assert!(poll_acquire(&mut second, &second_wakes).is_pending());

    drop(first);
    drop(held);

    assert_eq!(
        first_wakes.count(),
        0,
        "a dropped waiter must not be granted"
    );
    assert_eq!(
        second_wakes.count(),
        1,
        "the surviving waiter must be woken"
    );
    let Poll::Ready(permit) = poll_acquire(&mut second, &second_wakes) else {
        panic!("the woken waiter must hold the released permit");
    };
    assert_eq!(semaphore.available_permits(), 0);
    drop(permit);
    assert_eq!(semaphore.available_permits(), 1);
}

#[test]
fn dropped_granted_acquire_hands_its_permit_to_the_following_waiter() {
    let semaphore = Semaphore::new(1);
    let held = semaphore.try_acquire().expect("one permit is free");

    let first_wakes = Arc::new(WakeCount::default());
    let mut first = Box::pin(semaphore.acquire());
    assert!(poll_acquire(&mut first, &first_wakes).is_pending());
    let second_wakes = Arc::new(WakeCount::default());
    let mut second = Box::pin(semaphore.acquire());
    assert!(poll_acquire(&mut second, &second_wakes).is_pending());

    // The release grants the oldest waiter, which is dropped before it polls
    // and takes the grant.
    drop(held);
    assert_eq!(first_wakes.count(), 1);
    drop(first);

    assert_eq!(second_wakes.count(), 1, "the unconsumed grant must move on");
    let Poll::Ready(permit) = poll_acquire(&mut second, &second_wakes) else {
        panic!("the woken waiter must hold the forwarded permit");
    };
    drop(permit);
    assert_eq!(semaphore.available_permits(), 1);
}

#[test]
fn dropped_last_waiter_returns_an_unconsumed_grant_to_the_pool() {
    let semaphore = Semaphore::new(1);
    let held = semaphore.try_acquire().expect("one permit is free");
    let wakes = Arc::new(WakeCount::default());
    let mut waiter = Box::pin(semaphore.acquire());
    assert!(poll_acquire(&mut waiter, &wakes).is_pending());

    drop(held);
    assert_eq!(wakes.count(), 1);
    drop(waiter);

    assert_eq!(semaphore.available_permits(), 1);
    assert!(semaphore.try_acquire().is_some());
}
