//! The future-creation boundary used by state-locked protocol drivers.

use crate::sync::Notify;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

fn poll<F: Future + Unpin>(future: &mut F) -> Poll<F::Output> {
    Pin::new(future).poll(&mut Context::from_waker(Waker::noop()))
}

#[test]
fn broadcast_between_creation_and_first_poll_is_observed() {
    let notify = Notify::new();
    let mut future = notify.notified();
    notify.notify_waiters();
    assert!(poll(&mut future).is_ready());
}

#[test]
fn broadcast_reaches_every_preexisting_unpolled_future() {
    let notify = Notify::new();
    let mut futures = std::array::from_fn::<_, 8, _>(|_| notify.notified());
    notify.notify_waiters();
    for future in &mut futures {
        assert!(poll(future).is_ready());
    }
}

#[test]
fn future_created_after_broadcast_still_waits() {
    let notify = Notify::new();
    notify.notify_waiters();
    assert!(poll(&mut notify.notified()).is_pending());
}

#[test]
fn unpolled_broadcast_does_not_consume_a_single_permit() {
    let notify = Notify::new();
    let mut broadcast = notify.notified();
    notify.notify_one();
    notify.notify_waiters();
    assert!(poll(&mut broadcast).is_ready());
    assert!(poll(&mut notify.notified()).is_ready());
    assert!(poll(&mut notify.notified()).is_pending());
}

#[test]
fn registered_broadcast_does_not_consume_a_single_permit() {
    let notify = Notify::new();
    let mut broadcast = notify.notified();
    assert!(poll(&mut broadcast).is_pending());
    notify.notify_waiters();
    notify.notify_one();
    assert!(poll(&mut broadcast).is_ready());
    assert!(poll(&mut notify.notified()).is_ready());
    assert!(poll(&mut notify.notified()).is_pending());
}

#[test]
fn registered_single_grant_does_not_consume_a_second_stored_permit() {
    let notify = Notify::new();
    let mut first = notify.notified();
    assert!(poll(&mut first).is_pending());
    notify.notify_one();
    notify.notify_one();
    assert!(poll(&mut first).is_ready());
    assert!(poll(&mut notify.notified()).is_ready());
    assert!(poll(&mut notify.notified()).is_pending());
}

#[derive(Default)]
struct WakeCount(AtomicUsize);

impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn broadcast_wakes_registered_waiter_and_unpolled_future() {
    let notify = Notify::new();
    let mut registered = notify.notified();
    let mut unpolled = notify.notified();
    let count = Arc::new(WakeCount::default());
    let waker = Waker::from(Arc::clone(&count));
    assert!(
        Pin::new(&mut registered)
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    notify.notify_waiters();
    assert_eq!(count.0.load(Ordering::Relaxed), 1);
    assert!(poll(&mut registered).is_ready());
    assert!(poll(&mut unpolled).is_ready());
}

#[test]
fn canceled_single_grant_transfers_to_next_registered_waiter() {
    let notify = Notify::new();
    let mut first = notify.notified();
    let mut next = notify.notified();
    assert!(poll(&mut first).is_pending());
    assert!(poll(&mut next).is_pending());
    notify.notify_one();
    drop(first);
    assert!(poll(&mut next).is_ready());
    assert!(poll(&mut notify.notified()).is_pending());
}

#[test]
fn canceled_broadcast_does_not_create_single_permit() {
    let notify = Notify::new();
    let mut canceled = notify.notified();
    assert!(poll(&mut canceled).is_pending());
    notify.notify_waiters();
    drop(canceled);
    assert!(poll(&mut notify.notified()).is_pending());
}

#[test]
fn dropping_unpolled_broadcast_preserves_single_permit() {
    let notify = Notify::new();
    let canceled = notify.notified();
    notify.notify_waiters();
    notify.notify_one();
    drop(canceled);
    assert!(poll(&mut notify.notified()).is_ready());
    assert!(poll(&mut notify.notified()).is_pending());
}

#[test]
fn broadcast_generations_do_not_leak_to_later_futures() {
    let notify = Notify::new();
    for _ in 0..32 {
        let mut before = notify.notified();
        notify.notify_waiters();
        assert!(poll(&mut before).is_ready());
        assert!(poll(&mut notify.notified()).is_pending());
    }
}
