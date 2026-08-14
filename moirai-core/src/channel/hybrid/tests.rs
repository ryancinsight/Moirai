#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::*;
use core::mem::size_of;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};

#[test]
fn hybrid_channel_factory_is_zero_sized() {
    assert_eq!(size_of::<HybridChannel<u8>>(), 0);
}

#[test]
fn test_recv_future_waker_cleanup_on_drop() {
    use std::task::{RawWaker, RawWakerVTable, Waker};

    fn dummy_raw_waker() -> RawWaker {
        fn clone_raw(_: *const ()) -> RawWaker {
            dummy_raw_waker()
        }
        fn wake_raw(_: *const ()) {}
        fn wake_by_ref_raw(_: *const ()) {}
        fn drop_raw(_: *const ()) {}
        static VTABLE: RawWakerVTable =
            RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }

    let (_tx, rx) = HybridChannel::<i32>::new(4);
    let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
    let mut cx = Context::from_waker(&waker);

    {
        let mut fut = rx.recv_async();
        let mut pinned = Pin::new(&mut fut);
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);

        // Waker should be registered
        let wakers = rx.async_wakers.lock().unwrap();
        assert_eq!(wakers.len(), 1);
    }

    // After dropping the future, the waker list must be cleaned up
    let wakers = rx.async_wakers.lock().unwrap();
    assert_eq!(wakers.len(), 0);
}

#[test]
fn test_hybrid_channel_lost_wakeup() {
    use std::task::{RawWaker, RawWakerVTable, Waker};

    fn dummy_raw_waker() -> RawWaker {
        fn clone_raw(_: *const ()) -> RawWaker {
            dummy_raw_waker()
        }
        fn wake_raw(_: *const ()) {}
        fn wake_by_ref_raw(_: *const ()) {}
        fn drop_raw(_: *const ()) {}
        static VTABLE: RawWakerVTable =
            RawWakerVTable::new(clone_raw, wake_raw, wake_by_ref_raw, drop_raw);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }

    let (tx, rx) = HybridChannel::<i32>::new(4);
    let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
    let mut cx = Context::from_waker(&waker);

    let mut fut = rx.recv_async();
    let mut pinned = Pin::new(&mut fut);

    // 1. Initial poll registers the waker. waker_count becomes 1.
    assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);
    assert_eq!(rx.waker_count.load(Ordering::Relaxed), 1);

    // 2. Sender sends an item, which drains the wakers list and wakes the future.
    // waker_count becomes 0.
    tx.send(42).unwrap();
    assert_eq!(rx.waker_count.load(Ordering::Relaxed), 0);

    // 3. Consume the item externally, so the ring buffer is empty again.
    assert_eq!(rx.try_recv().unwrap(), 42);

    // 4. Poll the future again. This should re-register the waker under the same ID.
    // If the bug is present, waker_count will remain 0. If fixed, it becomes 1.
    assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);
    assert_eq!(rx.waker_count.load(Ordering::Relaxed), 1);
}
