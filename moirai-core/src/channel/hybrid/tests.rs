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

/// `Waker::wake` may execute the woken task inline (the executor polls inline
/// under injector saturation), and that task may re-enter this channel's waker
/// registry. A wake dispatched while `notify_consumers` still holds the
/// registry lock would therefore self-deadlock. The probe waker re-locks the
/// registry from inside `wake` via `try_lock` and records a violation instead
/// of deadlocking, so the assertion fails fast if the lock discipline
/// regresses.
#[test]
fn send_wakes_outside_the_waker_registry_lock() {
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};
    use std::task::{Wake, Waker};

    struct RegistryProbe {
        registry: Arc<Mutex<Vec<(u64, Waker)>>>,
        woken: AtomicBool,
        held_during_wake: AtomicBool,
    }

    impl Wake for RegistryProbe {
        fn wake(self: Arc<Self>) {
            self.woken.store(true, Ordering::SeqCst);
            if self.registry.try_lock().is_err() {
                self.held_during_wake.store(true, Ordering::SeqCst);
            }
        }
    }

    let (tx, rx) = HybridChannel::<i32>::new(4);
    let probe = Arc::new(RegistryProbe {
        registry: Arc::clone(&rx.async_wakers),
        woken: AtomicBool::new(false),
        held_during_wake: AtomicBool::new(false),
    });
    let waker = Waker::from(Arc::clone(&probe));
    let mut cx = Context::from_waker(&waker);

    let mut fut = rx.recv_async();
    assert_eq!(Pin::new(&mut fut).poll(&mut cx), Poll::Pending);

    tx.send(9).unwrap();
    assert!(
        probe.woken.load(Ordering::SeqCst),
        "send must wake the future"
    );
    assert!(
        !probe.held_during_wake.load(Ordering::SeqCst),
        "wake ran while notify_consumers held the waker registry lock"
    );
    assert_eq!(Pin::new(&mut fut).poll(&mut cx), Poll::Ready(Ok(9)));
}

/// Last-message delivery under blocking park/unpark churn.
///
/// Each round races one `send` against one parked-path `recv` from a barrier,
/// exercising the register-then-recheck window on the receiver against the
/// produce-then-gate window on the sender. A missed wake parks the receiver
/// against a delivered message forever; the nextest budget then terminates
/// the hang. All synchronization is the barrier and the channel itself — no
/// sleeps, no wall-clock assertions.
#[test]
fn recv_delivers_last_message_under_park_unpark_churn() {
    use std::sync::{Arc, Barrier};

    const ROUNDS: usize = 4096;

    let (tx, rx) = HybridChannel::<usize>::new(2);
    let barrier = Arc::new(Barrier::new(2));

    let receiver_barrier = Arc::clone(&barrier);
    let receiver = std::thread::spawn(move || {
        for round in 0..ROUNDS {
            receiver_barrier.wait();
            assert_eq!(rx.recv().unwrap(), round, "round {round} lost its message");
        }
    });

    for round in 0..ROUNDS {
        barrier.wait();
        tx.send(round).unwrap();
    }
    receiver.join().unwrap();
}

/// The same churn through `recv_timeout`: a missed wake there costs the whole
/// remaining timeout for a message that already sits in the ring. The timeout
/// is a failure bound only — a correct run never parks longer than one
/// unpark, and no assertion reads the clock.
#[test]
fn recv_timeout_observes_message_racing_registration() {
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    const ROUNDS: usize = 1024;

    let (tx, rx) = HybridChannel::<usize>::new(2);
    let barrier = Arc::new(Barrier::new(2));

    let receiver_barrier = Arc::clone(&barrier);
    let receiver = std::thread::spawn(move || {
        for round in 0..ROUNDS {
            receiver_barrier.wait();
            assert_eq!(
                rx.recv_timeout(Duration::from_secs(30)).unwrap(),
                round,
                "round {round} lost its message"
            );
        }
    });

    for round in 0..ROUNDS {
        barrier.wait();
        tx.send(round).unwrap();
    }
    receiver.join().unwrap();
}

/// Async last-message delivery: each round polls to `Pending` (registering the
/// waker), races a `send` from the barrier, then waits for the wake by
/// spinning on the probe counter — an event, not the clock. Before the
/// register-then-recheck fix a send interleaved between the future's re-check
/// and its registration was missed by both sides, leaving a registered waker
/// nobody would ever wake.
#[test]
fn recv_future_delivers_last_message_under_wake_churn() {
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Barrier};
    use std::task::{Wake, Waker};

    const ROUNDS: usize = 4096;

    struct CountingWaker {
        wakes: AtomicUsize,
    }

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }

    let (tx, rx) = HybridChannel::<usize>::new(2);
    let barrier = Arc::new(Barrier::new(2));

    let sender_barrier = Arc::clone(&barrier);
    let sender = std::thread::spawn(move || {
        for round in 0..ROUNDS {
            sender_barrier.wait();
            tx.send(round).unwrap();
        }
    });

    let probe = Arc::new(CountingWaker {
        wakes: AtomicUsize::new(0),
    });
    let waker = Waker::from(Arc::clone(&probe));
    let mut cx = Context::from_waker(&waker);

    for round in 0..ROUNDS {
        let mut fut = rx.recv_async();
        let mut pinned = Pin::new(&mut fut);
        barrier.wait();
        loop {
            let wakes_before = probe.wakes.load(Ordering::SeqCst);
            match pinned.as_mut().poll(&mut cx) {
                Poll::Ready(value) => {
                    assert_eq!(value.unwrap(), round, "round {round} lost its message");
                    break;
                }
                Poll::Pending => {
                    // Wait for the wake event; a lost wake spins here until
                    // the nextest budget terminates the test.
                    while probe.wakes.load(Ordering::SeqCst) == wakes_before {
                        std::hint::spin_loop();
                    }
                }
            }
        }
    }
    sender.join().unwrap();
}
