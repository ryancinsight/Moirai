use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Wake, Waker};

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(tracing_subscriber::fmt().with_test_writer().finish())
}

struct CountingWake(AtomicUsize);

impl Wake for CountingWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn counter() -> (Arc<CountingWake>, Waker) {
    let count = Arc::new(CountingWake(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&count));
    (count, waker)
}

#[derive(Debug)]
struct Payload {
    value: usize,
    drops: Arc<AtomicUsize>,
}

impl Drop for Payload {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn bounded_receiver_drop_releases_accepted_payload_with_live_sender() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let drops = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = channel(1);
    sender
        .try_send(Payload {
            value: 0,
            drops: Arc::clone(&drops),
        })
        .expect("open admission");
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    drop(receiver);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert!(sender.is_closed());
}

#[test]
fn bounded_qualified_send_retains_its_unambiguous_public_api() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, mut receiver) = channel(1);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    let mut sending = Sender::send(&sender, 37);
    let result = Pin::new(&mut sending).poll(&mut Context::from_waker(Waker::noop()));
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(result, Poll::Ready(Ok(())));
    assert_eq!(receiver.try_recv(), Some(37));
}

#[test]
fn direct_poll_retains_and_replaces_the_actual_waker() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, mut receiver) = unbounded_channel();
    let (first, first_waker) = counter();
    let (second, second_waker) = counter();
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    assert_eq!(
        receiver.poll_recv(&mut Context::from_waker(&first_waker)),
        Poll::Pending
    );
    assert_eq!(
        receiver.poll_recv(&mut Context::from_waker(&second_waker)),
        Poll::Pending
    );
    assert_eq!(sender.enqueue(37), Ok(()));
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(first.0.load(Ordering::Relaxed), 0);
    assert_eq!(second.0.load(Ordering::Relaxed), 1);
    assert_eq!(
        receiver.poll_recv(&mut Context::from_waker(&second_waker)),
        Poll::Ready(Some(37))
    );
    assert_eq!(Arc::strong_count(&second), 2);
}

#[test]
fn last_sender_wakes_then_drains_to_permanent_closure() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, mut receiver) = unbounded_channel();
    let other = sender.clone();
    let (count, waker) = counter();
    let mut cx = Context::from_waker(&waker);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Pending);
    drop(sender);
    assert_eq!(count.0.load(Ordering::Relaxed), 0);
    assert_eq!(other.enqueue(11), Ok(()));
    assert_eq!(other.enqueue(19), Ok(()));
    drop(other);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(count.0.load(Ordering::Relaxed), 1);
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(Some(11)));
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(Some(19)));
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(None));
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(None));
    assert_eq!(Arc::strong_count(&count), 2);
}

#[test]
fn last_empty_sender_wakes_the_pending_receiver() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, mut receiver) = unbounded_channel::<usize>();
    let (count, waker) = counter();
    let mut cx = Context::from_waker(&waker);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Pending);
    drop(sender);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(count.0.load(Ordering::Relaxed), 1);
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(None));
}

#[test]
fn close_returns_exact_unsent_value_and_preserves_the_queue() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, mut receiver) = unbounded_channel();
    let value = String::from("owned protocol event");
    let identity = value.as_ptr();
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    assert_eq!(sender.enqueue(String::from("accepted")), Ok(()));
    receiver.close();
    let returned = sender
        .enqueue(value)
        .expect_err("closed admission must return ownership");
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(returned, "owned protocol event");
    assert_eq!(returned.as_ptr(), identity);
    assert!(sender.is_closed());
    assert_eq!(receiver.try_recv().as_deref(), Some("accepted"));
    assert_eq!(
        receiver.poll_recv(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(None)
    );
}

#[test]
fn receiver_drop_releases_queued_values_and_returns_new_payloads() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let drops = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = unbounded_channel();
    for value in 0..3 {
        sender
            .enqueue(Payload {
                value,
                drops: Arc::clone(&drops),
            })
            .expect("open admission");
    }
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    drop(receiver);
    let returned = sender
        .enqueue(Payload {
            value: 3,
            drops: Arc::clone(&drops),
        })
        .expect_err("dropped receiver must return ownership");
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(drops.load(Ordering::Relaxed), 3);
    assert_eq!(returned.value, 3);
    drop(returned);
    assert_eq!(drops.load(Ordering::Relaxed), 4);
}

#[test]
fn recv_future_cancellation_removes_only_the_waiter() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, mut receiver) = unbounded_channel();
    let (count, waker) = counter();
    let mut cx = Context::from_waker(&waker);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    let mut receive = receiver.recv();
    assert_eq!(Pin::new(&mut receive).poll(&mut cx), Poll::Pending);
    assert_eq!(sender.enqueue(23), Ok(()));
    drop(receive);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(count.0.load(Ordering::Relaxed), 1);
    assert_eq!(Arc::strong_count(&count), 2);
    assert_eq!(receiver.try_recv(), Some(23));
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Pending);
    assert_eq!(sender.enqueue(29), Ok(()));
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(Some(29)));
}

#[test]
fn unpolled_recv_future_preserves_a_direct_poll_registration() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, mut receiver) = unbounded_channel();
    let (count, waker) = counter();
    let mut cx = Context::from_waker(&waker);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Pending);
    drop(receiver.recv());
    assert_eq!(sender.enqueue(31), Ok(()));
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(count.0.load(Ordering::Relaxed), 1);
    assert_eq!(receiver.poll_recv(&mut cx), Poll::Ready(Some(31)));
}

#[test]
fn receiver_drop_clears_a_direct_poll_registration() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, mut receiver) = unbounded_channel::<usize>();
    let (count, waker) = counter();
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    assert_eq!(
        receiver.poll_recv(&mut Context::from_waker(&waker)),
        Poll::Pending
    );
    drop(receiver);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(Arc::strong_count(&count), 2);
    assert_eq!(sender.enqueue(41), Err(41));
}

#[test]
fn producers_preserve_every_payload_and_each_producers_fifo_order() {
    let _trace = trace();
    for producer_count in [1, 2, 4] {
        for message_count in [0, 1, 2, 17] {
            let _case = tracing::info_span!("case", producer_count, message_count).entered();
            let _arrange = tracing::info_span!("arrange").entered();
            let (sender, mut receiver) = unbounded_channel();
            let mut next = vec![0; producer_count];
            drop(_arrange);
            let _act = tracing::info_span!("act").entered();
            std::thread::scope(|scope| {
                for producer in 0..producer_count {
                    let sender = sender.clone();
                    scope.spawn(move || {
                        for sequence in 0..message_count {
                            assert_eq!(sender.enqueue((producer, sequence)), Ok(()));
                        }
                    });
                }
            });
            drop(sender);
            drop(_act);
            let _assert = tracing::info_span!("assert").entered();
            let mut total = 0;
            while let Some((producer, sequence)) = receiver.try_recv() {
                assert_eq!(sequence, next[producer]);
                next[producer] += 1;
                total += 1;
            }
            assert_eq!(next, vec![message_count; producer_count]);
            assert_eq!(total, producer_count * message_count);
            assert_eq!(
                receiver.poll_recv(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(None)
            );
        }
    }
}

#[test]
fn queued_payload_destructor_can_reenter_a_surviving_sender() {
    let _trace = trace();
    struct Reenter {
        sender: Sender<Reenter, Unbounded>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Reenter {
        fn drop(&mut self) {
            assert!(self.sender.is_closed());
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }
    let _arrange = tracing::info_span!("arrange").entered();
    let (sender, receiver) = unbounded_channel();
    let drops = Arc::new(AtomicUsize::new(0));
    let payload = Reenter {
        sender: sender.clone(),
        drops: Arc::clone(&drops),
    };
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    assert!(sender.enqueue(payload).is_ok());
    drop(receiver);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(sender.sender_strong_count(), 1);
}

#[test]
fn payload_unwind_cannot_bypass_any_parked_sender_wake() {
    let _trace = trace();
    #[derive(Debug)]
    struct UnwindPayload {
        value: Box<usize>,
        panic_on_drop: bool,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for UnwindPayload {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            if self.panic_on_drop {
                panic!("queued payload destructor panic");
            }
        }
    }
    for capacity in [1, 3] {
        for parked in [1, 3] {
            let _case = tracing::info_span!("case", capacity, parked).entered();
            let _arrange = tracing::info_span!("arrange").entered();
            let (sender, receiver) = channel(capacity);
            let drops = Arc::new(AtomicUsize::new(0));
            for value in 0..capacity {
                sender
                    .try_send(UnwindPayload {
                        value: Box::new(value),
                        panic_on_drop: value == 0,
                        drops: Arc::clone(&drops),
                    })
                    .expect("accepted queued payload");
            }
            let counters: Vec<_> = (0..parked).map(|_| counter()).collect();
            let mut sending = Vec::new();
            let mut identities = Vec::new();
            for (index, (_, waker)) in counters.iter().enumerate() {
                let value = Box::new(capacity + index);
                identities.push(std::ptr::from_ref(&*value));
                let mut future = sender.send(UnwindPayload {
                    value,
                    panic_on_drop: false,
                    drops: Arc::clone(&drops),
                });
                assert!(
                    Pin::new(&mut future)
                        .poll(&mut Context::from_waker(waker))
                        .is_pending()
                );
                sending.push(future);
            }
            drop(_arrange);
            let _act = tracing::info_span!("act").entered();
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(receiver)))
                .expect_err("the payload panic must propagate");
            drop(_act);
            let _assert = tracing::info_span!("assert").entered();
            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"queued payload destructor panic")
            );
            assert!(sender.is_closed());
            assert_eq!(drops.load(Ordering::SeqCst), capacity);
            for (index, (future, (count, waker))) in sending.iter_mut().zip(&counters).enumerate() {
                assert_eq!(count.0.load(Ordering::Relaxed), 1);
                let Poll::Ready(Err(returned)) =
                    Pin::new(future).poll(&mut Context::from_waker(waker))
                else {
                    panic!("closed channel must return the parked payload");
                };
                assert_eq!(*returned.value, capacity + index);
                assert_eq!(std::ptr::from_ref(&*returned.value), identities[index]);
                drop(returned);
            }
            assert_eq!(drops.load(Ordering::SeqCst), capacity + parked);
        }
    }
}
