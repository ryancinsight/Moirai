use super::*;
use crate::net::AsyncUdpSocket;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
struct CountWake(AtomicUsize);

impl Wake for CountWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn shared_waiters_preserve_notifications_and_cancellation() {
    let reactor = IoReactor::new().expect("reactor");
    reactor.with_active(|| {
        let socket = AsyncUdpSocket::from_std(UdpSocket::bind("127.0.0.1:0").expect("socket"))
            .expect("nonblocking socket");
        let mut first = socket.writable();
        let mut second = socket.writable();
        let first_wake = Arc::new(CountWake::default());
        let second_wake = Arc::new(CountWake::default());
        let first_waker = Waker::from(Arc::clone(&first_wake));
        let second_waker = Waker::from(Arc::clone(&second_wake));
        assert!(
            first
                .poll_ready(&mut Context::from_waker(&first_waker))
                .is_pending()
        );
        assert!(
            second
                .poll_ready(&mut Context::from_waker(&second_waker))
                .is_pending()
        );
        assert_eq!(
            lock(
                &socket.readiness[1]
                    .get()
                    .expect("writable group initialized")
                    .signal
                    .slots
            )
            .len(),
            2
        );
        reactor
            .run_iteration(Some(Duration::from_secs(1)))
            .expect("dispatch writable");
        assert_eq!(first_wake.0.load(Ordering::Relaxed), 1);
        assert_eq!(second_wake.0.load(Ordering::Relaxed), 1);
        assert!(matches!(
            first.poll_ready(&mut Context::from_waker(&first_waker)),
            Poll::Ready(Ok(()))
        ));
        // Rearm first, then consume second's old notification. The old event
        // must not retire the new registration and strand first.
        assert!(
            first
                .poll_ready(&mut Context::from_waker(&first_waker))
                .is_pending()
        );
        assert!(matches!(
            second.poll_ready(&mut Context::from_waker(&second_waker)),
            Poll::Ready(Ok(()))
        ));
        assert!(
            lock(
                &socket.readiness[1]
                    .get()
                    .expect("writable group initialized")
                    .registration
            )
            .is_some()
        );
        drop(second);
        assert_eq!(
            lock(
                &socket.readiness[1]
                    .get()
                    .expect("writable group initialized")
                    .signal
                    .slots
            )
            .len(),
            1
        );
        reactor
            .run_iteration(Some(Duration::from_secs(1)))
            .expect("dispatch rearmed writable");
        assert_eq!(first_wake.0.load(Ordering::Relaxed), 2);
        drop(first);
        assert_eq!(
            lock(
                &socket.readiness[1]
                    .get()
                    .expect("writable group initialized")
                    .signal
                    .slots
            )
            .len(),
            0
        );
        assert!(
            lock(
                &socket.readiness[1]
                    .get()
                    .expect("writable group initialized")
                    .registration
            )
            .is_none()
        );
    });
}

#[test]
fn batched_receive_rearms_and_preserves_payload_and_address() {
    let reactor = IoReactor::new().expect("reactor");
    reactor.with_active(|| {
        let socket = AsyncUdpSocket::from_std(UdpSocket::bind("127.0.0.1:0").expect("receiver"))
            .expect("nonblocking receiver");
        let sender = UdpSocket::bind("127.0.0.1:0").expect("sender");
        let mut poller = socket.recv_poller();
        let wake = Waker::from(Arc::new(CountWake::default()));
        let mut cx = Context::from_waker(&wake);
        let mut bytes = [0; 37];
        assert!(
            poller
                .poll_io(&mut cx, || socket.as_std().recv_from(&mut bytes))
                .is_pending()
        );
        let expected: Vec<_> = (0..37).map(|index| index * 7 % 251).collect();
        assert_eq!(
            sender
                .send_to(&expected, socket.local_addr().expect("receiver address"))
                .expect("send"),
            expected.len()
        );
        reactor
            .run_iteration(Some(Duration::from_secs(1)))
            .expect("dispatch readable");
        let Poll::Ready(Ok((length, address))) =
            poller.poll_io(&mut cx, || socket.as_std().recv_from(&mut bytes))
        else {
            panic!("receive must produce a datagram");
        };
        assert_eq!(&bytes[..length], expected);
        assert_eq!(address, sender.local_addr().expect("sender address"));
        assert!(
            poller
                .poll_io(&mut cx, || socket.as_std().recv_from(&mut bytes))
                .is_pending()
        );
        drop(poller);
        assert!(
            lock(
                &socket.readiness[0]
                    .get()
                    .expect("receive group initialized")
                    .registration
            )
            .is_none()
        );
    });
}

#[test]
fn late_subscriber_cannot_observe_a_consumed_but_unretired_notification() {
    let reactor = IoReactor::new().expect("reactor");
    reactor.with_active(|| {
        let socket = AsyncUdpSocket::from_std(UdpSocket::bind("127.0.0.1:0").expect("socket"))
            .expect("nonblocking socket");
        let mut first = socket.writable();
        let first_slot = Arc::clone(&first.slot);
        let wake = Waker::from(Arc::new(CountWake::default()));
        assert!(
            first
                .poll_ready(&mut Context::from_waker(&wake))
                .is_pending()
        );
        reactor
            .run_iteration(Some(Duration::from_secs(1)))
            .expect("writable dispatch");
        let group = socket.readiness[1]
            .get()
            .expect("writable group initialized");
        let registration = lock(&group.registration);
        let reader = std::thread::spawn(move || first.poll_ready(&mut Context::from_waker(&wake)));
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while lock(&first_slot.state).0 {
            assert!(
                std::time::Instant::now() < deadline,
                "ready subscriber must enter its poll"
            );
            std::thread::yield_now();
        }
        // The first subscriber consumes its private flag but cannot clear the
        // shared notification while another thread owns the registration lock.
        assert!(group.signal.dispatched.load(Ordering::Acquire));
        let mut late = socket.writable();
        assert!(
            lock(&late.slot.state).0,
            "late subscriber retains the dispatched event"
        );
        drop(registration);
        assert!(matches!(
            reader.join().expect("ready reader"),
            Poll::Ready(Ok(()))
        ));
        let wake = Waker::from(Arc::new(CountWake::default()));
        assert!(matches!(
            late.poll_ready(&mut Context::from_waker(&wake)),
            Poll::Ready(Ok(()))
        ));
    });
}
