//! One-shot readiness consumption and stale-generation event rejection.

use super::super::core::{FdKey, IoReactor};
use super::harness::{WakeCount, socket_to_raw};
use crate::{Event, Interest, Reactor};
use std::net::UdpSocket;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::Waker;
use std::time::Duration;

#[test]
#[cfg(any(unix, windows))]
fn readiness_delivery_consumes_only_reported_interest() {
    let reactor = IoReactor::new().expect("reactor");
    let receiver = UdpSocket::bind("127.0.0.1:0").expect("receiver bind");
    receiver
        .set_nonblocking(true)
        .expect("receiver nonblocking");
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    let fd = socket_to_raw(&receiver);
    let read_count = Arc::new(WakeCount::default());
    let write_count = Arc::new(WakeCount::default());

    reactor
        .register_waker(fd, Interest::READABLE, Waker::from(Arc::clone(&read_count)))
        .expect("register read interest");
    reactor
        .register_waker(
            fd,
            Interest::WRITABLE,
            Waker::from(Arc::clone(&write_count)),
        )
        .expect("register write interest");

    reactor
        .wake_fd_waiters(Event {
            fd,
            readable: false,
            writable: true,
            error: false,
            hangup: false,
        })
        .expect("consume write readiness");
    assert_eq!(write_count.0.load(Ordering::Relaxed), 1);
    assert_eq!(read_count.0.load(Ordering::Relaxed), 0);
    {
        let fds = reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let remaining = fds.get(&FdKey::from(fd)).expect("read interest remains");
        assert!(remaining.interest.readable);
        assert!(!remaining.interest.writable);
    }

    sender
        .send_to(b"ready", receiver.local_addr().expect("receiver address"))
        .expect("send readiness payload");
    for _ in 0..2 {
        reactor
            .run_iteration(Some(Duration::from_secs(1)))
            .expect("process read readiness");
        if read_count.0.load(Ordering::Relaxed) == 1 {
            break;
        }
    }
    assert_eq!(read_count.0.load(Ordering::Relaxed), 1);
    assert!(
        !reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd))
    );

    let residual = reactor
        .platform_reactor
        .poll_events(Some(Duration::ZERO))
        .expect("poll after one-shot consumption");
    assert!(
        residual
            .iter()
            .all(|event| FdKey::from(event.fd) != FdKey::from(fd)),
        "consumed descriptor must be absent from the platform poll set"
    );
}

#[test]
#[cfg(any(unix, windows))]
fn stale_polled_generation_cannot_consume_replacement_registration() {
    let reactor = IoReactor::new().expect("reactor");
    let receiver = UdpSocket::bind("127.0.0.1:0").expect("receiver bind");
    receiver
        .set_nonblocking(true)
        .expect("receiver nonblocking");
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    let fd = socket_to_raw(&receiver);
    let replaced_count = Arc::new(WakeCount::default());
    let current_count = Arc::new(WakeCount::default());

    reactor
        .register_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&replaced_count)),
        )
        .expect("register replaced interest");
    sender
        .send_to(b"stale", receiver.local_addr().expect("receiver address"))
        .expect("send readiness payload");
    let stale_event = reactor
        .platform_reactor
        .poll_registered_events(Some(Duration::from_secs(1)))
        .expect("poll replaced readiness")
        .into_iter()
        .find(|event| FdKey::from(event.descriptor()) == FdKey::from(fd))
        .expect("replaced descriptor is readable");

    reactor
        .register_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&current_count)),
        )
        .expect("register current interest");
    reactor
        .handle_polled_event(stale_event)
        .expect("discard stale readiness");

    assert_eq!(replaced_count.0.load(Ordering::Relaxed), 0);
    assert_eq!(current_count.0.load(Ordering::Relaxed), 0);
    let fds = reactor
        .registered_fds
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let current = fds
        .get(&FdKey::from(fd))
        .expect("replacement registration remains");
    assert!(current.interest.readable);
    assert!(!current.interest.writable);
    drop(fds);

    reactor
        .run_iteration(Some(Duration::from_secs(1)))
        .expect("current readiness must remain dispatchable");
    assert_eq!(current_count.0.load(Ordering::Relaxed), 1);
    let mut payload = [0_u8; 5];
    assert_eq!(
        receiver
            .recv(&mut payload)
            .expect("receive retained readiness payload"),
        5
    );
    assert_eq!(&payload, b"stale");
}
