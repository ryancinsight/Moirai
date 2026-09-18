//! Windows owned-waiter cancellation: interest-specific and reactor-bound
//! scoping, stale-generation replacement, and displaced-waker drop timing.

use super::super::core::{FdKey, IoReactor};
use super::super::socket_owner::SocketLease;
use super::harness::{DropObservation, DropObservingWake, WakeCount};
use crate::Interest;
use std::net::UdpSocket;
use std::os::windows::io::AsRawSocket;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;
use std::time::Duration;

#[test]
#[cfg(windows)]
fn cancellation_is_interest_specific_and_reactor_bound() {
    let reactor_a = Arc::new(IoReactor::new().expect("reactor A"));
    let reactor_b = Arc::new(IoReactor::new().expect("reactor B"));
    let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("socket bind"));
    socket.set_nonblocking(true).expect("socket nonblocking");
    let fd = socket.as_raw_socket() as crate::RawFd;
    let read_wake = Arc::new(WakeCount::default());
    let write_wake = Arc::new(WakeCount::default());

    let read = reactor_a.with_active(|| {
        reactor_a
            .register_owned_waker(
                fd,
                Interest::READABLE,
                Waker::from(Arc::clone(&read_wake)),
                SocketLease::from(&socket),
            )
            .expect("read registration")
    });
    let write_waker = Waker::from(Arc::clone(&write_wake));
    let write = reactor_a
        .register_owned_waker(
            fd,
            Interest::WRITABLE,
            write_waker.clone(),
            SocketLease::from(&socket),
        )
        .expect("write registration");

    let dropping_reactor = Arc::clone(&reactor_b);
    std::thread::spawn(move || dropping_reactor.with_active(|| drop(read)))
        .join()
        .expect("migrated cancellation");

    let central = reactor_a
        .registered_fds
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let current = central
        .get(&FdKey::from(fd))
        .expect("write remains registered");
    assert!(!current.interest.readable);
    assert!(current.interest.writable);
    assert!(current.read_waker.is_none());
    let retained = current.write_waker.as_ref().expect("write waker retained");
    assert!(retained.will_wake(&write_waker));
    drop(central);
    assert_eq!(read_wake.0.load(Ordering::Relaxed), 0);

    reactor_a
        .run_iteration(Some(Duration::from_secs(1)))
        .expect("write readiness dispatch");
    assert_eq!(write_wake.0.load(Ordering::Relaxed), 1);
    drop(write);

    let receiver = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("receiver bind"));
    receiver
        .set_nonblocking(true)
        .expect("receiver nonblocking");
    let receiver_fd = receiver.as_raw_socket() as crate::RawFd;
    let retired_write = reactor_a
        .register_owned_waker(
            receiver_fd,
            Interest::WRITABLE,
            Waker::noop().clone(),
            SocketLease::from(&receiver),
        )
        .expect("retired write registration");
    let read_wake = Arc::new(WakeCount::default());
    let read = reactor_a
        .register_owned_waker(
            receiver_fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&read_wake)),
            SocketLease::from(&receiver),
        )
        .expect("read replacement registration");
    drop(retired_write);
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    sender
        .send_to(b"read", receiver.local_addr().expect("receiver address"))
        .expect("readiness payload");
    reactor_a
        .run_iteration(Some(Duration::from_secs(1)))
        .expect("read readiness dispatch");
    assert_eq!(read_wake.0.load(Ordering::Relaxed), 1);
    let mut payload = [0_u8; 4];
    assert_eq!(receiver.recv(&mut payload).expect("payload receive"), 4);
    assert_eq!(&payload, b"read");
    drop(read);
}

#[test]
#[cfg(windows)]
fn stale_same_interest_cancellation_preserves_replacement() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("receiver bind"));
    socket.set_nonblocking(true).expect("receiver nonblocking");
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    let fd = socket.as_raw_socket() as crate::RawFd;
    let old_wake = Arc::new(WakeCount::default());
    let current_wake = Arc::new(WakeCount::default());
    let old = reactor
        .register_owned_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&old_wake)),
            SocketLease::from(&socket),
        )
        .expect("old read registration");
    sender
        .send_to(b"new", socket.local_addr().expect("receiver address"))
        .expect("readiness payload");
    let stale_event = reactor
        .platform_reactor
        .poll_registered_events(Some(Duration::from_secs(1)))
        .expect("stale readiness poll")
        .into_iter()
        .find(|event| event.descriptor() == fd)
        .expect("stale readiness event");
    let current = reactor
        .register_owned_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&current_wake)),
            SocketLease::from(&socket),
        )
        .expect("replacement read registration");
    drop(old);
    reactor
        .handle_polled_event(stale_event)
        .expect("discard stale readiness");
    assert_eq!(old_wake.0.load(Ordering::Relaxed), 0);
    assert_eq!(current_wake.0.load(Ordering::Relaxed), 0);

    reactor
        .run_iteration(Some(Duration::from_secs(1)))
        .expect("replacement readiness dispatch");
    assert_eq!(current_wake.0.load(Ordering::Relaxed), 1);
    let mut payload = [0_u8; 3];
    assert_eq!(socket.recv(&mut payload).expect("payload receive"), 3);
    assert_eq!(&payload, b"new");
    drop(current);
}

#[test]
#[cfg(windows)]
fn replacing_owned_waiter_destroys_displaced_waker_unlocked() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("socket bind"));
    socket.set_nonblocking(true).expect("socket nonblocking");
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    let fd = socket.as_raw_socket() as crate::RawFd;
    let observation = Arc::new(DropObservation {
        dropped: AtomicBool::new(false),
        dropped_while_locked: AtomicBool::new(false),
    });
    let old_waker = Arc::new(DropObservingWake {
        observation: Arc::clone(&observation),
        registrations: Arc::downgrade(&reactor.registered_fds),
        cancellation: Mutex::new(None),
    });
    let old = reactor
        .register_owned_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&old_waker)),
            SocketLease::from(&socket),
        )
        .expect("old registration");
    *old_waker
        .cancellation
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = Some(old);
    drop(old_waker);
    let current_wake = Arc::new(WakeCount::default());
    let current = reactor
        .register_owned_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&current_wake)),
            SocketLease::from(&socket),
        )
        .expect("replacement registration");
    assert!(observation.dropped.load(Ordering::Relaxed));
    assert!(!observation.dropped_while_locked.load(Ordering::Relaxed));
    assert!(
        reactor
            .waiter_cancellations
            .has_interest(FdKey::from(fd), Interest::READABLE)
    );
    assert!(reactor.platform_reactor.has_registration(fd));
    sender
        .send_to(b"current", socket.local_addr().expect("socket address"))
        .expect("send current payload");
    reactor
        .run_iteration(Some(Duration::from_secs(1)))
        .expect("replacement readiness dispatch");
    assert_eq!(current_wake.0.load(Ordering::Relaxed), 1);
    let mut payload = [0_u8; 7];
    assert_eq!(socket.recv(&mut payload).expect("receive payload"), 7);
    assert_eq!(&payload, b"current");
    drop(current);
    assert!(
        !reactor
            .waiter_cancellations
            .has_interest(FdKey::from(fd), Interest::READABLE)
    );
}

#[test]
#[cfg(windows)]
fn raw_registration_replaces_owned_waiter_atomically() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("socket bind"));
    socket.set_nonblocking(true).expect("socket nonblocking");
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    let fd = socket.as_raw_socket() as crate::RawFd;
    let observation = Arc::new(DropObservation {
        dropped: AtomicBool::new(false),
        dropped_while_locked: AtomicBool::new(false),
    });
    let old_waker = Arc::new(DropObservingWake {
        observation: Arc::clone(&observation),
        registrations: Arc::downgrade(&reactor.registered_fds),
        cancellation: Mutex::new(None),
    });
    let old = reactor
        .register_owned_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&old_waker)),
            SocketLease::from(&socket),
        )
        .expect("owned registration");
    *old_waker
        .cancellation
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = Some(old);
    drop(old_waker);

    reactor
        .register_fd(fd, Interest::READABLE)
        .expect("raw replacement registration");
    assert!(observation.dropped.load(Ordering::Relaxed));
    assert!(!observation.dropped_while_locked.load(Ordering::Relaxed));
    assert!(reactor.platform_reactor.has_registration(fd));
    assert!(
        reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd))
    );
    assert!(
        !reactor
            .waiter_cancellations
            .has_interest(FdKey::from(fd), Interest::READABLE)
    );

    sender
        .send_to(b"raw", socket.local_addr().expect("socket address"))
        .expect("send raw payload");
    let event = reactor
        .platform_reactor
        .poll_registered_events(Some(Duration::from_secs(1)))
        .expect("raw readiness poll")
        .into_iter()
        .find(|event| event.descriptor() == fd)
        .expect("raw readiness event");
    assert!(event.event().readable);
    assert!(
        reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd))
    );
    let mut payload = [0_u8; 3];
    assert_eq!(socket.recv(&mut payload).expect("receive payload"), 3);
    assert_eq!(&payload, b"raw");
    reactor.unregister_fd(fd).expect("raw unregister");
}

#[test]
#[cfg(windows)]
fn unregister_destroys_owned_waiter_after_reactor_unlock() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("socket bind"));
    let fd = socket.as_raw_socket() as crate::RawFd;
    let observation = Arc::new(DropObservation {
        dropped: AtomicBool::new(false),
        dropped_while_locked: AtomicBool::new(false),
    });
    let old_waker = Arc::new(DropObservingWake {
        observation: Arc::clone(&observation),
        registrations: Arc::downgrade(&reactor.registered_fds),
        cancellation: Mutex::new(None),
    });
    let cancellation = reactor
        .register_owned_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&old_waker)),
            SocketLease::from(&socket),
        )
        .expect("owned registration");
    *old_waker
        .cancellation
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = Some(cancellation);
    drop(old_waker);

    reactor.unregister_fd(fd).expect("unregister owned waiter");
    assert!(observation.dropped.load(Ordering::Relaxed));
    assert!(!observation.dropped_while_locked.load(Ordering::Relaxed));
    assert!(!reactor.platform_reactor.has_registration(fd));
    assert!(
        !reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd))
    );
    assert!(
        !reactor
            .waiter_cancellations
            .has_interest(FdKey::from(fd), Interest::READABLE)
    );
}

#[test]
#[cfg(windows)]
fn cancellation_after_scoped_reactor_teardown_is_inert() {
    let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("socket bind"));
    let fd = socket.as_raw_socket() as crate::RawFd;
    let cancellation = {
        let reactor = IoReactor::new().expect("scoped reactor");
        reactor.with_active(|| {
            reactor
                .register_owned_waker(
                    fd,
                    Interest::READABLE,
                    Waker::noop().clone(),
                    SocketLease::from(&socket),
                )
                .expect("scoped registration")
        })
    };
    drop(cancellation);
    assert_eq!(
        socket.local_addr().expect("socket remains usable").ip(),
        "127.0.0.1"
            .parse::<std::net::IpAddr>()
            .expect("loopback address")
    );
}
