//! Backend registration-update failures: retained-interest preservation and
//! absent-registration removal, both waking every stranded waiter unlocked.

use super::super::core::{FdKey, IoReactor};
use super::super::registration::PlatformUpdateFailure;
use super::harness::{LockObservingWake, WakeCount, socket_to_raw};
use crate::{Event, Interest};
use std::net::UdpSocket;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;

#[test]
#[cfg(any(unix, windows))]
fn backend_update_failure_preserves_retained_registration_and_wakes_unlocked() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = UdpSocket::bind("127.0.0.1:0").expect("socket bind");
    socket.set_nonblocking(true).expect("socket nonblocking");
    let fd = socket_to_raw(&socket);
    let read_wake = Arc::new(LockObservingWake {
        count: AtomicUsize::new(0),
        woke_while_locked: AtomicBool::new(false),
        registrations: Arc::downgrade(&reactor.registered_fds),
    });
    let write_wake = Arc::new(LockObservingWake {
        count: AtomicUsize::new(0),
        woke_while_locked: AtomicBool::new(false),
        registrations: Arc::downgrade(&reactor.registered_fds),
    });

    reactor
        .register_waker(fd, Interest::READABLE, Waker::from(Arc::clone(&read_wake)))
        .expect("register read interest");
    reactor
        .register_waker(fd, Interest::WRITABLE, Waker::from(Arc::clone(&write_wake)))
        .expect("register write interest");

    let platform_interest = Mutex::new(Some(Interest::READ_WRITE));
    let result = reactor.wake_fd_waiters_with_platform(
        Event {
            fd,
            readable: true,
            writable: false,
            error: false,
            hangup: false,
        },
        |_| true,
        |_, _, _| {
            let armed = *platform_interest
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            Err(PlatformUpdateFailure::new(
                std::io::Error::other("injected update failure"),
                armed,
            ))
        },
    );

    assert_eq!(
        result
            .expect_err("injected update failure must propagate")
            .kind(),
        std::io::ErrorKind::Other
    );
    assert_eq!(read_wake.count.load(Ordering::Relaxed), 1);
    assert_eq!(write_wake.count.load(Ordering::Relaxed), 1);
    assert!(!read_wake.woke_while_locked.load(Ordering::Relaxed));
    assert!(!write_wake.woke_while_locked.load(Ordering::Relaxed));
    assert!(
        platform_interest
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_some()
    );
    let fds = reactor
        .registered_fds
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let central = fds
        .get(&FdKey::from(fd))
        .expect("retained platform registration remains central");
    assert!(central.interest.readable);
    assert!(central.interest.writable);
    assert!(central.read_waker.is_none());
    assert!(central.write_waker.is_none());
}

#[test]
#[cfg(any(unix, windows))]
fn backend_update_failure_removes_absent_registration_and_wakes_waiters() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = UdpSocket::bind("127.0.0.1:0").expect("socket bind");
    socket.set_nonblocking(true).expect("socket nonblocking");
    let fd = socket_to_raw(&socket);
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

    let platform_interest = Mutex::new(Some(Interest::READ_WRITE));
    let result = reactor.wake_fd_waiters_with_platform(
        Event {
            fd,
            readable: true,
            writable: false,
            error: false,
            hangup: false,
        },
        |_| true,
        |_, _, _| {
            *platform_interest
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = None;
            Err(PlatformUpdateFailure::new(
                std::io::Error::other("injected replacement failure"),
                None,
            ))
        },
    );

    assert_eq!(
        result
            .expect_err("injected replacement failure must propagate")
            .kind(),
        std::io::ErrorKind::Other
    );
    assert_eq!(read_count.0.load(Ordering::Relaxed), 1);
    assert_eq!(write_count.0.load(Ordering::Relaxed), 1);
    assert!(
        platform_interest
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_none()
    );
    assert!(
        !reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd))
    );
}
