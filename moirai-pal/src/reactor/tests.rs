#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::core::IoReactor;
#[cfg(any(unix, windows))]
use super::core::{FdInfo, FdKey};
use super::registration::PlatformUpdateFailure;
#[cfg(any(unix, windows))]
use crate::{Event, Interest, Reactor};
#[cfg(any(unix, windows))]
use std::collections::HashMap;
#[cfg(any(unix, windows))]
use std::net::UdpSocket;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::sync::atomic::Ordering;
#[cfg(any(unix, windows))]
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(any(unix, windows))]
use std::sync::{Arc, Mutex, Weak};
#[cfg(any(unix, windows))]
use std::task::{Wake, Waker};
#[cfg(any(unix, windows))]
use std::time::Duration;

#[cfg(any(unix, windows))]
#[derive(Default)]
struct WakeCount(AtomicUsize);

#[cfg(any(unix, windows))]
impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(any(unix, windows))]
struct LockObservingWake {
    count: AtomicUsize,
    woke_while_locked: AtomicBool,
    registrations: Weak<Mutex<HashMap<FdKey, FdInfo>>>,
}

#[cfg(any(unix, windows))]
impl Wake for LockObservingWake {
    fn wake(self: Arc<Self>) {
        self.record();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.record();
    }
}

#[cfg(any(unix, windows))]
impl LockObservingWake {
    fn record(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
        if self
            .registrations
            .upgrade()
            .is_some_and(|registrations| registrations.try_lock().is_err())
        {
            self.woke_while_locked.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(unix)]
fn socket_to_raw(socket: &UdpSocket) -> crate::RawFd {
    socket.as_raw_fd()
}

#[cfg(windows)]
fn socket_to_raw(socket: &UdpSocket) -> crate::RawFd {
    socket.as_raw_socket() as crate::RawFd
}

#[test]
fn test_reactor_creation() {
    let reactor = IoReactor::new();
    assert!(reactor.is_ok());
}

#[test]
fn test_reactor_metrics() {
    let reactor = IoReactor::new().unwrap();
    let metrics = reactor.metrics();
    assert_eq!(metrics.events_processed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.peak_fd_count.load(Ordering::Relaxed), 0);
}

#[test]
fn with_active_restores_thread_local_on_panic() {
    // Regression: if `f` panics, `with_active` must still restore the previous
    // thread-local reactor (via RAII), not leave a dangling pointer to the inner
    // reactor that a later `get_active()` would dereference (use-after-free).
    let outer = IoReactor::new().expect("outer reactor");
    let inner = IoReactor::new().expect("inner reactor");

    outer.with_active(|| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            inner.with_active(|| panic!("boom"));
        }));
        assert!(result.is_err(), "inner closure must have panicked");

        // The active reactor must be restored to `outer`, never left as `inner`.
        let active = IoReactor::get_active().expect("outer is still active");
        assert!(
            std::ptr::eq(active, &outer),
            "thread-local must be restored to the outer reactor after panic"
        );
    });
}

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
    assert!(!reactor
        .registered_fds
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .contains_key(&FdKey::from(fd)));

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
}

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
    assert!(platform_interest
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .is_some());
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
    assert!(platform_interest
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .is_none());
    assert!(!reactor
        .registered_fds
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .contains_key(&FdKey::from(fd)));
}
