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

#[cfg(windows)]
fn bind_reusing_socket(fd: crate::RawFd) -> UdpSocket {
    const REUSE_ATTEMPTS: usize = 256;
    let mut held = Vec::with_capacity(REUSE_ATTEMPTS);
    for _ in 0..REUSE_ATTEMPTS {
        let socket = UdpSocket::bind("127.0.0.1:0").expect("replacement socket bind");
        if socket_to_raw(&socket) == fd {
            return socket;
        }
        held.push(socket);
    }
    panic!("Winsock did not reuse the retired socket value within {REUSE_ATTEMPTS} allocations");
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
        let payload = result.expect_err("inner closure must have panicked");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));

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
#[cfg(windows)]
fn closed_socket_retires_central_and_platform_waiters() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = UdpSocket::bind("127.0.0.1:0").expect("socket bind");
    socket.set_nonblocking(true).expect("socket nonblocking");
    let fd = socket_to_raw(&socket);
    let wake_count = Arc::new(WakeCount::default());

    reactor
        .register_waker(fd, Interest::READABLE, Waker::from(Arc::clone(&wake_count)))
        .expect("register read interest");
    drop(socket);

    reactor
        .run_iteration(Some(Duration::from_millis(50)))
        .expect("retire closed socket");

    assert_eq!(
        wake_count.0.load(Ordering::Relaxed),
        1,
        "closed socket must wake its waiter to observe the socket failure"
    );
    assert!(
        !reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd)),
        "closed socket must leave no central registration"
    );
    assert!(
        !reactor.platform_reactor.has_registration(fd),
        "closed socket must leave no platform registration"
    );
    assert!(
        !reactor.has_platform_generation(fd),
        "closed socket must leave no central platform generation"
    );
}

#[test]
#[cfg(windows)]
fn successive_reused_socket_invalidations_preserve_generation_order() {
    let reactor = IoReactor::new().expect("reactor");
    let retired_socket = UdpSocket::bind("127.0.0.1:0").expect("retired socket bind");
    retired_socket
        .set_nonblocking(true)
        .expect("retired socket nonblocking");
    let fd = socket_to_raw(&retired_socket);
    let retired_wake_count = Arc::new(WakeCount::default());
    let current_wake_count = Arc::new(WakeCount::default());
    reactor
        .register_waker(
            fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&retired_wake_count)),
        )
        .expect("register retired read interest");
    drop(retired_socket);

    let retired_event = reactor
        .platform_reactor
        .poll_registered_events(Some(Duration::from_millis(50)))
        .expect("poll retired socket")
        .into_iter()
        .find(|event| FdKey::from(event.descriptor()) == FdKey::from(fd))
        .expect("closed socket invalidation");
    assert!(retired_event.was_invalidated());
    assert!(!reactor.platform_reactor.has_registration(fd));
    assert_eq!(retired_wake_count.0.load(Ordering::Relaxed), 0);

    let current_socket = bind_reusing_socket(fd);
    current_socket
        .set_nonblocking(true)
        .expect("current socket nonblocking");
    reactor
        .register_waker(
            fd,
            Interest::WRITABLE,
            Waker::from(Arc::clone(&current_wake_count)),
        )
        .expect("register current write interest");
    assert_eq!(
        retired_wake_count.0.load(Ordering::Relaxed),
        1,
        "replacing an invalidated registration must wake its retired waiter"
    );
    drop(current_socket);
    let current_event = reactor
        .platform_reactor
        .poll_registered_events(Some(Duration::from_millis(50)))
        .expect("poll current socket")
        .into_iter()
        .find(|event| FdKey::from(event.descriptor()) == FdKey::from(fd))
        .expect("current socket invalidation");
    assert!(current_event.was_invalidated());

    reactor
        .handle_polled_event(retired_event)
        .expect("discard retired generation");

    assert_eq!(retired_wake_count.0.load(Ordering::Relaxed), 1);
    assert_eq!(current_wake_count.0.load(Ordering::Relaxed), 0);
    {
        let fds = reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let current = fds
            .get(&FdKey::from(fd))
            .expect("current registration remains");
        assert!(!current.interest.readable);
        assert!(current.interest.writable);
        assert!(current.read_waker.is_none());
        assert!(current.write_waker.is_some());
    }
    assert!(!reactor.platform_reactor.has_registration(fd));
    reactor
        .handle_polled_event(current_event)
        .expect("retire current generation");
    assert_eq!(retired_wake_count.0.load(Ordering::Relaxed), 1);
    assert_eq!(current_wake_count.0.load(Ordering::Relaxed), 1);
    assert!(
        !reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&FdKey::from(fd))
    );
    assert!(!reactor.has_platform_generation(fd));
}

#[test]
#[cfg(windows)]
fn socket_closed_after_poll_snapshot_preserves_subsequent_readiness() {
    let reactor = Arc::new(IoReactor::new().expect("reactor"));
    let retired = UdpSocket::bind("127.0.0.1:0").expect("retired socket bind");
    retired
        .set_nonblocking(true)
        .expect("retired socket nonblocking");
    let retired_fd = socket_to_raw(&retired);
    reactor
        .register_waker(
            retired_fd,
            Interest::READABLE,
            Waker::from(Arc::new(WakeCount::default())),
        )
        .expect("register retired socket");
    reactor
        .run_iteration(Some(Duration::ZERO))
        .expect("drain retired registration wake");

    let (snapshot_sender, snapshot_receiver) = std::sync::mpsc::sync_channel(1);
    let (continue_sender, continue_receiver) = std::sync::mpsc::sync_channel(1);
    let (first_poll_sender, first_poll_receiver) = std::sync::mpsc::sync_channel(1);
    let (driver_sender, driver_receiver) = std::sync::mpsc::sync_channel(1);
    let driver_reactor = Arc::clone(&reactor);
    let driver = std::thread::spawn(move || {
        let result = (|| -> std::io::Result<()> {
            let first_poll = driver_reactor
                .platform_reactor
                .poll_registered_events_after_snapshot(Some(Duration::from_secs(2)), || {
                    snapshot_sender.send(()).expect("publish poll snapshot");
                    continue_receiver.recv().expect("release snapshotted poll");
                });
            first_poll_sender
                .send(
                    first_poll
                        .as_ref()
                        .map(Vec::len)
                        .map_err(|error| (error.kind(), error.raw_os_error(), error.to_string())),
                )
                .expect("publish first poll result");
            let events = first_poll?;
            for event in events {
                driver_reactor.handle_polled_event(event)?;
            }
            driver_reactor.run_iteration(Some(Duration::from_secs(2)))
        })();
        driver_sender.send(result).expect("publish driver result");
    });

    snapshot_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("poll snapshot must complete");
    drop(retired);

    let current = UdpSocket::bind("127.0.0.1:0").expect("current socket bind");
    current
        .set_nonblocking(true)
        .expect("current socket nonblocking");
    let current_fd = socket_to_raw(&current);
    let current_address = current.local_addr().expect("current socket address");
    let current_wake_count = Arc::new(WakeCount::default());
    reactor
        .register_waker(
            current_fd,
            Interest::READABLE,
            Waker::from(Arc::clone(&current_wake_count)),
        )
        .expect("register current socket");
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    assert_eq!(
        sender
            .send_to(b"ready", current_address)
            .expect("send current readiness"),
        5
    );
    continue_sender.send(()).expect("release poll driver");

    first_poll_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("first poll must report its result")
        .expect("snapshotted close must not fail WSAPoll");
    driver_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("driver must report its result")
        .expect("snapshotted close must not terminate readiness dispatch");
    driver.join().expect("driver thread");
    assert_eq!(current_wake_count.0.load(Ordering::Relaxed), 1);
    let mut payload = [0_u8; 5];
    assert_eq!(
        current
            .recv(&mut payload)
            .expect("receive readiness payload"),
        5
    );
    assert_eq!(&payload, b"ready");
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
