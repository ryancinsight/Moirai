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
use std::error::Error as _;
#[cfg(any(unix, windows))]
use std::fmt;
#[cfg(any(unix, windows))]
use std::net::{TcpListener, TcpStream, UdpSocket};
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
use std::task::{Context, Poll, Wake, Waker};
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

#[cfg(any(unix, windows))]
#[derive(Debug)]
struct InjectedDriverFailure(u32);

#[cfg(any(unix, windows))]
impl fmt::Display for InjectedDriverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "injected driver failure {}", self.0)
    }
}

#[cfg(any(unix, windows))]
impl std::error::Error for InjectedDriverFailure {}

#[cfg(any(unix, windows))]
struct ReentrantWake {
    count: AtomicUsize,
    reactor: Weak<IoReactor>,
    fd: usize,
    registration_error: Mutex<Option<std::io::Error>>,
}

#[cfg(any(unix, windows))]
impl Wake for ReentrantWake {
    fn wake(self: Arc<Self>) {
        self.record();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.record();
    }
}

#[cfg(any(unix, windows))]
impl ReentrantWake {
    fn record(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
        let reactor = self.reactor.upgrade().expect("reactor remains live");
        let error = reactor
            .register_waker(
                key_to_raw(self.fd),
                Interest::READABLE,
                Waker::noop().clone(),
            )
            .expect_err("terminal driver rejects reentrant registration");
        *self
            .registration_error
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(error);
    }
}

#[cfg(unix)]
fn socket_to_raw(socket: &UdpSocket) -> crate::RawFd {
    socket.as_raw_fd()
}

#[cfg(unix)]
fn key_to_raw(key: usize) -> crate::RawFd {
    i32::try_from(key).expect("Unix descriptor fits i32")
}

#[cfg(windows)]
fn socket_to_raw(socket: &UdpSocket) -> crate::RawFd {
    socket.as_raw_socket() as crate::RawFd
}

#[cfg(windows)]
fn key_to_raw(key: usize) -> crate::RawFd {
    key as crate::RawFd
}

#[cfg(any(unix, windows))]
fn assert_injected_driver_source(error: &std::io::Error, expected: u32) {
    assert_eq!(error.kind(), std::io::ErrorKind::ConnectionAborted);
    let mut source = error.source();
    while let Some(cause) = source {
        if let Some(retained) = cause.downcast_ref::<std::io::Error>()
            && let Some(injected) = retained
                .get_ref()
                .and_then(|inner| inner.downcast_ref::<InjectedDriverFailure>())
        {
            assert_eq!(injected.0, expected);
            return;
        }
        if let Some(injected) = cause.downcast_ref::<InjectedDriverFailure>() {
            assert_eq!(injected.0, expected);
            return;
        }
        source = cause.source();
    }
    panic!("retained driver error must preserve its typed source");
}

#[cfg(any(unix, windows))]
fn assert_direct_injected_error(error: &std::io::Error, expected: u32) {
    assert_eq!(error.kind(), std::io::ErrorKind::ConnectionAborted);
    let injected = error
        .get_ref()
        .and_then(|source| source.downcast_ref::<InjectedDriverFailure>())
        .expect("direct iteration error preserves its typed payload");
    assert_eq!(injected.0, expected);
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
fn terminal_driver_failure_wakes_waiters_and_preserves_its_source() {
    let reactor = Arc::new(IoReactor::new().expect("reactor"));
    let socket = UdpSocket::bind("127.0.0.1:0").expect("socket bind");
    socket.set_nonblocking(true).expect("socket nonblocking");
    let fd = socket_to_raw(&socket);
    let read_wake = Arc::new(ReentrantWake {
        count: AtomicUsize::new(0),
        reactor: Arc::downgrade(&reactor),
        fd: FdKey::from(fd).0,
        registration_error: Mutex::new(None),
    });
    let write_wake = Arc::new(LockObservingWake {
        count: AtomicUsize::new(0),
        woke_while_locked: AtomicBool::new(false),
        registrations: Arc::downgrade(&reactor.registered_fds),
    });

    reactor
        .register_waker(fd, Interest::READABLE, Waker::from(Arc::clone(&read_wake)))
        .expect("register read waiter");
    reactor
        .register_waker(fd, Interest::WRITABLE, Waker::from(Arc::clone(&write_wake)))
        .expect("register write waiter");

    let published = reactor.inject_driver_failure(std::io::Error::new(
        std::io::ErrorKind::ConnectionAborted,
        InjectedDriverFailure(17),
    ));
    assert_injected_driver_source(&published, 17);
    assert_eq!(read_wake.count.load(Ordering::Relaxed), 1);
    assert_eq!(write_wake.count.load(Ordering::Relaxed), 1);
    assert!(!write_wake.woke_while_locked.load(Ordering::Relaxed));
    let reentrant_error = read_wake
        .registration_error
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .take()
        .expect("reentrant registration reports terminal failure");
    assert_injected_driver_source(&reentrant_error, 17);
    assert!(
        reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_empty()
    );
    #[cfg(windows)]
    assert!(!reactor.has_platform_generation(fd));

    let later_socket = UdpSocket::bind("127.0.0.1:0").expect("later socket bind");
    let later_fd = socket_to_raw(&later_socket);
    let waiter_error = reactor
        .register_waker(later_fd, Interest::READABLE, Waker::noop().clone())
        .expect_err("later waiter registration must fail");
    assert_injected_driver_source(&waiter_error, 17);
    let descriptor_error = reactor
        .register_fd(later_fd, Interest::READABLE)
        .expect_err("later descriptor registration must fail");
    assert_injected_driver_source(&descriptor_error, 17);

    let repeated = reactor.inject_driver_failure(std::io::Error::new(
        std::io::ErrorKind::BrokenPipe,
        InjectedDriverFailure(99),
    ));
    assert_injected_driver_source(&repeated, 17);
}

#[test]
#[cfg(any(unix, windows))]
fn driven_failure_wakes_pending_tcp_read_and_repolls_terminal_error() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener bind");
    let address = listener.local_addr().expect("listener address");
    let client = TcpStream::connect(address).expect("client connect");
    let (server, _) = listener.accept().expect("server accept");
    let mut stream = crate::net::AsyncTcpStream::from_std(server).expect("async server stream");
    let reactor = IoReactor::new().expect("reactor");
    let wake_count = Arc::new(WakeCount::default());
    let waker = Waker::from(Arc::clone(&wake_count));
    let mut context = Context::from_waker(&waker);
    let mut payload = [0_u8; 1];

    reactor.with_active(|| {
        assert!(matches!(
            stream.poll_read(&mut context, &mut payload),
            Poll::Pending
        ));
    });
    reactor.inject_iteration_failure(std::io::Error::new(
        std::io::ErrorKind::ConnectionAborted,
        InjectedDriverFailure(31),
    ));
    let run_error = reactor
        .run()
        .expect_err("driven iteration failure must terminate run");
    assert_injected_driver_source(&run_error, 31);
    assert_eq!(wake_count.0.load(Ordering::Relaxed), 1);

    reactor.with_active(|| {
        let Poll::Ready(Err(read_error)) = stream.poll_read(&mut context, &mut payload) else {
            panic!("pending TCP read must observe retained driver failure");
        };
        assert_injected_driver_source(&read_error, 31);
    });
    drop(client);
}

#[test]
#[cfg(any(unix, windows))]
fn nonterminal_reactor_outcomes_do_not_poison_registration() {
    let manual = IoReactor::new().expect("manual reactor");
    manual.inject_iteration_failure(std::io::Error::new(
        std::io::ErrorKind::ConnectionAborted,
        InjectedDriverFailure(37),
    ));
    let direct_error = manual
        .run_iteration(Some(Duration::ZERO))
        .expect_err("manual iteration receives injected error");
    assert_direct_injected_error(&direct_error, 37);
    assert!(!manual.running.load(Ordering::Relaxed));
    let manual_socket = UdpSocket::bind("127.0.0.1:0").expect("manual socket bind");
    manual
        .register_waker(
            socket_to_raw(&manual_socket),
            Interest::READABLE,
            Waker::noop().clone(),
        )
        .expect("manual iteration error remains caller-owned");

    let stopped = IoReactor::new().expect("stopped reactor");
    stopped.stop().expect("normal stop");
    assert!(!stopped.running.load(Ordering::Relaxed));
    let stopped_socket = UdpSocket::bind("127.0.0.1:0").expect("stopped socket bind");
    stopped
        .register_waker(
            socket_to_raw(&stopped_socket),
            Interest::READABLE,
            Waker::noop().clone(),
        )
        .expect("normal stop does not publish failure");

    let duplicate = IoReactor::new().expect("duplicate reactor");
    duplicate
        .metrics
        .start_time
        .set(std::time::Instant::now())
        .expect("seed prior run");
    assert_eq!(
        duplicate.run().expect_err("duplicate run must fail").kind(),
        std::io::ErrorKind::Other
    );
    assert!(!duplicate.running.load(Ordering::Relaxed));
    let duplicate_socket = UdpSocket::bind("127.0.0.1:0").expect("duplicate socket bind");
    duplicate
        .register_waker(
            socket_to_raw(&duplicate_socket),
            Interest::READABLE,
            Waker::noop().clone(),
        )
        .expect("duplicate run error does not publish platform failure");
}

#[test]
#[cfg(any(unix, windows))]
fn terminal_publication_serializes_with_waiter_registration() {
    let reactor = Arc::new(IoReactor::new().expect("reactor"));
    let socket = UdpSocket::bind("127.0.0.1:0").expect("socket bind");
    socket.set_nonblocking(true).expect("socket nonblocking");
    let fd = FdKey::from(socket_to_raw(&socket)).0;
    let wake_count = Arc::new(WakeCount::default());
    let central_gate = reactor
        .registered_fds
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(2);
    let (registration_sender, registration_receiver) = std::sync::mpsc::sync_channel(1);
    let (publication_sender, publication_receiver) = std::sync::mpsc::sync_channel(1);

    let registering_reactor = Arc::clone(&reactor);
    let registering_wake = Arc::clone(&wake_count);
    let registering_started = started_sender.clone();
    let registrar = std::thread::spawn(move || {
        registering_started
            .send(())
            .expect("announce registration contender");
        let result = registering_reactor.register_waker(
            key_to_raw(fd),
            Interest::READABLE,
            Waker::from(registering_wake),
        );
        registration_sender
            .send(result)
            .expect("publish registration result");
    });

    let publishing_reactor = Arc::clone(&reactor);
    let publisher = std::thread::spawn(move || {
        started_sender
            .send(())
            .expect("announce publication contender");
        let result = publishing_reactor.inject_driver_failure(std::io::Error::new(
            std::io::ErrorKind::ConnectionAborted,
            InjectedDriverFailure(23),
        ));
        publication_sender
            .send(result)
            .expect("publish terminal result");
    });

    for _ in 0..2 {
        started_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("both contenders must reach the central gate");
    }
    drop(central_gate);

    let registration = registration_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("registration contender must finish");
    let publication = publication_receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("publication contender must finish");
    assert_injected_driver_source(&publication, 23);
    registrar.join().expect("registration thread");
    publisher.join().expect("publication thread");

    match registration {
        Ok(()) => assert_eq!(wake_count.0.load(Ordering::Relaxed), 1),
        Err(error) => {
            assert_injected_driver_source(&error, 23);
            assert_eq!(wake_count.0.load(Ordering::Relaxed), 0);
        }
    }
    assert!(
        reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_empty()
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
