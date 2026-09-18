//! Terminal driver-failure publication: waiter wakeup, source preservation,
//! non-terminal outcomes, and registration/publication serialization.

use super::super::core::{FdKey, IoReactor};
use super::harness::{
    InjectedDriverFailure, LockObservingWake, ReentrantWake, WakeCount,
    assert_direct_injected_error, assert_injected_driver_source, key_to_raw, socket_to_raw,
};
use crate::Interest;
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

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
