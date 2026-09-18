//! Windows platform-generation bookkeeping: socket close retirement, reused
//! raw-socket generation ordering, and the closed-after-snapshot race.

use super::super::core::{FdKey, IoReactor};
use super::harness::{WakeCount, bind_reusing_socket, socket_to_raw};
use crate::Interest;
use std::net::UdpSocket;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::Waker;
use std::time::Duration;

#[test]
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
    let write_waker = Waker::from(Arc::clone(&current_wake_count));
    reactor
        .register_waker(fd, Interest::WRITABLE, write_waker.clone())
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
        let retained = current.write_waker.as_ref().expect("write waker retained");
        assert!(retained.will_wake(&write_waker));
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
