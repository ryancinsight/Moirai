//! `WsaPollReactor` readiness, wake, generation, and owned-socket-lease tests.

use std::io::Read;
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::os::windows::io::AsRawSocket;
use std::sync::Arc;
use std::time::Duration;

use crate::reactor::registration::{PlatformUpdateFailure, RegistrationTable};
use crate::reactor::socket_owner::SocketLease;
use crate::{Interest, RawFd, Reactor};

use super::types::WsaPollReactor;
use super::types::lock_mutex;

#[test]
fn wsapoll_reactor_reports_socket_readiness() {
    let reactor = WsaPollReactor::new().expect("reactor");
    let recv = UdpSocket::bind("127.0.0.1:0").expect("recv bind");
    recv.set_nonblocking(true).expect("nonblocking");
    let recv_addr = recv.local_addr().expect("recv addr");
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    let raw = recv.as_raw_socket() as RawFd;
    let raw_key = raw as usize;

    reactor
        .register_fd(raw, Interest::READABLE)
        .expect("register");

    // Before any datagram arrives the receiver is not readable. (register_fd
    // pinged the wake socket; poll_events drains that and reports no recv
    // readiness.)
    let events = reactor
        .poll_events(Some(Duration::from_millis(50)))
        .expect("poll");
    assert!(
        events.iter().all(|e| e.fd as usize != raw_key),
        "no readiness before data"
    );

    // After a datagram, WSAPoll reports the receiver readable.
    sender.send_to(b"x", recv_addr).expect("send");
    let events = reactor
        .poll_events(Some(Duration::from_millis(500)))
        .expect("poll");
    assert!(
        events
            .iter()
            .any(|e| e.fd as usize == raw_key && e.readable),
        "reactor must report the receiver readable, got {events:?}"
    );
}

#[test]
fn wsapoll_reactor_wake_interrupts_a_blocking_poll() {
    let reactor = Arc::new(WsaPollReactor::new().expect("reactor"));
    let polling = Arc::clone(&reactor);
    let start = Arc::new(std::sync::Barrier::new(2));
    let polling_start = Arc::clone(&start);
    let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
    let handle = std::thread::spawn(move || {
        // Would block up to 5s; wake() must interrupt it well before.
        // (Don't return the `Vec<Event>` — it is `!Send` on Windows.)
        polling_start.wait();
        let result = polling.poll_events(Some(Duration::from_secs(5))).map(drop);
        done_tx.send(result).expect("report poll completion");
    });
    start.wait();
    reactor.wake().expect("wake");
    done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("wake must complete the poll before the deadline")
        .expect("poll");
    handle.join().expect("poll thread");
}

#[test]
fn wsapoll_reactor_self_cleans_closed_sockets() {
    let reactor = WsaPollReactor::new().expect("reactor");
    let sock = UdpSocket::bind("127.0.0.1:0").expect("bind");
    let raw = sock.as_raw_socket() as RawFd;
    reactor
        .register_fd(raw, Interest::READABLE)
        .expect("register");
    drop(sock); // close without unregistering -> POLLNVAL

    // The poll surfaces POLLNVAL for the closed socket and removes it; the
    // call must not error or wedge.
    let _ = reactor
        .poll_events(Some(Duration::from_millis(50)))
        .expect("poll must not error on a stale socket");
    assert!(
        lock_mutex(&reactor.registrations).is_empty(),
        "closed socket must be self-cleaned from the interest set"
    );
}

#[test]
fn stale_snapshot_cannot_remove_or_match_reused_socket() {
    let socket = 41;
    let mut registrations = RegistrationTable::default();
    let first_generation = registrations.issue_generation().expect("first generation");
    registrations.commit(socket, Interest::READABLE, first_generation);
    let stale_generation = registrations
        .get(socket)
        .expect("first registration exists")
        .generation;

    let replacement_generation = registrations
        .issue_generation()
        .expect("replacement generation");
    registrations.commit(socket, Interest::WRITABLE, replacement_generation);
    let current_generation = registrations
        .get(socket)
        .expect("replacement registration exists")
        .generation;

    assert_ne!(stale_generation, current_generation);
    assert!(!registrations.is_current(socket, stale_generation));
    assert!(!registrations.remove_if_current(socket, stale_generation));
    assert!(registrations.is_current(socket, current_generation));
}

#[test]
fn owned_snapshot_defers_close_until_kernel_return() {
    let reactor = WsaPollReactor::new().expect("reactor");
    let mut socket = Some(Arc::new(
        UdpSocket::bind("127.0.0.1:0").expect("owned socket bind"),
    ));
    socket
        .as_ref()
        .expect("owned socket")
        .set_nonblocking(true)
        .expect("owned socket nonblocking");
    let weak = Arc::downgrade(socket.as_ref().expect("owned socket"));
    let owner = SocketLease::from(socket.as_ref().expect("owned socket"));
    let fd = owner.raw_socket() as RawFd;
    reactor
        .register_owned_waiter(fd, Interest::READABLE, owner.downgrade())
        .map_err(PlatformUpdateFailure::into_error)
        .expect("owned waiter registration");
    drop(owner);

    let events = reactor
        .poll_registered_events_after_snapshot(Some(Duration::ZERO), || {
            drop(socket.take());
            assert!(
                weak.upgrade().is_some(),
                "snapshot lease must retain the real socket during WSAPoll"
            );
        })
        .expect("owned snapshot poll");
    assert!(events.iter().all(|event| event.descriptor() != fd));
    assert!(
        weak.upgrade().is_none(),
        "last snapshot lease must release after WSAPoll"
    );

    let invalidated = reactor
        .poll_registered_events(Some(Duration::ZERO))
        .expect("expired owner cleanup");
    assert!(
        invalidated
            .iter()
            .any(|event| { event.descriptor() == fd && event.was_invalidated() })
    );
    assert!(!reactor.has_registration(fd));
}

#[test]
fn tcp_snapshot_defers_peer_eof_until_kernel_return() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener bind");
    let mut peer =
        TcpStream::connect(listener.local_addr().expect("listener address")).expect("peer connect");
    peer.set_nonblocking(true).expect("peer nonblocking");
    let (owned, _) = listener.accept().expect("owned accept");
    owned.set_nonblocking(true).expect("owned nonblocking");
    let mut owned = Some(Arc::new(owned));
    let weak = Arc::downgrade(owned.as_ref().expect("owned stream"));
    let owner = SocketLease::from(owned.as_ref().expect("owned stream"));
    let fd = owner.raw_socket() as RawFd;
    let reactor = WsaPollReactor::new().expect("reactor");
    reactor
        .register_owned_waiter(fd, Interest::READABLE, owner.downgrade())
        .map_err(PlatformUpdateFailure::into_error)
        .expect("owned stream registration");
    drop(owner);

    reactor
        .poll_registered_events_after_snapshot(Some(Duration::ZERO), || {
            drop(owned.take());
            assert!(weak.upgrade().is_some());
            let mut byte = [0_u8; 1];
            assert_eq!(
                peer.read(&mut byte)
                    .expect_err("snapshot lease must defer peer EOF")
                    .kind(),
                std::io::ErrorKind::WouldBlock
            );
        })
        .expect("owned stream snapshot");
    assert!(weak.upgrade().is_none());

    peer.set_nonblocking(false).expect("peer blocking mode");
    peer.set_read_timeout(Some(Duration::from_secs(2)))
        .expect("peer EOF deadline");
    let mut byte = [0_u8; 1];
    assert_eq!(
        peer.read(&mut byte)
            .expect("peer EOF after snapshot release"),
        0
    );
}

#[test]
fn retired_owner_does_not_hide_live_socket_readiness() {
    let reactor = WsaPollReactor::new().expect("reactor");
    let mut retired = Some(Arc::new(
        UdpSocket::bind("127.0.0.1:0").expect("retired socket bind"),
    ));
    let live = Arc::new(UdpSocket::bind("127.0.0.1:0").expect("live socket bind"));
    live.set_nonblocking(true).expect("live socket nonblocking");
    let retired_owner = SocketLease::from(retired.as_ref().expect("retired socket"));
    let retired_fd = retired_owner.raw_socket() as RawFd;
    let retired_weak = Arc::downgrade(retired.as_ref().expect("retired socket"));
    let live_owner = SocketLease::from(&live);
    let live_fd = live_owner.raw_socket() as RawFd;
    reactor
        .register_owned_waiter(retired_fd, Interest::READABLE, retired_owner.downgrade())
        .map_err(PlatformUpdateFailure::into_error)
        .expect("retired registration");
    reactor
        .register_owned_waiter(live_fd, Interest::READABLE, live_owner.downgrade())
        .map_err(PlatformUpdateFailure::into_error)
        .expect("live registration");
    drop(retired_owner);
    drop(live_owner);
    let sender = UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    assert_eq!(
        sender
            .send_to(b"ready", live.local_addr().expect("live address"))
            .expect("readiness datagram"),
        5
    );

    let events = reactor
        .poll_registered_events_after_snapshot(Some(Duration::from_secs(1)), || {
            drop(retired.take());
            assert!(retired_weak.upgrade().is_some());
        })
        .expect("mixed snapshot poll");
    assert!(
        events
            .iter()
            .any(|event| { event.descriptor() == live_fd && event.event().readable })
    );
    assert!(retired_weak.upgrade().is_none());
    let mut payload = [0_u8; 5];
    assert_eq!(live.recv(&mut payload).expect("live payload"), 5);
    assert_eq!(&payload, b"ready");
}

#[test]
fn snapshot_error_releases_owner_after_unlock() {
    let reactor = WsaPollReactor::new().expect("reactor");
    let mut socket = Some(Arc::new(
        UdpSocket::bind("127.0.0.1:0").expect("owned socket bind"),
    ));
    let weak = Arc::downgrade(socket.as_ref().expect("owned socket"));
    let owner = SocketLease::from(socket.as_ref().expect("owned socket"));
    let fd = owner.raw_socket() as RawFd;
    reactor
        .register_owned_waiter(fd, Interest::READABLE, owner.downgrade())
        .map_err(PlatformUpdateFailure::into_error)
        .expect("owned waiter registration");
    drop(owner);

    let result = reactor.poll_registered_events_after_snapshot_error(|| {
        drop(socket.take());
        assert!(weak.upgrade().is_some());
    });
    let Err(error) = result else {
        panic!("unsupported WSAPoll event flags must fail");
    };
    assert_eq!(error.raw_os_error(), Some(10022));
    assert!(
        weak.upgrade().is_none(),
        "error path releases the final snapshot owner"
    );
    assert!(reactor.poll_buffer.try_lock().is_ok());
}
