use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::windows::io::{AsRawSocket, RawSocket};
use std::sync::{Barrier, mpsc};
use std::time::Duration;

use super::{AfdPort, Token};
use crate::{Event, Interest, RawFd};

/// Upper bound on a blocking dequeue. It is a failure backstop: every wait in
/// these tests returns as soon as its completion exists.
const WAIT: Duration = Duration::from_secs(5);

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener bind");
    let client = TcpStream::connect(listener.local_addr().expect("listener address"))
        .expect("client connect");
    let (server, _) = listener.accept().expect("server accept");
    (server, client)
}

fn socket_id(socket: &impl AsRawSocket) -> RawSocket {
    socket.as_raw_socket()
}

/// Dequeue one batch and return the dequeued count with the finished polls.
fn dequeue(port: &AfdPort) -> (usize, Vec<(Token, io::Result<Event>)>) {
    let mut finished = Vec::new();
    let polls = port
        .poll(Some(WAIT), |token, event| finished.push((token, event)))
        .expect("dequeue must succeed");
    (polls, finished)
}

fn single_event(port: &AfdPort) -> (Token, Event) {
    let (polls, mut finished) = dequeue(port);
    assert_eq!(polls, 1, "exactly one poll must complete");
    let (token, event) = finished.pop().expect("the poll must report readiness");
    (token, event.expect("the poll must not fail"))
}

#[test]
fn peer_write_reports_exactly_readable() {
    let (server, mut client) = pair();
    let port = AfdPort::new(4).expect("port");
    let token = port
        .arm(socket_id(&server), Interest::READABLE)
        .expect("arm");
    assert_eq!(port.armed(), 1);
    client.write_all(b"x").expect("peer write");
    let (finished, event) = single_event(&port);
    assert_eq!(finished, token);
    assert_eq!(event.fd, socket_id(&server) as RawFd);
    assert!(event.readable && !event.writable && !event.error && !event.hangup);
    assert_eq!(port.armed(), 0);
}

#[test]
fn fresh_connection_reports_exactly_writable() {
    let (_server, client) = pair();
    let port = AfdPort::new(4).expect("port");
    port.arm(socket_id(&client), Interest::WRITABLE)
        .expect("arm");
    let (_, event) = single_event(&port);
    assert!(event.writable && !event.readable && !event.error && !event.hangup);
}

#[test]
fn peer_close_reports_readable_with_hangup() {
    let (server, client) = pair();
    let port = AfdPort::new(4).expect("port");
    port.arm(socket_id(&server), Interest::READABLE)
        .expect("arm");
    drop(client);
    let (_, event) = single_event(&port);
    assert!(event.readable && event.hangup && !event.error);
}

#[test]
fn cancel_suppresses_readiness_and_frees_the_slot() {
    let (server, _client) = pair();
    let port = AfdPort::new(1).expect("port");
    let token = port
        .arm(socket_id(&server), Interest::READABLE)
        .expect("arm");
    port.cancel(token).expect("cancel");
    let (polls, finished) = dequeue(&port);
    assert_eq!(polls, 1);
    assert!(finished.is_empty(), "a cancelled poll reports nothing");
    assert_eq!(port.armed(), 0);
    port.arm(socket_id(&server), Interest::READABLE)
        .expect("the slot must be reusable");
}

#[test]
fn stale_token_cannot_cancel_a_later_poll_of_the_reused_slot() {
    let (first, mut first_peer) = pair();
    let (second, mut second_peer) = pair();
    let port = AfdPort::new(1).expect("port");
    let stale = port
        .arm(socket_id(&first), Interest::READABLE)
        .expect("first arm");
    first_peer.write_all(b"a").expect("first write");
    let (finished, _) = single_event(&port);
    assert_eq!(finished, stale);

    let live = port
        .arm(socket_id(&second), Interest::READABLE)
        .expect("second arm");
    assert_ne!(live, stale, "a reused slot issues a new generation");
    port.cancel(stale).expect("a stale cancel is a no-op");
    second_peer.write_all(b"b").expect("second write");
    let (finished, event) = single_event(&port);
    assert_eq!(finished, live);
    assert_eq!(event.fd, socket_id(&second) as RawFd);
}

#[test]
fn exhausted_table_is_a_typed_error() {
    let (first, _first_peer) = pair();
    let (second, _second_peer) = pair();
    let port = AfdPort::new(1).expect("port");
    port.arm(socket_id(&first), Interest::READABLE)
        .expect("first arm");
    let error = port
        .arm(socket_id(&second), Interest::READABLE)
        .expect_err("a full table must refuse");
    assert_eq!(error.kind(), io::ErrorKind::QuotaExceeded);
}

#[test]
fn invalid_requests_are_rejected() {
    assert_eq!(
        AfdPort::new(0).err().map(|error| error.kind()),
        Some(io::ErrorKind::InvalidInput)
    );
    let (server, _client) = pair();
    let port = AfdPort::new(1).expect("port");
    let none = Interest {
        readable: false,
        writable: false,
        error: true,
    };
    let error = port
        .arm(socket_id(&server), none)
        .expect_err("an interest with no direction must be refused");
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(port.armed(), 0, "a refused arm leaves no slot armed");
}

#[test]
fn wake_returns_a_poller_whether_posted_before_or_after_it_blocks() {
    let port = AfdPort::new(1).expect("port");
    port.wake().expect("wake before poll");
    assert_eq!(port.poll(None, |_, _| {}).expect("poll after wake"), 0);

    let (blocking, blocked) = mpsc::channel();
    std::thread::scope(|scope| {
        let poller = scope.spawn(|| {
            blocking.send(()).expect("announce");
            port.poll(None, |_, _| {}).expect("blocked poll")
        });
        blocked.recv().expect("poller started");
        port.wake().expect("wake during poll");
        assert_eq!(poller.join().expect("poller"), 0);
    });
}

#[test]
fn dropping_the_port_with_armed_polls_returns_and_leaves_sockets_usable() {
    let (mut server, mut client) = pair();
    let (other, _other_peer) = pair();
    let port = AfdPort::new(8).expect("port");
    port.arm(socket_id(&server), Interest::READABLE)
        .expect("arm server");
    port.arm(socket_id(&other), Interest::READ_WRITE)
        .expect("arm other");
    drop(port);

    client.write_all(b"ping").expect("client write");
    let mut received = [0_u8; 4];
    server.read_exact(&mut received).expect("server read");
    assert_eq!(&received, b"ping");
}

#[test]
fn cancel_racing_readiness_resolves_to_one_outcome_and_frees_the_slot() {
    for _ in 0..64 {
        let (server, mut client) = pair();
        let port = AfdPort::new(1).expect("port");
        let token = port
            .arm(socket_id(&server), Interest::READABLE)
            .expect("arm");
        let start = Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                start.wait();
                client.write_all(b"x").expect("racing write");
            });
            start.wait();
            port.cancel(token).expect("racing cancel");
        });
        let (polls, finished) = dequeue(&port);
        assert_eq!(polls, 1);
        match finished.as_slice() {
            [] => {}
            [(delivered, event)] => {
                assert_eq!(*delivered, token);
                assert!(event.as_ref().expect("readiness").readable);
            }
            other => panic!("one poll produced {} reports", other.len()),
        }
        assert_eq!(port.armed(), 0);
        port.arm(socket_id(&server), Interest::READABLE)
            .expect("the slot must be reusable after the race");
    }
}
