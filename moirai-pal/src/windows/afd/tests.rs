use std::collections::HashSet;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::windows::io::{AsRawSocket, RawSocket};
use std::sync::{Barrier, Mutex, mpsc};
use std::time::Duration;

use windows::Win32::Foundation::{
    ERROR_INVALID_PARAMETER, NTSTATUS, STATUS_BUFFER_OVERFLOW, STATUS_CANCELLED,
    STATUS_INVALID_PARAMETER, STATUS_PENDING, STATUS_SUCCESS, STATUS_TIMEOUT,
};

use super::device::{finished, started};
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
fn cancel_racing_readiness_is_always_suppressed_and_frees_the_slot() {
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
        assert!(
            finished.is_empty(),
            "a cancelled poll reports no readiness whichever side won the race"
        );
        assert_eq!(port.armed(), 0);
        port.arm(socket_id(&server), Interest::READABLE)
            .expect("the slot must be reusable after the race");
    }
}

#[test]
fn cancel_after_the_completion_queued_still_suppresses_a_readable_poll() {
    let (server, mut client) = pair();
    client.write_all(b"x").expect("peer write");
    let port = AfdPort::new(1).expect("port");
    let token = port
        .arm(socket_id(&server), Interest::READABLE)
        .expect("arm an already-readable socket");
    port.cancel(token).expect("cancel");
    let (polls, finished) = dequeue(&port);
    assert_eq!(polls, 1);
    assert!(finished.is_empty(), "the cancelled poll must not report");
    assert_eq!(port.armed(), 0);
}

#[test]
fn cancel_after_the_completion_queued_still_suppresses_a_writable_poll() {
    let (_server, client) = pair();
    let port = AfdPort::new(1).expect("port");
    let token = port
        .arm(socket_id(&client), Interest::WRITABLE)
        .expect("arm a writable socket");
    port.cancel(token).expect("cancel");
    let (polls, finished) = dequeue(&port);
    assert_eq!(polls, 1);
    assert!(finished.is_empty(), "the cancelled poll must not report");
    assert_eq!(port.armed(), 0);
}

#[test]
fn a_second_poller_is_refused_instead_of_waiting_behind_the_first() {
    let port = AfdPort::new(1).expect("port");
    let (started, running) = mpsc::channel();
    std::thread::scope(|scope| {
        let long = scope.spawn(|| {
            started.send(()).expect("announce");
            // The main thread's zero-timeout polls may hold the dequeue slot
            // when this call arrives; retry until it owns the slot.
            loop {
                match port.poll(None, |_, _| {}) {
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::yield_now();
                    }
                    outcome => break outcome,
                }
            }
        });
        running.recv().expect("long poller started");
        // The long poller takes the dequeue slot some time after announcing;
        // every call before that returns 0 at once, every call after is refused.
        let refused = loop {
            match port.poll(Some(Duration::ZERO), |_, _| {}) {
                Ok(0) => std::thread::yield_now(),
                Ok(polls) => panic!("no poll was armed, dequeued {polls}"),
                Err(error) => break error,
            }
        };
        assert_eq!(refused.kind(), io::ErrorKind::WouldBlock);
        port.wake().expect("release the long poller");
        assert_eq!(long.join().expect("long poller").expect("long poll"), 0);
    });
}

#[test]
fn every_non_negative_start_status_leaves_the_request_with_the_kernel() {
    for status in [STATUS_SUCCESS, STATUS_PENDING, STATUS_TIMEOUT] {
        started(status).unwrap_or_else(|error| {
            panic!("{status:?} queues a packet, so the slot must stay armed: {error}")
        });
    }
}

#[test]
fn every_negative_start_status_is_an_error_without_a_packet() {
    for status in [
        STATUS_INVALID_PARAMETER,
        STATUS_CANCELLED,
        STATUS_BUFFER_OVERFLOW,
        NTSTATUS(i32::MIN),
    ] {
        started(status).expect_err("a failed start queues no packet and must release the slot");
    }
    let error = started(STATUS_INVALID_PARAMETER).expect_err("invalid parameter");
    assert_eq!(
        error.raw_os_error(),
        Some(ERROR_INVALID_PARAMETER.0.cast_signed())
    );
}

#[test]
fn a_failure_status_reaches_the_sink_as_its_win32_error() {
    let readiness = || Event {
        fd: std::ptr::null_mut(),
        readable: true,
        writable: false,
        error: false,
        hangup: false,
    };
    let reported = finished(STATUS_SUCCESS, readiness()).expect("success");
    assert!(reported.readable && !reported.writable && !reported.error);
    let error = finished(STATUS_INVALID_PARAMETER, readiness()).expect_err("failure");
    assert_eq!(
        error.raw_os_error(),
        Some(ERROR_INVALID_PARAMETER.0.cast_signed())
    );
}

#[test]
fn slots_beyond_the_first_device_group_report_through_their_own_device() {
    const COUNT: usize = 40;
    let pairs: Vec<_> = (0..COUNT).map(|_| pair()).collect();
    let port = AfdPort::new(COUNT).expect("port");
    let armed: Vec<(Token, RawSocket)> = pairs
        .iter()
        .map(|(server, _)| {
            let id = socket_id(server);
            (port.arm(id, Interest::WRITABLE).expect("arm"), id)
        })
        .collect();
    assert!(
        armed.iter().any(|(token, _)| token.index() >= 32),
        "the table must hand out slots past the first group of 32"
    );
    let mut seen = HashSet::new();
    while seen.len() < COUNT {
        let (_, finished) = dequeue(&port);
        for (token, event) in finished {
            let event = event.expect("a writable poll must not fail");
            let (_, id) = armed
                .iter()
                .find(|(armed_token, _)| *armed_token == token)
                .expect("a reported token was armed");
            assert_eq!(event.fd, *id as RawFd);
            assert!(event.writable && !event.readable && !event.error && !event.hangup);
            assert!(seen.insert(token), "a poll reports once");
        }
    }
    assert_eq!(port.armed(), 0);
}

#[test]
fn closing_the_socket_locally_reports_error_and_hangup_with_no_direction() {
    let (server, _client) = pair();
    let id = socket_id(&server);
    let port = AfdPort::new(1).expect("port");
    let token = port.arm(id, Interest::READABLE).expect("arm");
    drop(server);
    let (reported, event) = single_event(&port);
    assert_eq!(reported, token);
    assert_eq!(event.fd, id as RawFd);
    assert!(event.error && event.hangup && !event.readable && !event.writable);
    assert_eq!(port.armed(), 0);
}

#[test]
fn concurrent_pollers_deliver_each_completion_exactly_once() {
    const COUNT: usize = 16;
    const POLLERS: usize = 4;
    let mut pairs: Vec<_> = (0..COUNT).map(|_| pair()).collect();
    let port = AfdPort::new(COUNT).expect("port");
    let armed: HashSet<Token> = pairs
        .iter()
        .map(|(server, _)| {
            port.arm(socket_id(server), Interest::READABLE)
                .expect("arm")
        })
        .collect();
    let delivered = Mutex::new(Vec::new());
    let start = Barrier::new(POLLERS + 1);
    std::thread::scope(|scope| {
        for _ in 0..POLLERS {
            scope.spawn(|| {
                start.wait();
                // Zero-timeout polls only: a thread that keeps running on the
                // port while another is blocked in a dequeue starves it,
                // because the port admits one running thread at a time.
                while port.armed() > 0 {
                    match port.poll(Some(Duration::ZERO), |token, event| {
                        let event = event.expect("a readable poll must not fail");
                        assert!(event.readable);
                        delivered.lock().expect("sink lock").push(token);
                    }) {
                        Ok(_) => {}
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                        Err(error) => panic!("poll failed: {error}"),
                    }
                    std::thread::yield_now();
                }
            });
        }
        start.wait();
        for (_, client) in &mut pairs {
            client.write_all(b"x").expect("peer write");
        }
    });
    let delivered = delivered.into_inner().expect("sink lock");
    assert_eq!(delivered.len(), COUNT, "no completion is lost or repeated");
    assert_eq!(delivered.into_iter().collect::<HashSet<_>>(), armed);
}
