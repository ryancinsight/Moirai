use super::*;
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;

#[derive(Default)]
struct RelayState {
    stop: AtomicBool,
    impair: AtomicBool,
    ordinal: AtomicUsize,
    dropped: AtomicUsize,
    reordered: AtomicUsize,
}

struct Relay {
    address: SocketAddr,
    state: Arc<RelayState>,
    task: Option<JoinHandle<io::Result<()>>>,
}

impl Relay {
    fn new(server: SocketAddr, fault_events: Option<usize>) -> Self {
        let socket = UdpSocket::bind("127.0.0.1:0").expect("real UDP relay");
        socket
            .set_read_timeout(Some(Duration::from_millis(10)))
            .expect("bounded relay shutdown polling");
        let address = socket.local_addr().expect("relay address");
        let state = Arc::new(RelayState::default());
        let running = Arc::clone(&state);
        let task = std::thread::spawn(move || {
            let mut client = None;
            let mut delayed = None;
            // Every UDP datagram fits a u16 length field. No packet is truncated.
            let mut bytes = vec![0; usize::from(u16::MAX)];
            while !running.stop.load(Ordering::Acquire) {
                let (length, source) = match socket.recv_from(&mut bytes) {
                    Ok(received) => received,
                    Err(error)
                        if matches!(
                            error.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                        ) =>
                    {
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                let target = if source == server {
                    client.ok_or_else(|| {
                        io::Error::other("server datagram precedes client routing")
                    })?
                } else {
                    client = Some(source);
                    server
                };
                // The declared model impairs short-header application traffic:
                // drop ordinal 1 modulo 4; hold ordinal 2 until ordinal 3 passes.
                // Handshake authentication completes before impairment starts.
                if length > 0 && bytes[0] & 0x80 == 0 && running.impair.load(Ordering::Acquire) {
                    let ordinal = running.ordinal.fetch_add(1, Ordering::Relaxed);
                    match if fault_events.is_none_or(|limit| ordinal < limit) {
                        ordinal % 4
                    } else {
                        0
                    } {
                        1 => {
                            running.dropped.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        2 => {
                            delayed = Some((bytes[..length].to_vec(), target));
                            continue;
                        }
                        _ => {}
                    }
                }
                socket.send_to(&bytes[..length], target)?;
                if let Some((bytes, target)) = delayed.take() {
                    socket.send_to(&bytes, target)?;
                    running.reordered.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        });
        Self {
            address,
            state,
            task: Some(task),
        }
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Release);
        if let Some(task) = self.task.take() {
            task.join()
                .expect("relay thread remains joined")
                .expect("relay I/O");
        }
    }
}

#[test]
fn finite_loss_and_reordering_preserve_the_complete_application_bytes() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let executor = executor();
    let (server, client) = endpoints(&executor, &config(65_536, 2, DEADLINE));
    // Two complete four-phase fault cycles exercise both directions; subsequent
    // packets are delivered. A finite loss tape admits eventual recovery.
    let relay = Relay::new(server.local_addr().expect("server address"), Some(2 * 4));
    let (server_connection, client_connection) = moirai_executor::block_on(futures::future::join(
        server.accept(),
        client.connect(relay.address, "localhost"),
    ));
    let server_connection = server_connection.expect("authenticated relayed server");
    let client_connection = client_connection.expect("authenticated relayed client");
    relay.state.impair.store(true, Ordering::Release);
    drop(_arrange);
    let expected = payload(65_536);
    let (received, echoed) = exchange(&server_connection, &client_connection, &expected);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(received.expect("loss-recovered input"), expected);
    assert_eq!(echoed.expect("loss-recovered echo"), expected);
    assert_eq!(relay.state.dropped.load(Ordering::Relaxed), 2);
    assert_eq!(relay.state.reordered.load(Ordering::Relaxed), 2);
}

fn exchange(
    server_connection: &QuicConnection,
    client_connection: &QuicConnection,
    expected: &[u8],
) -> (Result<Vec<u8>, QuicError>, Result<Vec<u8>, QuicError>) {
    moirai_executor::block_on(futures::future::join(
        async {
            let mut stream = server_connection
                .accept_stream()
                .await
                .expect("server stream");
            let bytes = stream.recv_message().await?.ok_or(QuicError::Closed)?;
            stream.send_message(&bytes).await?;
            stream.finish()?;
            Ok(bytes)
        },
        async {
            let mut stream = client_connection
                .open_stream()
                .await
                .expect("client stream");
            stream.send_message(expected).await?;
            stream.finish()?;
            let reply = stream.recv_message().await;
            if matches!(reply, Err(QuicError::TimedOut(_))) {
                assert!(
                    matches!(stream.recv_message().await, Err(QuicError::Closed)),
                    "timed-out framing remains terminal"
                );
            }
            reply?.ok_or(QuicError::Closed)
        },
    ))
}

#[test]
fn perpetual_periodic_loss_returns_complete_bytes_or_a_terminal_deadline() {
    let _trace = trace();
    let executor = executor();
    let (server, client) = endpoints(&executor, &config(65_536, 2, DEADLINE));
    let relay = Relay::new(server.local_addr().expect("server address"), None);
    let (server_connection, client_connection) = moirai_executor::block_on(futures::future::join(
        server.accept(),
        client.connect(relay.address, "localhost"),
    ));
    let server_connection = server_connection.expect("authenticated server");
    let client_connection = client_connection.expect("authenticated client");
    relay.state.impair.store(true, Ordering::Release);
    let expected = payload(65_536);
    let (received, echoed) = exchange(&server_connection, &client_connection, &expected);
    // Loss rate alone provides no delivery guarantee: a repeating pattern can
    // discard every retransmission of one byte interval. Success must contain
    // every byte; starvation must fail under the unchanged operation deadline.
    for result in [received, echoed] {
        match result {
            Ok(bytes) => assert_eq!(bytes, expected),
            Err(QuicError::TimedOut(operation)) => {
                assert!(matches!(operation, "receive message" | "send message"))
            }
            Err(error) if matches!(error.io_kind(), io::ErrorKind::ConnectionReset) => {
                // The sibling endpoint resets its stream after its deadline.
                assert!(matches!(
                    error,
                    QuicError::Read(_) | QuicError::ReadExact(_) | QuicError::Write(_)
                ));
            }
            Err(error) => panic!("unexpected loss outcome: {error}"),
        }
    }
    assert!(relay.state.dropped.load(Ordering::Relaxed) > 0);
    assert!(relay.state.reordered.load(Ordering::Relaxed) > 0);
}
