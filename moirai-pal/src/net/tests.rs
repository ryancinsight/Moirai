use super::*;
use futures::executor::block_on;
use std::future::Future;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::time::Duration;

const MAX_SELF_WAKE_POLLS: usize = 100_000;

#[cfg(unix)]
fn raw_id(socket: &impl AsRawFd) -> crate::RawFd {
    socket.as_raw_fd()
}

#[cfg(windows)]
fn raw_id(socket: &impl AsRawSocket) -> crate::RawFd {
    socket.as_raw_socket() as crate::RawFd
}

fn poll_until_ready<F>(mut future: std::pin::Pin<&mut F>, context: &mut Context<'_>) -> F::Output
where
    F: Future,
{
    for _ in 0..MAX_SELF_WAKE_POLLS {
        if let Poll::Ready(output) = future.as_mut().poll(context) {
            return output;
        }
        std::thread::yield_now();
    }
    panic!("self-wake operation did not resolve within the poll bound");
}

#[test]
fn tcp_accept_read_write_self_wakes_without_active_reactor() {
    // Suppress the global reactor for this thread, so `accept`/`read`/
    // `write` make progress only via the `wake_without_active_reactor`
    // self-wake fallback (busy-poll). Polling before the client exists
    // deterministically exercises the initial `WouldBlock` path; the
    // client is started only after that pending event is observed.
    IoReactor::with_reactor_disabled(|| {
        assert!(
            IoReactor::with_current(|reactor| reactor.is_none()),
            "self-wake path requires no active reactor"
        );
        block_on(async {
            let listener =
                AsyncTcpListener::bind("127.0.0.1:0".parse().expect("loopback address must parse"))
                    .await
                    .expect("listener bind must succeed");
            let addr = listener.local_addr().expect("listener address must exist");

            let (mut stream, peer, client) = {
                let noop_waker = futures::task::noop_waker();
                let mut context = Context::from_waker(&noop_waker);
                let mut accept = std::pin::pin!(listener.accept());
                assert!(matches!(accept.as_mut().poll(&mut context), Poll::Pending));

                let client = std::thread::spawn(move || {
                    let mut stream =
                        StdTcpStream::connect(addr).expect("client connection must succeed");
                    stream
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .expect("client read timeout must be set");
                    stream
                        .set_write_timeout(Some(Duration::from_secs(2)))
                        .expect("client write timeout must be set");
                    stream
                        .write_all(b"ping")
                        .expect("client write must succeed");

                    let mut echo = [0_u8; 4];
                    stream
                        .read_exact(&mut echo)
                        .expect("client echo must be readable");
                    assert_eq!(&echo, b"pong");
                });

                let (stream, peer) =
                    poll_until_ready(accept.as_mut(), &mut context).expect("accept must complete");
                (stream, peer, client)
            };
            assert_eq!(peer.ip(), addr.ip());

            let mut inbound = [0_u8; 4];
            let read = stream.read(&mut inbound).await.expect("read must complete");
            assert_eq!(read, 4);
            assert_eq!(&inbound, b"ping");

            let mut written = 0;
            while written < 4 {
                let n = stream
                    .write(&b"pong"[written..])
                    .await
                    .expect("write must complete");
                assert_ne!(n, 0);
                written += n;
            }

            client.join().expect("client thread must complete");
        });
    });
}

#[test]
fn udp_recv_self_wakes_without_active_reactor() {
    // As above: with the global reactor suppressed, `recv_from` completes
    // only through the self-wake busy-poll fallback.
    IoReactor::with_reactor_disabled(|| {
        assert!(
            IoReactor::with_current(|reactor| reactor.is_none()),
            "self-wake path requires no active reactor"
        );
        block_on(async {
            let receiver =
                AsyncUdpSocket::bind("127.0.0.1:0".parse().expect("loopback address must parse"))
                    .await
                    .expect("receiver bind must succeed");
            let target = receiver.local_addr().expect("receiver address must exist");

            let noop_waker = futures::task::noop_waker();
            let mut context = Context::from_waker(&noop_waker);
            let mut buf = [0_u8; 16];
            let (result, sender) = {
                let mut receive = std::pin::pin!(receiver.recv_from(&mut buf));
                assert!(matches!(receive.as_mut().poll(&mut context), Poll::Pending));

                // Publish the sender only after the first poll observes
                // WouldBlock. Otherwise a fast localhost datagram can arrive
                // before that poll and turn the assertion into a race.
                let sender = std::thread::spawn(move || {
                    let socket =
                        std::net::UdpSocket::bind("127.0.0.1:0").expect("sender bind must succeed");
                    let sent = socket
                        .send_to(b"datagram", target)
                        .expect("datagram send must succeed");
                    assert_eq!(sent, 8);
                });

                let result = poll_until_ready(receive.as_mut(), &mut context);
                (result, sender)
            };
            let (received, _peer) = result.expect("recv_from must complete");
            assert_eq!(received, 8);
            assert_eq!(&buf[..received], b"datagram");

            sender.join().expect("sender thread must complete");
        });
    });
}

#[test]
fn dropping_owned_recv_future_retires_only_its_waiter() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = block_on(AsyncUdpSocket::bind(
        "127.0.0.1:0".parse().expect("loopback address"),
    ))
    .expect("receiver bind");
    let fd = raw_id(&*socket.inner);
    let noop = futures::task::noop_waker();
    let mut context = Context::from_waker(&noop);
    let mut buffer = [0_u8; 8];
    {
        let mut receive = std::pin::pin!(socket.recv_from(&mut buffer));
        reactor.with_active(|| {
            assert!(matches!(receive.as_mut().poll(&mut context), Poll::Pending));
        });
    }
    assert!(
        !reactor
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&crate::reactor::core::FdKey::from(fd))
    );
    assert!(!reactor.platform_reactor.has_registration(fd));

    let sender = std::net::UdpSocket::bind("127.0.0.1:0").expect("sender bind");
    assert_eq!(
        sender
            .send_to(b"survives", socket.local_addr().expect("receiver address"))
            .expect("send replacement payload"),
        8
    );
    let received = {
        let mut receive = std::pin::pin!(socket.recv_from(&mut buffer));
        let Poll::Ready(Ok((received, _))) =
            reactor.with_active(|| receive.as_mut().poll(&mut context))
        else {
            panic!("replacement receive must consume queued payload");
        };
        received
    };
    assert_eq!(received, 8);
    assert_eq!(&buffer, b"survives");
}

#[test]
fn dropping_polled_stream_retires_waiter_before_socket() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind");
    let client = StdTcpStream::connect(listener.local_addr().expect("listener address"))
        .expect("client connect");
    let (server, _) = listener.accept().expect("server accept");
    let mut stream = AsyncTcpStream::from_std(server).expect("async stream");
    let fd = raw_id(&*stream.inner);
    let reactor_a = IoReactor::new().expect("reactor A");
    let reactor_b = IoReactor::new().expect("reactor B");
    let noop = futures::task::noop_waker();
    let mut context = Context::from_waker(&noop);
    let mut byte = [0_u8; 1];
    reactor_a.with_active(|| {
        assert!(matches!(
            stream.poll_read(&mut context, &mut byte),
            Poll::Pending
        ));
    });
    reactor_b.with_active(|| drop(stream));
    assert!(
        !reactor_a
            .registered_fds
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains_key(&crate::reactor::core::FdKey::from(fd))
    );
    assert!(!reactor_a.platform_reactor.has_registration(fd));
    drop(client);
}

/// Waker that counts wake-ups, so a test observes how often a pending
/// operation is told to re-poll.
struct WakeCounter(std::sync::atomic::AtomicUsize);

impl WakeCounter {
    fn new() -> Arc<Self> {
        Arc::new(Self(std::sync::atomic::AtomicUsize::new(0)))
    }

    fn count(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl std::task::Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn connected_pair() -> (AsyncTcpStream, StdTcpStream) {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind");
    let peer = StdTcpStream::connect(listener.local_addr().expect("listener address"))
        .expect("client connect");
    let (server, _) = listener.accept().expect("server accept");
    (
        AsyncTcpStream::from_std(server).expect("async stream"),
        peer,
    )
}

#[test]
fn self_wake_fallback_wakes_once_per_pending_poll() {
    // Without an active reactor every `WouldBlock` poll re-wakes its own task:
    // N pending polls produce N wakes however long the socket stays idle.
    const POLLS: usize = 1_000;
    let (mut stream, peer) = connected_pair();
    let counter = WakeCounter::new();
    let waker = std::task::Waker::from(Arc::clone(&counter));
    let mut context = Context::from_waker(&waker);
    let mut byte = [0_u8; 1];
    IoReactor::with_reactor_disabled(|| {
        for _ in 0..POLLS {
            assert!(matches!(
                stream.poll_read(&mut context, &mut byte),
                Poll::Pending
            ));
        }
    });
    assert_eq!(counter.count(), POLLS);
    drop(peer);
}

#[test]
fn reactor_readiness_wakes_an_idle_read_exactly_once() {
    // With a reactor the idle read registers once and is woken only by
    // readiness: zero wakes over any number of idle reactor iterations, then
    // one wake when the peer writes.
    const IDLE_ITERATIONS: usize = 1_000;
    let (mut stream, mut peer) = connected_pair();
    let counter = WakeCounter::new();
    let waker = std::task::Waker::from(Arc::clone(&counter));
    let mut context = Context::from_waker(&waker);
    let mut byte = [0_u8; 1];
    let reactor = IoReactor::new().expect("reactor must build");
    reactor.with_active(|| {
        assert!(matches!(
            stream.poll_read(&mut context, &mut byte),
            Poll::Pending
        ));
        for _ in 0..IDLE_ITERATIONS {
            reactor
                .run_iteration(Some(Duration::ZERO))
                .expect("idle iteration must succeed");
        }
        assert_eq!(counter.count(), 0);

        peer.write_all(b"x").expect("peer write");
        reactor
            .run_iteration(Some(Duration::from_secs(5)))
            .expect("readiness iteration must succeed");
        assert_eq!(counter.count(), 1);
        assert!(matches!(
            stream.poll_read(&mut context, &mut byte),
            Poll::Ready(Ok(1))
        ));
        assert_eq!(&byte, b"x");
    });
}
