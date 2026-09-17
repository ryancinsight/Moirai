use super::*;
use futures::executor::block_on;
use std::future::Future;
use std::io::{Read, Write};
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::time::Duration;

const MAX_SELF_WAKE_POLLS: usize = 100_000;

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
            IoReactor::get_active().is_none(),
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
            IoReactor::get_active().is_none(),
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
#[cfg(windows)]
fn dropping_owned_recv_future_retires_only_its_waiter() {
    let reactor = IoReactor::new().expect("reactor");
    let socket = block_on(AsyncUdpSocket::bind(
        "127.0.0.1:0".parse().expect("loopback address"),
    ))
    .expect("receiver bind");
    let fd = socket.inner.as_raw_socket() as crate::RawFd;
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
#[cfg(windows)]
fn dropping_polled_stream_retires_waiter_before_socket() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("listener bind");
    let client = StdTcpStream::connect(listener.local_addr().expect("listener address"))
        .expect("client connect");
    let (server, _) = listener.accept().expect("server accept");
    let mut stream = AsyncTcpStream::from_std(server).expect("async stream");
    let fd = stream.inner.as_raw_socket() as crate::RawFd;
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
