//! Platform-agnostic async network I/O operations.

use std::io;
use std::net::{Shutdown, SocketAddr};
use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::task::Poll;
use std::{future::poll_fn, task::Context};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;

use crate::reactor::IoReactor;
use crate::Interest;

#[cfg(unix)]
fn socket_to_raw(s: &impl AsRawFd) -> crate::RawFd {
    s.as_raw_fd()
}

#[cfg(windows)]
fn socket_to_raw(s: &impl AsRawSocket) -> crate::RawFd {
    s.as_raw_socket() as crate::RawFd
}

fn wake_without_active_reactor(cx: &Context<'_>) {
    cx.waker().wake_by_ref();
    std::thread::yield_now();
}

/// Shared readiness scaffolding for every non-blocking socket operation: run
/// `op` once; on success or a real error resolve immediately, on `WouldBlock`
/// register the task's waker with the active reactor for (`fd`, `interest`) —
/// or self-wake (cooperative busy-poll) when no reactor is active — and stay
/// pending.
fn poll_ready_op<T>(
    cx: &mut Context<'_>,
    fd: crate::RawFd,
    interest: Interest,
    op: impl FnOnce() -> io::Result<T>,
) -> Poll<io::Result<T>> {
    match op() {
        Ok(value) => Poll::Ready(Ok(value)),
        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
            if let Some(reactor) = IoReactor::get_active() {
                if let Err(err) = reactor.register_waker(fd, interest, cx.waker().clone()) {
                    return Poll::Ready(Err(err));
                }
                Poll::Pending
            } else {
                wake_without_active_reactor(cx);
                Poll::Pending
            }
        }
        Err(e) => Poll::Ready(Err(e)),
    }
}

/// Non-blocking TCP stream driven by the fd-readiness reactor.
pub struct AsyncTcpStream {
    inner: StdTcpStream,
}

impl AsyncTcpStream {
    /// Peer socket address.
    ///
    /// # Errors
    /// Propagates the underlying socket error.
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    /// Local socket address.
    ///
    /// # Errors
    /// Propagates the underlying socket error.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Enable or disable `TCP_NODELAY`.
    ///
    /// # Errors
    /// Propagates the underlying socket error.
    pub fn set_nodelay(&self, on: bool) -> io::Result<()> {
        self.inner.set_nodelay(on)
    }

    /// Wrap a std stream, switching it to non-blocking mode.
    ///
    /// # Errors
    /// Propagates the non-blocking-mode error.
    pub fn from_std(inner: StdTcpStream) -> io::Result<Self> {
        inner.set_nonblocking(true)?;
        Ok(Self { inner })
    }

    /// Shut down the write half of the connection.
    ///
    /// # Errors
    /// Propagates the underlying socket error.
    pub fn shutdown_write(&self) -> io::Result<()> {
        self.inner.shutdown(Shutdown::Write)
    }

    /// Connect to `addr` and switch the stream to non-blocking mode.
    ///
    /// # Errors
    /// Propagates connection and non-blocking-mode errors.
    pub async fn connect(addr: SocketAddr) -> io::Result<Self> {
        let inner = StdTcpStream::connect(addr)?;
        Self::from_std(inner)
    }

    /// Poll a non-blocking read into `buf`.
    pub fn poll_read(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<usize>> {
        let fd = socket_to_raw(&self.inner);
        poll_ready_op(cx, fd, Interest::READABLE, || {
            io::Read::read(&mut &self.inner, buf)
        })
    }

    /// Poll a non-blocking write of `buf`.
    pub fn poll_write(&mut self, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        let fd = socket_to_raw(&self.inner);
        poll_ready_op(cx, fd, Interest::WRITABLE, || {
            io::Write::write(&mut &self.inner, buf)
        })
    }

    /// Poll a non-blocking flush.
    pub fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let fd = socket_to_raw(&self.inner);
        poll_ready_op(cx, fd, Interest::WRITABLE, || {
            io::Write::flush(&mut &self.inner)
        })
    }

    /// Read into `buf`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket read errors.
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        poll_fn(|cx| self.poll_read(cx, buf)).await
    }

    /// Write `buf`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket write errors.
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        poll_fn(|cx| self.poll_write(cx, buf)).await
    }

    /// Flush the stream, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket flush errors.
    pub async fn flush(&mut self) -> io::Result<()> {
        poll_fn(|cx| self.poll_flush(cx)).await
    }
}

/// Non-blocking TCP listener driven by the fd-readiness reactor.
pub struct AsyncTcpListener {
    inner: StdTcpListener,
}

impl AsyncTcpListener {
    /// Bind a non-blocking listener to `addr`.
    ///
    /// # Errors
    /// Propagates bind and non-blocking-mode errors.
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = StdTcpListener::bind(addr)?;
        inner.set_nonblocking(true)?;
        Ok(Self { inner })
    }

    /// Accept one inbound connection, awaiting readiness.
    ///
    /// # Errors
    /// Propagates accept and non-blocking-mode errors.
    pub async fn accept(&self) -> io::Result<(AsyncTcpStream, SocketAddr)> {
        // `socket_to_raw` is evaluated inside the poll closure: on Windows a
        // `RawFd` is a raw pointer (`!Send`), so holding it across an await
        // would make this future `!Send`.
        poll_fn(|cx| {
            poll_ready_op(cx, socket_to_raw(&self.inner), Interest::READABLE, || {
                let (stream, addr) = self.inner.accept()?;
                stream.set_nonblocking(true)?;
                Ok((AsyncTcpStream { inner: stream }, addr))
            })
        })
        .await
    }

    /// Local listener address.
    ///
    /// # Errors
    /// Propagates the underlying socket error.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

/// Non-blocking UDP socket driven by the fd-readiness reactor.
pub struct AsyncUdpSocket {
    inner: std::net::UdpSocket,
}

impl AsyncUdpSocket {
    /// Bind a non-blocking UDP socket to `addr`.
    ///
    /// # Errors
    /// Propagates bind and non-blocking-mode errors.
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = std::net::UdpSocket::bind(addr)?;
        inner.set_nonblocking(true)?;
        Ok(Self { inner })
    }

    /// Send one datagram to `target`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket send errors.
    pub async fn send_to(&self, buf: &[u8], target: SocketAddr) -> io::Result<usize> {
        // `socket_to_raw` stays inside the poll closure (`RawFd` is `!Send` on
        // Windows; see `AsyncTcpListener::accept`).
        poll_fn(|cx| {
            poll_ready_op(cx, socket_to_raw(&self.inner), Interest::WRITABLE, || {
                self.inner.send_to(buf, target)
            })
        })
        .await
    }

    /// Receive one datagram into `buf`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket receive errors.
    pub async fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        // `socket_to_raw` stays inside the poll closure (`RawFd` is `!Send` on
        // Windows; see `AsyncTcpListener::accept`).
        poll_fn(|cx| {
            poll_ready_op(cx, socket_to_raw(&self.inner), Interest::READABLE, || {
                self.inner.recv_from(buf)
            })
        })
        .await
    }

    /// Local socket address.
    ///
    /// # Errors
    /// Propagates the underlying socket error.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    /// Enable or disable `SO_BROADCAST`.
    ///
    /// # Errors
    /// Propagates the underlying socket error.
    pub fn set_broadcast(&self, on: bool) -> io::Result<()> {
        self.inner.set_broadcast(on)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::io::{Read, Write};
    use std::time::Duration;

    #[test]
    fn tcp_accept_read_write_self_wakes_without_active_reactor() {
        // Suppress the global reactor for this thread, so `accept`/`read`/
        // `write` make progress only via the `wake_without_active_reactor`
        // self-wake fallback (busy-poll). The 10 ms client delay guarantees the
        // first poll observes `WouldBlock` and takes that path.
        IoReactor::with_reactor_disabled(|| {
            assert!(
                IoReactor::get_active().is_none(),
                "self-wake path requires no active reactor"
            );
            block_on(async {
                let listener = AsyncTcpListener::bind(
                    "127.0.0.1:0".parse().expect("loopback address must parse"),
                )
                .await
                .expect("listener bind must succeed");
                let addr = listener.local_addr().expect("listener address must exist");

                let client = std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(10));
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

                let (mut stream, peer) = listener.accept().await.expect("accept must complete");
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
                let receiver = AsyncUdpSocket::bind(
                    "127.0.0.1:0".parse().expect("loopback address must parse"),
                )
                .await
                .expect("receiver bind must succeed");
                let target = receiver.local_addr().expect("receiver address must exist");

                let sender = std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(10));
                    let socket =
                        std::net::UdpSocket::bind("127.0.0.1:0").expect("sender bind must succeed");
                    let sent = socket
                        .send_to(b"datagram", target)
                        .expect("datagram send must succeed");
                    assert_eq!(sent, 8);
                });

                let mut buf = [0_u8; 16];
                let (received, _peer) = receiver
                    .recv_from(&mut buf)
                    .await
                    .expect("recv_from must complete");
                assert_eq!(received, 8);
                assert_eq!(&buf[..received], b"datagram");

                sender.join().expect("sender thread must complete");
            });
        });
    }
}
