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

pub struct AsyncTcpStream {
    inner: StdTcpStream,
}

impl AsyncTcpStream {
    pub fn peer_addr(&self) -> io::Result<SocketAddr> {
        self.inner.peer_addr()
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

    pub fn set_nodelay(&self, on: bool) -> io::Result<()> {
        self.inner.set_nodelay(on)
    }

    pub fn from_std(inner: StdTcpStream) -> io::Result<Self> {
        inner.set_nonblocking(true)?;
        Ok(Self { inner })
    }

    pub fn shutdown_write(&self) -> io::Result<()> {
        self.inner.shutdown(Shutdown::Write)
    }

    pub async fn connect(addr: SocketAddr) -> io::Result<Self> {
        let inner = StdTcpStream::connect(addr)?;
        Self::from_std(inner)
    }

    pub fn poll_read(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<usize>> {
        match io::Read::read(&mut &self.inner, buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(reactor) = IoReactor::get_active() {
                    let raw = socket_to_raw(&self.inner);
                    if let Err(err) =
                        reactor.register_waker(raw, Interest::READABLE, cx.waker().clone())
                    {
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

    pub fn poll_write(&mut self, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        match io::Write::write(&mut &self.inner, buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(reactor) = IoReactor::get_active() {
                    let raw = socket_to_raw(&self.inner);
                    if let Err(err) =
                        reactor.register_waker(raw, Interest::WRITABLE, cx.waker().clone())
                    {
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

    pub fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match io::Write::flush(&mut &self.inner) {
            Ok(()) => Poll::Ready(Ok(())),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(reactor) = IoReactor::get_active() {
                    let raw = socket_to_raw(&self.inner);
                    if let Err(err) =
                        reactor.register_waker(raw, Interest::WRITABLE, cx.waker().clone())
                    {
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

    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        poll_fn(|cx| self.poll_read(cx, buf)).await
    }

    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        poll_fn(|cx| self.poll_write(cx, buf)).await
    }

    pub async fn flush(&mut self) -> io::Result<()> {
        poll_fn(|cx| self.poll_flush(cx)).await
    }
}

pub struct AsyncTcpListener {
    inner: StdTcpListener,
}

impl AsyncTcpListener {
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = StdTcpListener::bind(addr)?;
        inner.set_nonblocking(true)?;
        Ok(Self { inner })
    }

    pub async fn accept(&self) -> io::Result<(AsyncTcpStream, SocketAddr)> {
        poll_fn(|cx| match self.inner.accept() {
            Ok((stream, addr)) => {
                stream.set_nonblocking(true)?;
                Poll::Ready(Ok((AsyncTcpStream { inner: stream }, addr)))
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(reactor) = IoReactor::get_active() {
                    let raw = socket_to_raw(&self.inner);
                    if let Err(err) =
                        reactor.register_waker(raw, Interest::READABLE, cx.waker().clone())
                    {
                        return Poll::Ready(Err(err));
                    }
                    Poll::Pending
                } else {
                    wake_without_active_reactor(cx);
                    Poll::Pending
                }
            }
            Err(e) => Poll::Ready(Err(e)),
        })
        .await
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

pub struct AsyncUdpSocket {
    inner: std::net::UdpSocket,
}

impl AsyncUdpSocket {
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = std::net::UdpSocket::bind(addr)?;
        inner.set_nonblocking(true)?;
        Ok(Self { inner })
    }

    pub async fn send_to(&self, buf: &[u8], target: SocketAddr) -> io::Result<usize> {
        poll_fn(|cx| match self.inner.send_to(buf, target) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(reactor) = IoReactor::get_active() {
                    let raw = socket_to_raw(&self.inner);
                    if let Err(err) =
                        reactor.register_waker(raw, Interest::WRITABLE, cx.waker().clone())
                    {
                        return Poll::Ready(Err(err));
                    }
                    Poll::Pending
                } else {
                    wake_without_active_reactor(cx);
                    Poll::Pending
                }
            }
            Err(e) => Poll::Ready(Err(e)),
        })
        .await
    }

    pub async fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)> {
        poll_fn(|cx| match self.inner.recv_from(buf) {
            Ok((n, addr)) => Poll::Ready(Ok((n, addr))),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(reactor) = IoReactor::get_active() {
                    let raw = socket_to_raw(&self.inner);
                    if let Err(err) =
                        reactor.register_waker(raw, Interest::READABLE, cx.waker().clone())
                    {
                        return Poll::Ready(Err(err));
                    }
                    Poll::Pending
                } else {
                    wake_without_active_reactor(cx);
                    Poll::Pending
                }
            }
            Err(e) => Poll::Ready(Err(e)),
        })
        .await
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }

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
    fn tcp_accept_read_write_round_trip_via_reactor() {
        block_on(async {
            let listener =
                AsyncTcpListener::bind("127.0.0.1:0".parse().expect("loopback address must parse"))
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
    }

    #[test]
    fn udp_recv_round_trip_via_reactor() {
        block_on(async {
            let receiver =
                AsyncUdpSocket::bind("127.0.0.1:0".parse().expect("loopback address must parse"))
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
    }
}
