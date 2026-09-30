//! Platform-agnostic async network I/O operations.

#![cfg(any(unix, windows))]

use std::io;
use std::net::{Shutdown, SocketAddr};
use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::sync::Arc;
use std::task::Poll;
use std::{future::poll_fn, task::Context};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

use crate::Interest;
use crate::reactor::IoReactor;
#[cfg(windows)]
use crate::reactor::socket_owner::SocketLease;
use crate::reactor::waiter_cancellation::WaiterCancellation;

mod connect;

/// Descriptor a readiness waiter registers under.
///
/// Unix readiness syscalls hold no user memory, so a waiter needs no ownership
/// of the descriptor: it only has to retire its registration before the
/// descriptor closes, which the field and local declaration order of every
/// waiter owner guarantees.
#[cfg(unix)]
#[derive(Clone)]
struct SocketLease(crate::RawFd);

#[cfg(unix)]
impl<S: AsRawFd> From<&Arc<S>> for SocketLease {
    fn from(socket: &Arc<S>) -> Self {
        Self(socket.as_raw_fd())
    }
}

fn wake_without_active_reactor(cx: &Context<'_>) {
    cx.waker().wake_by_ref();
    std::thread::yield_now();
}

fn register_readiness(
    reactor: &IoReactor,
    owner: &SocketLease,
    interest: Interest,
    cx: &Context<'_>,
) -> io::Result<WaiterCancellation> {
    #[cfg(unix)]
    {
        reactor.register_owned_waker(owner.0, interest, cx.waker().clone())
    }
    #[cfg(windows)]
    {
        let fd = owner.raw_socket() as crate::RawFd;
        reactor.register_owned_waker(fd, interest, cx.waker().clone(), owner.clone())
    }
}

/// Shared readiness scaffolding for every non-blocking socket operation: run
/// `op` once; on success or a real error resolve immediately, on `WouldBlock`
/// register the task's waker with the active reactor for (`owner`, `interest`)
/// — or self-wake (cooperative busy-poll) when no reactor is active — and stay
/// pending.
///
/// `waiter` holds the armed registration. Dropping it retires the registration,
/// so its owner declares it before the socket it guards.
fn poll_ready_op<T>(
    cx: &mut Context<'_>,
    owner: SocketLease,
    interest: Interest,
    waiter: &mut Option<WaiterCancellation>,
    op: impl FnOnce() -> io::Result<T>,
) -> Poll<io::Result<T>> {
    match op() {
        Ok(value) => {
            waiter.take();
            Poll::Ready(Ok(value))
        }
        Err(ref error) if error.kind() == io::ErrorKind::WouldBlock => {
            let registration = IoReactor::with_current(|reactor| {
                reactor.map(|reactor| register_readiness(reactor, &owner, interest, cx))
            });
            match registration {
                Some(Ok(registration)) => {
                    *waiter = Some(registration);
                    Poll::Pending
                }
                Some(Err(error)) => {
                    waiter.take();
                    Poll::Ready(Err(error))
                }
                None => {
                    waiter.take();
                    wake_without_active_reactor(cx);
                    Poll::Pending
                }
            }
        }
        Err(error) => {
            waiter.take();
            Poll::Ready(Err(error))
        }
    }
}

/// Non-blocking TCP stream driven by the fd-readiness reactor.
///
/// The waiter fields are declared before `inner`, so dropping the stream
/// retires its armed registrations before the socket closes.
pub struct AsyncTcpStream {
    read_waiter: Option<WaiterCancellation>,
    write_waiter: Option<WaiterCancellation>,
    inner: Arc<StdTcpStream>,
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
        Ok(Self::from_nonblocking(inner))
    }

    /// Shut down the write half of the connection.
    ///
    /// # Errors
    /// Propagates the underlying socket error.
    pub fn shutdown_write(&self) -> io::Result<()> {
        self.inner.shutdown(Shutdown::Write)
    }

    /// Poll a non-blocking read into `buf`.
    ///
    /// The stream owns the armed read registration. Dropping a borrowing
    /// future does not retire that registration until another read replaces it
    /// or the stream is dropped; the socket stays open for every platform poll
    /// in either case. [`Self::read`] owns cancellation at the individual
    /// future boundary.
    pub fn poll_read(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<usize>> {
        let owner = SocketLease::from(&self.inner);
        poll_ready_op(cx, owner, Interest::READABLE, &mut self.read_waiter, || {
            io::Read::read(&mut &*self.inner, buf)
        })
    }

    /// Poll a non-blocking write of `buf`.
    ///
    /// The stream owns the armed write registration. Dropping a borrowing
    /// future leaves that registration reusable until another write replaces it
    /// or the stream is dropped. [`Self::write`] owns cancellation at the
    /// individual future boundary.
    pub fn poll_write(&mut self, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        let owner = SocketLease::from(&self.inner);
        poll_ready_op(
            cx,
            owner,
            Interest::WRITABLE,
            &mut self.write_waiter,
            || io::Write::write(&mut &*self.inner, buf),
        )
    }

    /// Poll a non-blocking flush.
    ///
    /// This shares the stream-owned write registration described by
    /// [`Self::poll_write`]. [`Self::flush`] owns cancellation at the individual
    /// future boundary.
    pub fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let owner = SocketLease::from(&self.inner);
        poll_ready_op(
            cx,
            owner,
            Interest::WRITABLE,
            &mut self.write_waiter,
            || io::Write::flush(&mut &*self.inner),
        )
    }

    /// Read into `buf`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket read errors.
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let owner = SocketLease::from(&self.inner);
        let mut waiter = None;
        poll_fn(|cx| {
            poll_ready_op(cx, owner.clone(), Interest::READABLE, &mut waiter, || {
                io::Read::read(&mut &*self.inner, buf)
            })
        })
        .await
    }

    /// Write `buf`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket write errors.
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let owner = SocketLease::from(&self.inner);
        let mut waiter = None;
        poll_fn(|cx| {
            poll_ready_op(cx, owner.clone(), Interest::WRITABLE, &mut waiter, || {
                io::Write::write(&mut &*self.inner, buf)
            })
        })
        .await
    }

    /// Flush the stream, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket flush errors.
    pub async fn flush(&mut self) -> io::Result<()> {
        let owner = SocketLease::from(&self.inner);
        let mut waiter = None;
        poll_fn(|cx| {
            poll_ready_op(cx, owner.clone(), Interest::WRITABLE, &mut waiter, || {
                io::Write::flush(&mut &*self.inner)
            })
        })
        .await
    }

    fn from_nonblocking(inner: StdTcpStream) -> Self {
        Self {
            read_waiter: None,
            write_waiter: None,
            inner: Arc::new(inner),
        }
    }
}

/// Non-blocking TCP listener driven by the fd-readiness reactor.
pub struct AsyncTcpListener {
    inner: Arc<StdTcpListener>,
}

impl AsyncTcpListener {
    /// Bind a non-blocking listener to `addr`.
    ///
    /// # Errors
    /// Propagates bind and non-blocking-mode errors.
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = StdTcpListener::bind(addr)?;
        inner.set_nonblocking(true)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Accept one inbound connection, awaiting readiness.
    ///
    /// # Errors
    /// Propagates accept and non-blocking-mode errors.
    pub async fn accept(&self) -> io::Result<(AsyncTcpStream, SocketAddr)> {
        let owner = SocketLease::from(&self.inner);
        let mut waiter = None;
        poll_fn(|cx| {
            poll_ready_op(cx, owner.clone(), Interest::READABLE, &mut waiter, || {
                let (stream, addr) = self.inner.accept()?;
                stream.set_nonblocking(true)?;
                Ok((AsyncTcpStream::from_nonblocking(stream), addr))
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
    inner: Arc<std::net::UdpSocket>,
}

impl AsyncUdpSocket {
    /// Bind a non-blocking UDP socket to `addr`.
    ///
    /// # Errors
    /// Propagates bind and non-blocking-mode errors.
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = std::net::UdpSocket::bind(addr)?;
        inner.set_nonblocking(true)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Send one datagram to `target`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket send errors.
    pub async fn send_to(&self, buf: &[u8], target: SocketAddr) -> io::Result<usize> {
        let owner = SocketLease::from(&self.inner);
        let mut waiter = None;
        poll_fn(|cx| {
            poll_ready_op(cx, owner.clone(), Interest::WRITABLE, &mut waiter, || {
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
        let owner = SocketLease::from(&self.inner);
        let mut waiter = None;
        poll_fn(|cx| {
            poll_ready_op(cx, owner.clone(), Interest::READABLE, &mut waiter, || {
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
#[path = "net/tests.rs"]
mod tests;
