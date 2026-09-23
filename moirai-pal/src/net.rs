//! Platform-agnostic async network I/O operations.

#![cfg(any(unix, windows))]

use std::io;
use std::net::{Shutdown, SocketAddr};
use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream};
#[cfg(windows)]
use std::sync::Arc;
use std::task::Poll;
use std::{future::poll_fn, task::Context};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

use crate::Interest;
use crate::reactor::IoReactor;
#[cfg(windows)]
use crate::reactor::socket_owner::SocketLease;
#[cfg(windows)]
use crate::reactor::waiter_cancellation::WaiterCancellation;

mod connect;

#[cfg(unix)]
fn socket_to_raw(s: &impl AsRawFd) -> crate::RawFd {
    s.as_raw_fd()
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
#[cfg(unix)]
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

#[cfg(windows)]
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
            if let Some(reactor) = IoReactor::get_active() {
                let fd = owner.raw_socket() as crate::RawFd;
                match reactor.register_owned_waker(fd, interest, cx.waker().clone(), owner) {
                    Ok(registration) => {
                        *waiter = Some(registration);
                        Poll::Pending
                    }
                    Err(error) => {
                        waiter.take();
                        Poll::Ready(Err(error))
                    }
                }
            } else {
                waiter.take();
                wake_without_active_reactor(cx);
                Poll::Pending
            }
        }
        Err(error) => {
            waiter.take();
            Poll::Ready(Err(error))
        }
    }
}

/// Non-blocking TCP stream driven by the fd-readiness reactor.
pub struct AsyncTcpStream {
    #[cfg(windows)]
    read_waiter: Option<WaiterCancellation>,
    #[cfg(windows)]
    write_waiter: Option<WaiterCancellation>,
    #[cfg(windows)]
    inner: Arc<StdTcpStream>,
    #[cfg(unix)]
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
    /// On Windows the stream owns the armed read registration. Dropping a
    /// borrowing future does not retire that registration until another read
    /// replaces it or the stream is dropped; the socket remains owned for
    /// every platform poll in either case. [`Self::read`] owns cancellation at
    /// the individual future boundary.
    pub fn poll_read(&mut self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<usize>> {
        #[cfg(unix)]
        {
            let fd = socket_to_raw(&self.inner);
            poll_ready_op(cx, fd, Interest::READABLE, || {
                io::Read::read(&mut &self.inner, buf)
            })
        }
        #[cfg(windows)]
        {
            let owner = SocketLease::from(&self.inner);
            poll_ready_op(cx, owner, Interest::READABLE, &mut self.read_waiter, || {
                io::Read::read(&mut &*self.inner, buf)
            })
        }
    }

    /// Poll a non-blocking write of `buf`.
    ///
    /// On Windows the stream owns the armed write registration. Dropping a
    /// borrowing future leaves that registration reusable until another write
    /// replaces it or the stream is dropped. [`Self::write`] owns cancellation
    /// at the individual future boundary.
    pub fn poll_write(&mut self, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        #[cfg(unix)]
        {
            let fd = socket_to_raw(&self.inner);
            poll_ready_op(cx, fd, Interest::WRITABLE, || {
                io::Write::write(&mut &self.inner, buf)
            })
        }
        #[cfg(windows)]
        {
            let owner = SocketLease::from(&self.inner);
            poll_ready_op(
                cx,
                owner,
                Interest::WRITABLE,
                &mut self.write_waiter,
                || io::Write::write(&mut &*self.inner, buf),
            )
        }
    }

    /// Poll a non-blocking flush.
    ///
    /// This shares the stream-owned Windows write registration described by
    /// [`Self::poll_write`]. [`Self::flush`] owns cancellation at the individual
    /// future boundary.
    pub fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        #[cfg(unix)]
        {
            let fd = socket_to_raw(&self.inner);
            poll_ready_op(cx, fd, Interest::WRITABLE, || {
                io::Write::flush(&mut &self.inner)
            })
        }
        #[cfg(windows)]
        {
            let owner = SocketLease::from(&self.inner);
            poll_ready_op(
                cx,
                owner,
                Interest::WRITABLE,
                &mut self.write_waiter,
                || io::Write::flush(&mut &*self.inner),
            )
        }
    }

    /// Read into `buf`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket read errors.
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        #[cfg(unix)]
        {
            poll_fn(|cx| self.poll_read(cx, buf)).await
        }
        #[cfg(windows)]
        {
            let owner = SocketLease::from(&self.inner);
            let mut waiter = None;
            poll_fn(|cx| {
                poll_ready_op(cx, owner.clone(), Interest::READABLE, &mut waiter, || {
                    io::Read::read(&mut &*self.inner, buf)
                })
            })
            .await
        }
    }

    /// Write `buf`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket write errors.
    pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        #[cfg(unix)]
        {
            poll_fn(|cx| self.poll_write(cx, buf)).await
        }
        #[cfg(windows)]
        {
            let owner = SocketLease::from(&self.inner);
            let mut waiter = None;
            poll_fn(|cx| {
                poll_ready_op(cx, owner.clone(), Interest::WRITABLE, &mut waiter, || {
                    io::Write::write(&mut &*self.inner, buf)
                })
            })
            .await
        }
    }

    /// Flush the stream, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket flush errors.
    pub async fn flush(&mut self) -> io::Result<()> {
        #[cfg(unix)]
        {
            poll_fn(|cx| self.poll_flush(cx)).await
        }
        #[cfg(windows)]
        {
            let owner = SocketLease::from(&self.inner);
            let mut waiter = None;
            poll_fn(|cx| {
                poll_ready_op(cx, owner.clone(), Interest::WRITABLE, &mut waiter, || {
                    io::Write::flush(&mut &*self.inner)
                })
            })
            .await
        }
    }

    fn from_nonblocking(inner: StdTcpStream) -> Self {
        Self {
            #[cfg(windows)]
            read_waiter: None,
            #[cfg(windows)]
            write_waiter: None,
            #[cfg(windows)]
            inner: Arc::new(inner),
            #[cfg(unix)]
            inner,
        }
    }
}

/// Non-blocking TCP listener driven by the fd-readiness reactor.
pub struct AsyncTcpListener {
    #[cfg(windows)]
    inner: Arc<StdTcpListener>,
    #[cfg(unix)]
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
        Ok(Self {
            #[cfg(windows)]
            inner: Arc::new(inner),
            #[cfg(unix)]
            inner,
        })
    }

    /// Accept one inbound connection, awaiting readiness.
    ///
    /// # Errors
    /// Propagates accept and non-blocking-mode errors.
    pub async fn accept(&self) -> io::Result<(AsyncTcpStream, SocketAddr)> {
        // `socket_to_raw` is evaluated inside the poll closure: on Windows a
        // `RawFd` is a raw pointer (`!Send`), so holding it across an await
        // would make this future `!Send`.
        #[cfg(windows)]
        let owner = SocketLease::from(&self.inner);
        #[cfg(windows)]
        let mut waiter = None;
        poll_fn(|cx| {
            #[cfg(unix)]
            {
                poll_ready_op(cx, socket_to_raw(&self.inner), Interest::READABLE, || {
                    let (stream, addr) = self.inner.accept()?;
                    stream.set_nonblocking(true)?;
                    Ok((AsyncTcpStream::from_nonblocking(stream), addr))
                })
            }
            #[cfg(windows)]
            {
                poll_ready_op(cx, owner.clone(), Interest::READABLE, &mut waiter, || {
                    let (stream, addr) = self.inner.accept()?;
                    stream.set_nonblocking(true)?;
                    Ok((AsyncTcpStream::from_nonblocking(stream), addr))
                })
            }
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
    #[cfg(windows)]
    inner: Arc<std::net::UdpSocket>,
    #[cfg(unix)]
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
        Ok(Self {
            #[cfg(windows)]
            inner: Arc::new(inner),
            #[cfg(unix)]
            inner,
        })
    }

    /// Send one datagram to `target`, awaiting readiness.
    ///
    /// # Errors
    /// Propagates socket send errors.
    pub async fn send_to(&self, buf: &[u8], target: SocketAddr) -> io::Result<usize> {
        // `socket_to_raw` stays inside the poll closure (`RawFd` is `!Send` on
        // Windows; see `AsyncTcpListener::accept`).
        #[cfg(windows)]
        let owner = SocketLease::from(&self.inner);
        #[cfg(windows)]
        let mut waiter = None;
        poll_fn(|cx| {
            #[cfg(unix)]
            {
                poll_ready_op(cx, socket_to_raw(&self.inner), Interest::WRITABLE, || {
                    self.inner.send_to(buf, target)
                })
            }
            #[cfg(windows)]
            {
                poll_ready_op(cx, owner.clone(), Interest::WRITABLE, &mut waiter, || {
                    self.inner.send_to(buf, target)
                })
            }
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
        #[cfg(windows)]
        let owner = SocketLease::from(&self.inner);
        #[cfg(windows)]
        let mut waiter = None;
        poll_fn(|cx| {
            #[cfg(unix)]
            {
                poll_ready_op(cx, socket_to_raw(&self.inner), Interest::READABLE, || {
                    self.inner.recv_from(buf)
                })
            }
            #[cfg(windows)]
            {
                poll_ready_op(cx, owner.clone(), Interest::READABLE, &mut waiter, || {
                    self.inner.recv_from(buf)
                })
            }
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
