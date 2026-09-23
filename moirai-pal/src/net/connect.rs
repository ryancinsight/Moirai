//! Non-blocking TCP connect: start the handshake without waiting, then decide
//! its outcome once the socket settles.

use std::future::poll_fn;
use std::io;
use std::net::{SocketAddr, TcpStream as StdTcpStream};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(windows)]
use std::task::Poll;
use std::time::Duration;

use super::{AsyncTcpStream, poll_ready_op};
use crate::Interest;
#[cfg(windows)]
use crate::reactor::socket_owner::SocketLease;

#[cfg(windows)]
mod reprobe;
#[cfg(test)]
mod tests;

impl AsyncTcpStream {
    /// Connect to `addr` without blocking the polling thread.
    ///
    /// The socket is created non-blocking and the connect starts immediately.
    /// While the handshake is in progress, the future registers writable
    /// interest with the active reactor and returns `Pending`. Each poll
    /// decides completion with a zero-timeout probe (`poll` on Unix, `select`
    /// with an exception set on Windows) and reads a failure from `SO_ERROR`.
    /// On Windows, a pending connect is also re-polled every
    /// `CONNECT_REPROBE_INTERVAL` (100 ms), because `WSAPoll` before Windows 10
    /// version 2004 never reports a failed connect. Dropping the future closes
    /// the half-open socket, which aborts the handshake. An outer timeout or
    /// cancellation therefore takes effect at its own deadline rather than the
    /// OS connect timeout.
    ///
    /// # Errors
    /// Propagates socket creation and non-blocking-mode errors, and the
    /// connect failure reported by the OS (for example `ConnectionRefused` or
    /// `TimedOut`).
    pub async fn connect(addr: SocketAddr) -> io::Result<Self> {
        let stream = Self::from_nonblocking(start_connect(addr)?);
        #[cfg(unix)]
        {
            let fd = stream.inner.as_raw_fd();
            poll_fn(|cx| {
                poll_ready_op(cx, fd, Interest::WRITABLE, || {
                    connect_outcome(&stream.inner, Duration::ZERO)
                })
            })
            .await?;
        }
        #[cfg(windows)]
        {
            let owner = SocketLease::from(&stream.inner);
            // Declared after `stream`, so a dropped future retires this
            // registration before the socket closes.
            let mut waiter = None;
            poll_fn(|cx| {
                let polled =
                    poll_ready_op(cx, owner.clone(), Interest::WRITABLE, &mut waiter, || {
                        connect_outcome(&stream.inner, Duration::ZERO)
                    });
                if polled.is_pending()
                    && let Err(error) = reprobe::schedule(cx.waker())
                {
                    return Poll::Ready(Err(error));
                }
                polled
            })
            .await?;
        }
        Ok(stream)
    }
}

/// Create a non-blocking TCP socket and start connecting it to `addr`.
///
/// Returns once the connect is either complete or in progress; it never waits
/// for the handshake.
fn start_connect(addr: SocketAddr) -> io::Result<StdTcpStream> {
    use socket2::{Domain, Protocol, Socket, Type};

    let socket = Socket::new(Domain::for_address(addr), Type::STREAM, Some(Protocol::TCP))?;
    socket.set_nonblocking(true)?;
    match socket.connect(&addr.into()) {
        Ok(()) => {}
        Err(error) if connect_in_progress(&error) => {}
        Err(error) => return Err(error),
    }
    Ok(socket.into())
}

/// Whether a non-blocking `connect` error means "handshake started".
///
/// POSIX reports `EINPROGRESS`; Winsock reports `WSAEWOULDBLOCK`, which std
/// maps to `WouldBlock`.
fn connect_in_progress(error: &io::Error) -> bool {
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(libc::EINPROGRESS)
    }
    #[cfg(windows)]
    {
        error.kind() == io::ErrorKind::WouldBlock
    }
}

/// Decide an in-progress non-blocking connect, waiting up to `wait` for it to
/// settle (`Duration::ZERO` from a poll, which must not block).
///
/// While unsettled the connect is in progress, reported as `WouldBlock` so the
/// caller arms writable interest. The probe is required rather than inferring
/// progress from `getpeername`: Winsock answers `getpeername` with the target
/// address while the handshake is still pending, and before a refusal lands.
/// Once settled, a pending `SO_ERROR` is the connect failure, and otherwise
/// `getpeername` confirms the connection.
fn connect_outcome(stream: &StdTcpStream, wait: Duration) -> io::Result<()> {
    if !connect_settled(stream, wait)? {
        return Err(io::ErrorKind::WouldBlock.into());
    }
    if let Some(error) = stream.take_error()? {
        return Err(error);
    }
    stream.peer_addr().map(drop)
}

/// Has the connect on `stream` completed or failed? `poll` reports both as
/// `POLLOUT`, `POLLERR`, or `POLLHUP` on every supported Unix.
#[cfg(unix)]
fn connect_settled(stream: &StdTcpStream, wait: Duration) -> io::Result<bool> {
    let wait_ms = libc::c_int::try_from(wait.as_millis()).unwrap_or(libc::c_int::MAX);
    let mut probe = libc::pollfd {
        fd: stream.as_raw_fd(),
        events: libc::POLLOUT,
        revents: 0,
    };
    // SAFETY: `probe` is one initialized `pollfd` that outlives the call and
    // the count is 1; `poll` writes only `revents`.
    match unsafe { libc::poll(&raw mut probe, 1, wait_ms) } {
        0 => Ok(false),
        -1 => {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                Ok(false)
            } else {
                Err(error)
            }
        }
        _ => Ok(probe.revents & (libc::POLLOUT | libc::POLLERR | libc::POLLHUP) != 0),
    }
}

/// Has the connect on `stream` completed or failed?
///
/// `select` puts a completed connect in the write set and a failed one in the
/// exception set on every Winsock version. `WSAPoll` is not used here: before
/// Windows 10 version 2004 it never reports a failed connect.
#[cfg(windows)]
fn connect_settled(stream: &StdTcpStream, wait: Duration) -> io::Result<bool> {
    use std::os::windows::io::AsRawSocket;
    use windows::Win32::Networking::WinSock::{
        FD_SET, SOCKET, SOCKET_ERROR, TIMEVAL, WSAGetLastError, select,
    };

    let socket = SOCKET(
        usize::try_from(stream.as_raw_socket())
            .map_err(|_| io::Error::other("socket handle exceeds the platform word"))?,
    );
    let single = || {
        let mut fd_array = FD_SET::default().fd_array;
        fd_array[0] = socket;
        FD_SET {
            fd_count: 1,
            fd_array,
        }
    };
    let mut writable = single();
    let mut failed = single();
    let timeout = TIMEVAL {
        tv_sec: i32::try_from(wait.as_secs()).unwrap_or(i32::MAX),
        // `subsec_micros` is below 1_000_000, so it always fits.
        tv_usec: i32::try_from(wait.subsec_micros()).unwrap_or(0),
    };
    // SAFETY: both sets are initialized, hold exactly one live socket, and
    // outlive the call; `timeout` is a valid `TIMEVAL`. Winsock ignores `nfds`.
    // `select` rewrites only the sets it is given.
    let ready = unsafe {
        select(
            0,
            None,
            Some(&raw mut writable),
            Some(&raw mut failed),
            Some(&raw const timeout),
        )
    };
    if ready == SOCKET_ERROR {
        // SAFETY: `WSAGetLastError` has no preconditions; it is read
        // immediately after the failed call, as Winsock requires.
        return Err(io::Error::from_raw_os_error(unsafe { WSAGetLastError() }.0));
    }
    Ok(writable.fd_count > 0 || failed.fd_count > 0)
}
