//! Non-blocking TCP connect: start the handshake without waiting, then decide
//! its outcome once the socket settles.

use std::io;
use std::net::{SocketAddr, TcpStream as StdTcpStream};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

/// Create a non-blocking TCP socket and start connecting it to `addr`.
///
/// Returns once the connect is either complete or in progress; it never waits
/// for the handshake.
pub(super) fn start_connect(addr: SocketAddr) -> io::Result<StdTcpStream> {
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

/// Resolve the state of an in-progress non-blocking connect.
///
/// Until the socket reports writable, error, or hangup readiness the connect
/// is in progress, reported as `WouldBlock` so the caller arms writable
/// interest. The readiness probe is required rather than inferring progress
/// from `getpeername`: Winsock answers `getpeername` with the target address
/// while the handshake is still pending, and before a refusal lands. Once
/// settled, a pending `SO_ERROR` is the connect failure, and otherwise
/// `getpeername` confirms the connection.
pub(super) fn connect_outcome(stream: &StdTcpStream) -> io::Result<()> {
    if !connect_settled(stream)? {
        return Err(io::ErrorKind::WouldBlock.into());
    }
    if let Some(error) = stream.take_error()? {
        return Err(error);
    }
    stream.peer_addr().map(drop)
}

/// Zero-timeout readiness probe: has the connect on `stream` completed or
/// failed?
#[cfg(unix)]
fn connect_settled(stream: &StdTcpStream) -> io::Result<bool> {
    let mut probe = libc::pollfd {
        fd: stream.as_raw_fd(),
        events: libc::POLLOUT,
        revents: 0,
    };
    // SAFETY: `probe` is one initialized `pollfd` that outlives the call, the
    // count is 1, and a zero timeout returns without blocking; `poll` writes
    // only `revents`.
    match unsafe { libc::poll(&raw mut probe, 1, 0) } {
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

/// Zero-timeout readiness probe: has the connect on `stream` completed or
/// failed?
///
/// `WSAPoll` reports a failed connect as `POLLHUP | POLLERR` from Windows 10
/// version 2004 on; earlier builds never signal it, so there a refused connect
/// resolves only at the caller's own deadline.
#[cfg(windows)]
fn connect_settled(stream: &StdTcpStream) -> io::Result<bool> {
    use std::os::windows::io::AsRawSocket;
    use windows::Win32::Networking::WinSock::{
        POLLERR, POLLHUP, POLLWRNORM, SOCKET, SOCKET_ERROR, WSAGetLastError, WSAPOLL_EVENT_FLAGS,
        WSAPOLLFD, WSAPoll,
    };

    let socket = usize::try_from(stream.as_raw_socket())
        .map_err(|_| io::Error::other("socket handle exceeds the platform word"))?;
    let mut probe = [WSAPOLLFD {
        fd: SOCKET(socket),
        events: POLLWRNORM,
        revents: WSAPOLL_EVENT_FLAGS(0),
    }];
    // SAFETY: `probe` is a valid one-element `WSAPOLLFD` array that outlives
    // the call, and a zero timeout returns without blocking; `WSAPoll` writes
    // only `revents`.
    let ready = unsafe { WSAPoll(probe.as_mut_ptr(), 1, 0) };
    if ready == SOCKET_ERROR {
        // SAFETY: `WSAGetLastError` has no preconditions; it is read
        // immediately after the failed call, as Winsock requires.
        return Err(io::Error::from_raw_os_error(unsafe { WSAGetLastError() }.0));
    }
    let settled = POLLWRNORM.0 | POLLERR.0 | POLLHUP.0;
    Ok(ready > 0 && probe[0].revents.0 & settled != 0)
}
