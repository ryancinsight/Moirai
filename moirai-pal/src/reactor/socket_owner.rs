//! Windows socket ownership retained by readiness snapshots.

use std::net::{TcpListener, TcpStream, UdpSocket};
use std::os::windows::io::{AsRawSocket, RawSocket};
use std::sync::{Arc, Weak};

/// Strong socket ownership held for the duration of one `WSAPoll` call.
#[derive(Clone)]
pub(crate) enum SocketLease {
    TcpStream(Arc<TcpStream>),
    TcpListener(Arc<TcpListener>),
    Udp(Arc<UdpSocket>),
}

impl SocketLease {
    pub(crate) fn raw_socket(&self) -> RawSocket {
        match self {
            Self::TcpStream(socket) => socket.as_raw_socket(),
            Self::TcpListener(socket) => socket.as_raw_socket(),
            Self::Udp(socket) => socket.as_raw_socket(),
        }
    }

    pub(crate) fn downgrade(&self) -> WeakSocketOwner {
        match self {
            Self::TcpStream(socket) => WeakSocketOwner::TcpStream(Arc::downgrade(socket)),
            Self::TcpListener(socket) => WeakSocketOwner::TcpListener(Arc::downgrade(socket)),
            Self::Udp(socket) => WeakSocketOwner::Udp(Arc::downgrade(socket)),
        }
    }
}

/// Non-owning socket identity kept in the platform registration table.
#[derive(Clone)]
pub(crate) enum WeakSocketOwner {
    TcpStream(Weak<TcpStream>),
    TcpListener(Weak<TcpListener>),
    Udp(Weak<UdpSocket>),
}

impl WeakSocketOwner {
    pub(crate) fn upgrade(&self) -> Option<SocketLease> {
        match self {
            Self::TcpStream(socket) => socket.upgrade().map(SocketLease::TcpStream),
            Self::TcpListener(socket) => socket.upgrade().map(SocketLease::TcpListener),
            Self::Udp(socket) => socket.upgrade().map(SocketLease::Udp),
        }
    }
}

impl From<&Arc<TcpStream>> for SocketLease {
    fn from(socket: &Arc<TcpStream>) -> Self {
        Self::TcpStream(Arc::clone(socket))
    }
}

impl From<&Arc<TcpListener>> for SocketLease {
    fn from(socket: &Arc<TcpListener>) -> Self {
        Self::TcpListener(Arc::clone(socket))
    }
}

impl From<&Arc<UdpSocket>> for SocketLease {
    fn from(socket: &Arc<UdpSocket>) -> Self {
        Self::Udp(Arc::clone(socket))
    }
}
