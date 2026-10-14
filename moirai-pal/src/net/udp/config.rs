//! Applied native datagram configuration; see ADR 0081.

use socket2::{Domain, Protocol, SockRef, Socket, Type};
use std::io;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};
use std::num::NonZeroUsize;

/// Independent socket-queue requests. Absent requests retain kernel defaults.
///
/// These are native SO_RCVBUF/SO_SNDBUF values, independent of application
/// message size and protocol flow control. Kernels may round or cap requests;
/// [`UdpBufferSizes`] reports their effective values without normalization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UdpBufferConfig {
    /// Requested native receive-buffer size in bytes.
    pub receive: Option<NonZeroUsize>,
    /// Requested native send-buffer size in bytes.
    pub send: Option<NonZeroUsize>,
}

impl UdpBufferConfig {
    /// Validate the signed native socket-option representation before casting.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a request above `i32::MAX` bytes.
    pub fn validate(&self) -> io::Result<()> {
        for size in [self.receive, self.send].into_iter().flatten() {
            i32::try_from(size.get()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "UDP buffer exceeds native signed integer",
                )
            })?;
        }
        Ok(())
    }

    fn apply(&self, socket: &Socket) -> io::Result<()> {
        self.validate()?;
        if let Some(size) = self.receive {
            socket.set_recv_buffer_size(size.get())?;
        }
        if let Some(size) = self.send {
            socket.set_send_buffer_size(size.get())?;
        }
        Ok(())
    }
}

/// Effective native socket-queue values, including kernel accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdpBufferSizes {
    /// Value returned by SO_RCVBUF in bytes.
    pub receive: usize,
    /// Value returned by SO_SNDBUF in bytes.
    pub send: usize,
}

impl UdpBufferSizes {
    /// Read both effective native options without modifying the socket.
    ///
    /// # Errors
    /// Propagates a native socket-option read error.
    pub fn read(socket: &UdpSocket) -> io::Result<Self> {
        let socket = SockRef::from(socket);
        Ok(Self {
            receive: socket.recv_buffer_size()?,
            send: socket.send_buffer_size()?,
        })
    }
}

/// One multicast group and its receive/outgoing interface.
///
/// A socket joins the group and selects the same interface for outbound
/// multicast. Membership lasts until socket closure. This supplies neither
/// authenticated peer identity nor reliable delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulticastConfig {
    /// IPv4 group and local interface address; unspecified chooses the OS route.
    V4 {
        /// Address in the IPv4 multicast range.
        group: Ipv4Addr,
        /// Address of the local multicast interface.
        interface: Ipv4Addr,
    },
    /// IPv6 group and local interface index; zero chooses the OS interface.
    V6 {
        /// Address in the IPv6 multicast range.
        group: Ipv6Addr,
        /// Native local interface index.
        interface: u32,
    },
}

impl MulticastConfig {
    fn validate(self, address: SocketAddr) -> io::Result<()> {
        let valid = match self {
            Self::V4 { group, interface } => {
                address.is_ipv4()
                    && group.is_multicast()
                    && !interface.is_multicast()
                    && !interface.is_broadcast()
            }
            Self::V6 { group, .. } => address.is_ipv6() && group.is_multicast(),
        };
        if !valid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "UDP multicast requires a multicast group, local interface and matching socket family",
            ));
        }
        Ok(())
    }

    fn apply(self, socket: &Socket) -> io::Result<()> {
        match self {
            Self::V4 { group, interface } => {
                socket.join_multicast_v4(&group, &interface)?;
                socket.set_multicast_if_v4(&interface)
            }
            Self::V6 { group, interface } => {
                socket.join_multicast_v6(&group, interface)?;
                socket.set_multicast_if_v6(interface)
            }
        }
    }
}

/// PAL-owned UDP options, applied before the bound socket is published.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UdpConfig {
    /// Independent native queue requests; default retains kernel settings.
    pub buffers: UdpBufferConfig,
    /// Enable IPv4 broadcast through SO_BROADCAST.
    pub broadcast: bool,
    /// Group membership and outgoing multicast interface, when requested.
    pub multicast: Option<MulticastConfig>,
}

impl UdpConfig {
    /// Create and configure a nonblocking native UDP socket.
    ///
    /// # Errors
    /// Rejects out-of-range buffers, IPv6 broadcast, invalid multicast groups
    /// and group/socket-family mismatch. Propagates native bind/option errors.
    /// Any failure drops the unpublished socket and its group membership.
    pub fn bind(&self, address: SocketAddr) -> io::Result<UdpSocket> {
        self.buffers.validate()?;
        if self.broadcast && address.is_ipv6() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "IPv6 has no broadcast",
            ));
        }
        if let Some(multicast) = self.multicast {
            multicast.validate(address)?;
        }
        let socket = Socket::new(
            Domain::for_address(address),
            Type::DGRAM,
            Some(Protocol::UDP),
        )?;
        self.buffers.apply(&socket)?;
        if self.broadcast {
            socket.set_broadcast(true)?;
        }
        socket.set_nonblocking(true)?;
        socket.bind(&address.into())?;
        if let Some(multicast) = self.multicast {
            multicast.apply(&socket)?;
        }
        Ok(socket.into())
    }
}

#[cfg(test)]
#[path = "config/tests.rs"]
mod tests;
