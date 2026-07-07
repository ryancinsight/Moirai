//! Blocking length-prefixed network transport.

use crate::{Address, RemoteAddress, Transport, TransportError, TransportResult};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    thread,
    time::Duration,
};

const NETWORK_LENGTH_PREFIX_BYTES: usize = core::mem::size_of::<u64>();
const MAX_NETWORK_MESSAGE_BYTES: u64 = 16 * 1024 * 1024;
const NETWORK_CONNECT_ATTEMPTS: usize = 64;
/// Bound on a single blocking read/write so a peer that connects then stalls
/// mid-frame cannot pin the calling thread indefinitely. A timeout surfaces as
/// `TransportError::Closed` rather than an unbounded hang.
pub(crate) const NETWORK_IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Network transport for distributed communication.
pub struct NetworkTransport {}

impl Transport for NetworkTransport {
    fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()> {
        match target {
            Address::Remote(address) => write_network_frame(address, &data),
            Address::Local(_) => Err(TransportError::Closed),
        }
    }

    fn recv(&self, source: &Address) -> TransportResult<Vec<u8>> {
        match source {
            Address::Remote(address) => read_network_frame(address),
            Address::Local(_) => Err(TransportError::Closed),
        }
    }

    fn supports(&self, address: &Address) -> bool {
        matches!(address, Address::Remote(_))
    }
}

/// TCP transport for reliable network communication.
#[cfg(feature = "network")]
pub struct TcpTransport {
    network: NetworkTransport,
}

#[cfg(feature = "network")]
impl TcpTransport {
    /// Construct a TCP transport.
    pub fn new() -> Self {
        Self {
            network: NetworkTransport {},
        }
    }
}

#[cfg(feature = "network")]
impl Transport for TcpTransport {
    fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()> {
        self.network.send(target, data)
    }

    fn recv(&self, source: &Address) -> TransportResult<Vec<u8>> {
        self.network.recv(source)
    }

    fn supports(&self, address: &Address) -> bool {
        matches!(address, Address::Remote(_))
    }
}

fn write_network_frame(address: &RemoteAddress, data: &[u8]) -> TransportResult<()> {
    let length = u64::try_from(data.len()).map_err(|_| TransportError::Closed)?;
    if length > MAX_NETWORK_MESSAGE_BYTES {
        return Err(TransportError::Full);
    }

    let mut stream = connect_network_stream(address)?;
    write_network_frame_to_stream(&mut stream, data)
}

fn read_network_frame(address: &RemoteAddress) -> TransportResult<Vec<u8>> {
    let listener =
        TcpListener::bind(socket_address(address)).map_err(|_| TransportError::Closed)?;
    let (mut stream, _) = listener.accept().map_err(|_| TransportError::Closed)?;
    // Bound the frame read: a peer that connects then stalls must not hang the
    // thread forever in `read_exact`.
    stream
        .set_read_timeout(Some(NETWORK_IO_TIMEOUT))
        .map_err(|_| TransportError::Closed)?;
    read_network_frame_from_stream(&mut stream)
}

pub(crate) fn write_network_frame_to_stream(
    stream: &mut impl Write,
    data: &[u8],
) -> TransportResult<()> {
    let length = u64::try_from(data.len()).map_err(|_| TransportError::Closed)?;
    if length > MAX_NETWORK_MESSAGE_BYTES {
        return Err(TransportError::Full);
    }

    stream
        .write_all(&length.to_le_bytes())
        .and_then(|_| stream.write_all(data))
        .map_err(|_| TransportError::Closed)
}

pub(crate) fn read_network_frame_from_stream(stream: &mut impl Read) -> TransportResult<Vec<u8>> {
    let mut length_bytes = [0u8; NETWORK_LENGTH_PREFIX_BYTES];
    stream
        .read_exact(&mut length_bytes)
        .map_err(|_| TransportError::Closed)?;

    let length = u64::from_le_bytes(length_bytes);
    if length > MAX_NETWORK_MESSAGE_BYTES {
        return Err(TransportError::Full);
    }

    let mut data = vec![0u8; length as usize];
    stream
        .read_exact(&mut data)
        .map_err(|_| TransportError::Closed)?;
    Ok(data)
}

fn socket_address(address: &RemoteAddress) -> String {
    format!("{}:{}", address.host, address.port)
}

fn connect_network_stream(address: &RemoteAddress) -> TransportResult<TcpStream> {
    let socket = socket_address(address);
    for attempt in 0..NETWORK_CONNECT_ATTEMPTS {
        match TcpStream::connect(&socket) {
            Ok(stream) => {
                // Bound writes so a peer whose receive buffer fills (and never
                // drains) cannot block the sender indefinitely.
                stream
                    .set_write_timeout(Some(NETWORK_IO_TIMEOUT))
                    .map_err(|_| TransportError::Closed)?;
                return Ok(stream);
            }
            Err(_) if attempt + 1 < NETWORK_CONNECT_ATTEMPTS => {
                thread::sleep(Duration::from_millis(1));
            }
            Err(_) => return Err(TransportError::Closed),
        }
    }

    Err(TransportError::Closed)
}
