use std::fmt;
use std::future::Future;
use std::io;
use std::time::Duration;

pub(crate) const CANCELLED: quinn::VarInt = quinn::VarInt::from_u32(1);
pub(crate) const PROTOCOL_REJECTED: quinn::VarInt = quinn::VarInt::from_u32(2);

/// Distinct transport failures retain the upstream error as their source.
#[derive(Debug)]
#[non_exhaustive]
pub enum QuicError {
    /// The completed TLS handshake lacks the required application protocol.
    ApplicationProtocol,
    /// A Moirai spawner rejected a required QUIC protocol driver.
    Executor(std::sync::Arc<moirai_core::error::ExecutorError>),
    /// Invalid protocol, admission or deadline configuration.
    InvalidConfig(&'static str),
    /// Native socket or allocation error.
    Io(io::Error),
    /// Certificate, signing key or trust-store configuration error.
    Tls(super::rustls::Error),
    /// Connection initiation error.
    Connect(quinn::ConnectError),
    /// Authenticated connection failure, including TLS rejection.
    Connection(quinn::ConnectionError),
    /// Stream read error, retaining reset/connection context.
    Read(quinn::ReadError),
    /// Truncated exact read or stream failure.
    ReadExact(quinn::ReadExactError),
    /// Stream write failure, retaining peer stop/connection context.
    Write(quinn::WriteError),
    /// A peer or caller exceeded the admitted message size.
    MessageTooLarge {
        /// Actual or declared message size.
        size: u64,
        /// Configured maximum message size.
        limit: u32,
    },
    /// A complete operation exceeded its finite deadline.
    TimedOut(&'static str),
    /// Endpoint connection admission is full.
    ConnectionCapacity,
    /// Live/pending application stream admission is full.
    StreamCapacity,
    /// The stream or endpoint is closed or cancelled.
    Closed,
}

impl fmt::Display for QuicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApplicationProtocol => {
                f.write_str("QUIC peer did not negotiate the required application protocol")
            }
            Self::Executor(error) => write!(f, "QUIC executor admission: {error}"),
            Self::InvalidConfig(message) => write!(f, "invalid QUIC configuration: {message}"),
            Self::Io(error) => write!(f, "QUIC I/O: {error}"),
            Self::Tls(error) => write!(f, "QUIC TLS: {error}"),
            Self::Connect(error) => write!(f, "QUIC connect: {error}"),
            Self::Connection(error) => write!(f, "QUIC connection: {error}"),
            Self::Read(error) => write!(f, "QUIC read: {error}"),
            Self::ReadExact(error) => write!(f, "QUIC frame read: {error}"),
            Self::Write(error) => write!(f, "QUIC write: {error}"),
            Self::MessageTooLarge { size, limit } => {
                write!(f, "QUIC message has {size} bytes; limit is {limit}")
            }
            Self::TimedOut(operation) => write!(f, "QUIC {operation} deadline elapsed"),
            Self::ConnectionCapacity => f.write_str("QUIC connection admission is full"),
            Self::StreamCapacity => f.write_str("QUIC application stream admission is full"),
            Self::Closed => f.write_str("QUIC stream or endpoint is closed"),
        }
    }
}

impl std::error::Error for QuicError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Executor(error) => Some(error.as_ref()),
            Self::Io(error) => Some(error),
            Self::Tls(error) => Some(error),
            Self::Connect(error) => Some(error),
            Self::Connection(error) => Some(error),
            Self::Read(error) => Some(error),
            Self::ReadExact(error) => Some(error),
            Self::Write(error) => Some(error),
            _ => None,
        }
    }
}

impl QuicError {
    /// Classify native I/O without discarding the distinct source error.
    ///
    /// Truncated frames differ from resets/connection loss even when either
    /// arrives during an exact read. Application consumers need no Quinn import.
    pub fn io_kind(&self) -> io::ErrorKind {
        match self {
            Self::Io(error) => error.kind(),
            Self::InvalidConfig(_) => io::ErrorKind::InvalidInput,
            Self::ReadExact(quinn::ReadExactError::FinishedEarly(_)) => {
                io::ErrorKind::UnexpectedEof
            }
            Self::ReadExact(quinn::ReadExactError::ReadError(error)) | Self::Read(error) => {
                io::Error::from(error.clone()).kind()
            }
            Self::Write(error) => io::Error::from(error.clone()).kind(),
            Self::Connection(_) | Self::Connect(_) => io::ErrorKind::ConnectionAborted,
            Self::MessageTooLarge { .. } => io::ErrorKind::InvalidData,
            Self::TimedOut(_) => io::ErrorKind::TimedOut,
            Self::ConnectionCapacity | Self::StreamCapacity => io::ErrorKind::WouldBlock,
            Self::Closed => io::ErrorKind::NotConnected,
            Self::ApplicationProtocol => io::ErrorKind::PermissionDenied,
            Self::Executor(_) | Self::Tls(_) => io::ErrorKind::Other,
        }
    }
}

pub(super) async fn bounded<F: Future>(
    deadline: Duration,
    operation: &'static str,
    future: F,
) -> Result<F::Output, QuicError> {
    moirai_async::timer::timeout(deadline, future)
        .await
        .map_err(|_| QuicError::TimedOut(operation))
}
