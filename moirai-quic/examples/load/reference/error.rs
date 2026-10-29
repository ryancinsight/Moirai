//! Typed failures from the independent transport oracle.

use std::fmt;

#[derive(Debug)]
pub(crate) enum Error {
    Fixture(Box<dyn std::error::Error>),
    Io(std::io::Error),
    Tls(moirai_quic::rustls::Error),
    InvalidConfig(&'static str),
    Connect(quinn::ConnectError),
    Connection(quinn::ConnectionError),
    Read(quinn::ReadError),
    ReadExact(quinn::ReadExactError),
    Write(quinn::WriteError),
    Stop(quinn::StoppedError),
    Allocation(std::collections::TryReserveError),
    Numeric(std::num::TryFromIntError),
    TimedOut(&'static str),
    MessageTooLarge { size: u64, limit: u32 },
    Closed,
    PeerIdentity,
    ApplicationProtocol,
    PayloadMismatch,
    ExtraData,
    StreamDomain,
    PeerStopped(quinn::VarInt),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixture(error) => write!(formatter, "probe identity fixture: {error}"),
            Self::Io(error) => write!(formatter, "QUIC reference I/O: {error}"),
            Self::Tls(error) => write!(formatter, "QUIC reference TLS: {error}"),
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid QUIC reference config: {message}")
            }
            Self::Connect(error) => write!(formatter, "QUIC reference connect: {error}"),
            Self::Connection(error) => write!(formatter, "QUIC reference connection: {error}"),
            Self::Read(error) => write!(formatter, "QUIC reference read: {error}"),
            Self::ReadExact(error) => write!(formatter, "QUIC reference frame read: {error}"),
            Self::Write(error) => write!(formatter, "QUIC reference write: {error}"),
            Self::Stop(error) => write!(formatter, "QUIC reference FIN acknowledgement: {error}"),
            Self::Allocation(error) => {
                write!(formatter, "QUIC reference frame allocation: {error}")
            }
            Self::Numeric(error) => write!(formatter, "QUIC reference numeric conversion: {error}"),
            Self::TimedOut(operation) => write!(formatter, "QUIC reference {operation} timed out"),
            Self::MessageTooLarge { size, limit } => {
                write!(
                    formatter,
                    "QUIC reference message has {size} bytes; limit is {limit}"
                )
            }
            Self::Closed => formatter.write_str("QUIC reference stream or endpoint is closed"),
            Self::PeerIdentity => formatter.write_str("QUIC reference peer identity differs"),
            Self::ApplicationProtocol => formatter.write_str("QUIC reference ALPN differs"),
            Self::PayloadMismatch => formatter.write_str("QUIC reference payload differs"),
            Self::ExtraData => formatter.write_str("QUIC reference peer sent extra framed data"),
            Self::StreamDomain => {
                formatter.write_str("QUIC reference stream ID is outside the client domain")
            }
            Self::PeerStopped(code) => write!(
                formatter,
                "QUIC reference peer stopped the stream with {code}"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fixture(error) => Some(error.as_ref()),
            Self::Io(error) => Some(error),
            Self::Tls(error) => Some(error),
            Self::Connect(error) => Some(error),
            Self::Connection(error) => Some(error),
            Self::Read(error) => Some(error),
            Self::ReadExact(error) => Some(error),
            Self::Write(error) => Some(error),
            Self::Stop(error) => Some(error),
            Self::Allocation(error) => Some(error),
            Self::Numeric(error) => Some(error),
            _ => None,
        }
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;
