//! Async networking primitives for Moirai concurrency library.
//!
//! This module provides Moirai-owned async networking facades for TCP and UDP
//! sockets without Tokio dependencies.

/// Async TCP listener facade.
pub mod listener;
mod resolve;
/// Async UDP socket facade.
pub mod socket;
/// Async TCP stream facade.
pub mod stream;
pub mod types;

pub use listener::TcpListener;
pub use moirai_pal::net::{MulticastConfig, UdpBufferConfig, UdpBufferSizes, UdpConfig};
pub use socket::UdpSocket;
pub use stream::TcpStream;
pub use types::{
    ConnectionInfo, ConnectionPool, ConnectionStats, ServerStats, TcpServerConfig, TcpServerStats,
    UdpSocketStats, UdpStats,
};

#[cfg(test)]
mod tests;
