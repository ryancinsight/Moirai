//! Async networking primitives for Moirai concurrency library.
//!
//! This module provides Moirai-owned async networking facades for TCP and UDP
//! sockets without Tokio dependencies.

pub mod listener;
pub mod stream;
pub mod socket;
pub mod types;

pub use listener::TcpListener;
pub use stream::TcpStream;
pub use socket::UdpSocket;
pub use types::{
    ConnectionInfo, ConnectionPool, ConnectionStats, ServerStats, TcpServerConfig, TcpServerStats,
    UdpConfig, UdpSocketStats, UdpStats,
};

#[cfg(test)]
mod tests;
