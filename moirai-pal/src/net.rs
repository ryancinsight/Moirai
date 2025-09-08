//! Platform-agnostic async network I/O operations.

use std::io;
use std::net::SocketAddr;

/// Placeholder for platform-agnostic network operations.
/// This will be fully implemented once the core reactor is complete.
pub struct AsyncTcpStream {
    // Will contain platform-specific socket handle
}

impl AsyncTcpStream {
    pub async fn connect(_addr: SocketAddr) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Native async network I/O not yet implemented - reactor in development"
        ))
    }
    
    pub async fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Native async network I/O not yet implemented - reactor in development"
        ))
    }
    
    pub async fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Native async network I/O not yet implemented - reactor in development"
        ))
    }
}

pub struct AsyncTcpListener {
    // Will contain platform-specific listener handle
}

impl AsyncTcpListener {
    pub async fn bind(_addr: SocketAddr) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Native async network I/O not yet implemented - reactor in development"
        ))
    }
    
    pub async fn accept(&self) -> io::Result<(AsyncTcpStream, SocketAddr)> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Native async network I/O not yet implemented - reactor in development"
        ))
    }
}