//! Platform Abstraction Layer (PAL) for Moirai async I/O operations.
//!
//! This module provides platform-specific implementations of async I/O primitives
//! that enable true non-blocking operations without external runtime dependencies.
//! 
//! ## Architecture
//! 
//! The PAL provides a unified interface across platforms while leveraging
//! optimal platform-specific async I/O mechanisms:
//! 
//! - **Linux**: epoll-based event notification
//! - **macOS/BSD**: kqueue-based event notification  
//! - **Windows**: IOCP (I/O Completion Ports)
//! - **WebAssembly**: Web APIs with JavaScript interop
//!
//! ## Design Principles
//!
//! - **Zero External Dependencies**: Only platform system libraries
//! - **Zero-Copy Operations**: Direct buffer management
//! - **Sub-microsecond Latency**: Optimized for performance-critical applications
//! - **Memory Efficiency**: Minimal per-operation overhead
//! - **Cross-Platform Consistency**: Uniform behavior across all targets

pub mod reactor;
pub mod fs;
pub mod net;
pub mod timer;

#[cfg(unix)]
pub mod unix;

#[cfg(windows)] 
pub mod windows;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

use std::io;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Platform-agnostic async I/O operation handle.
pub trait AsyncOperation {
    type Output;
    
    /// Poll the operation for completion.
    fn poll_operation(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<Self::Output>>;
}

/// Platform-specific reactor interface.
pub trait Reactor: Send + Sync + 'static {
    /// Register a file descriptor/handle for async operations.
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()>;
    
    /// Unregister a file descriptor/handle.
    fn unregister_fd(&self, fd: RawFd) -> io::Result<()>;
    
    /// Poll for ready events with timeout.
    fn poll_events(&self, timeout: Option<std::time::Duration>) -> io::Result<Vec<Event>>;
    
    /// Wake up the reactor from blocking poll.
    fn wake(&self) -> io::Result<()>;
}

/// Platform-agnostic file descriptor/handle type.
#[cfg(unix)]
pub type RawFd = std::os::unix::io::RawFd;

#[cfg(windows)]
pub type RawFd = std::os::windows::io::RawHandle;

#[cfg(target_arch = "wasm32")]
pub type RawFd = u32; // Placeholder for WASM

/// I/O event interest specification.
#[derive(Debug, Clone, Copy)]
pub struct Interest {
    pub readable: bool,
    pub writable: bool,
    pub error: bool,
}

impl Interest {
    pub const READABLE: Self = Self { readable: true, writable: false, error: true };
    pub const WRITABLE: Self = Self { readable: false, writable: true, error: true };
    pub const READ_WRITE: Self = Self { readable: true, writable: true, error: true };
}

/// I/O event notification.
#[derive(Debug, Clone)]
pub struct Event {
    pub fd: RawFd,
    pub readable: bool,
    pub writable: bool,
    pub error: bool,
    pub hangup: bool,
}

/// Platform-specific reactor factory.
pub fn create_reactor() -> io::Result<Box<dyn Reactor>> {
    #[cfg(target_os = "linux")]
    return Ok(Box::new(unix::epoll::EpollReactor::new()?));
    
    #[cfg(any(target_os = "macos", target_os = "freebsd", target_os = "openbsd", target_os = "netbsd"))]
    return Ok(Box::new(unix::kqueue::KqueueReactor::new()?));
    
    #[cfg(windows)]
    return Ok(Box::new(windows::iocp::IocpReactor::new()?));
    
    #[cfg(target_arch = "wasm32")]
    return Ok(Box::new(wasm::WebReactor::new()?));
    
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos", 
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        windows,
        target_arch = "wasm32"
    )))]
    return Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "Platform not supported for native async I/O"
    ));
}

/// Async operation result future.
pub struct AsyncResult<T> {
    result: Option<io::Result<T>>,
}

impl<T> AsyncResult<T> {
    pub fn ready(result: io::Result<T>) -> Self {
        Self { result: Some(result) }
    }
    
    pub fn pending() -> Self {
        Self { result: None }
    }
}

impl<T: std::marker::Unpin> Future for AsyncResult<T> {
    type Output = io::Result<T>;
    
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.result.take() {
            Some(result) => Poll::Ready(result),
            None => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_interest_flags() {
        let read_only = Interest::READABLE;
        assert!(read_only.readable);
        assert!(!read_only.writable);
        assert!(read_only.error);
        
        let write_only = Interest::WRITABLE;
        assert!(!write_only.readable);
        assert!(write_only.writable);
        assert!(write_only.error);
        
        let read_write = Interest::READ_WRITE;
        assert!(read_write.readable);
        assert!(read_write.writable);
        assert!(read_write.error);
    }
    
    #[test]
    fn test_async_result() {
        let _ready_result = AsyncResult::ready(Ok(42));
        let _pending_result: AsyncResult<i32> = AsyncResult::pending();
    }
}