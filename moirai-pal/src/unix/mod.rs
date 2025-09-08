//! Unix platform implementations for async I/O.

#[cfg(target_os = "linux")]
pub mod epoll;

#[cfg(any(
    target_os = "macos",
    target_os = "freebsd", 
    target_os = "openbsd",
    target_os = "netbsd"
))]
pub mod kqueue;