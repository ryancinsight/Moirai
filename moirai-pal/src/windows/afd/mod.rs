//! Socket readiness through `IOCTL_AFD_POLL` requests completed on one I/O
//! completion port (ADR 0067).
//!
//! One completion port receives the completions of every armed poll. AFD device
//! handles are bound to the port once; sockets are never associated with it.
//! Each armed poll is a record at a fixed address in a preallocated slot table,
//! reached from its completion packet by pointer arithmetic, so dequeuing and
//! dispatching allocate nothing.

mod abi;
mod completion_port;
mod device;
mod port;
mod slots;

#[cfg(test)]
mod tests;

pub use port::AfdPort;
pub use slots::Token;
