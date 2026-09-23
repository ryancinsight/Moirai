//! Socket-address resolution that never blocks the polling thread.
//!
//! A literal `ip:port` parses in place. Anything else needs the system
//! resolver (`getaddrinfo`), which has no non-blocking interface, so the lookup
//! runs on a short-lived dedicated thread and completes a oneshot the caller
//! awaits. Dropping the awaiting future abandons the result: the lookup itself
//! cannot be interrupted, but it holds no executor thread while it finishes.
//! One thread exists per in-flight hostname lookup; concurrency is bounded by
//! the number of lookups the caller has in flight.

use std::io;
use std::net::{SocketAddr, ToSocketAddrs};

use crate::sync::oneshot;

/// A non-empty resolution result, in resolver order.
pub(super) struct ResolvedAddrs {
    first: SocketAddr,
    rest: Vec<SocketAddr>,
}

impl ResolvedAddrs {
    /// The resolver's preferred address.
    pub(super) fn first(&self) -> SocketAddr {
        self.first
    }

    /// The preferred address and the remaining fallbacks.
    pub(super) fn into_parts(self) -> (SocketAddr, Vec<SocketAddr>) {
        (self.first, self.rest)
    }
}

/// Resolve `addr` (`host:port` or a literal socket address) to its addresses.
///
/// # Errors
/// Returns the resolver's error, `InvalidInput` when resolution yields no
/// address, and the spawn error when the lookup thread cannot start.
pub(super) async fn resolve(addr: &str) -> io::Result<ResolvedAddrs> {
    if let Ok(first) = addr.parse::<SocketAddr>() {
        return Ok(ResolvedAddrs {
            first,
            rest: Vec::new(),
        });
    }

    let (sender, mut receiver) = oneshot::channel();
    let query = addr.to_owned();
    std::thread::Builder::new()
        .name("moirai-resolve".to_owned())
        .spawn(move || {
            let resolved = query
                .to_socket_addrs()
                .map(Iterator::collect::<Vec<SocketAddr>>);
            // `send` returns the result only when the awaiting future was
            // dropped; that cancelled caller was its sole consumer.
            drop(sender.send(resolved));
        })?;

    let mut addrs = receiver
        .recv()
        .await
        .map_err(|()| {
            io::Error::other(format!(
                "resolver thread for {addr:?} exited without a result"
            ))
        })??
        .into_iter();
    let first = addrs.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{addr:?} resolved to no socket address"),
        )
    })?;
    Ok(ResolvedAddrs {
        first,
        rest: addrs.collect(),
    })
}
