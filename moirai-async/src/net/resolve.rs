//! Socket-address resolution that never blocks the polling thread.
//!
//! A literal `ip:port` parses in place. Anything else needs the system
//! resolver (`getaddrinfo`), which has no non-blocking interface, so the lookup
//! runs on the resolver's [`BlockingPool`]: at most [`RESOLVER_WORKERS`]
//! threads for the life of the process, with [`RESOLVER_QUEUE_DEPTH`] further
//! lookups queued and every other caller waiting asynchronously. A lookup is an
//! [`Abandoned::Skip`] job: dropping the future while the lookup is queued
//! skips `getaddrinfo`. Admission, cancellation, and panic containment are the
//! pool's (see [`crate::blocking`]).

use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::OnceLock;

use crate::blocking::{Abandoned, BlockingPool};

/// Resolver threads for the whole process.
///
/// A lookup costs latency rather than CPU. It waits on the network for up to
/// the resolver's timeout: glibc's default is 5 s per attempt, times 2
/// attempts, per configured nameserver. So the worker count sets how many
/// stuck names can stall at once before the rest queue, not how fast lookups
/// run. Connect-path resolution happens at connection setup, far below one
/// lookup per worker per second in the callers this serves, so four workers
/// leave three free while one name hangs. The count is also the hard ceiling
/// on resolver threads.
pub(super) const RESOLVER_WORKERS: usize = 4;

/// Admitted lookups that may wait in the queue behind the running ones.
///
/// One per worker lets each worker take its next job without a round trip
/// through admission. Deeper queueing only moves waiting from the async
/// admission wait, which costs no thread, into the channel.
pub(super) const RESOLVER_QUEUE_DEPTH: usize = RESOLVER_WORKERS;

/// The process-wide resolver pool.
pub(super) fn pool() -> &'static BlockingPool {
    static POOL: OnceLock<BlockingPool> = OnceLock::new();
    POOL.get_or_init(|| BlockingPool::new("moirai-resolve", RESOLVER_WORKERS, RESOLVER_QUEUE_DEPTH))
}

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
/// address, and the pool's admission or job-panic error.
pub(super) async fn resolve(addr: &str) -> io::Result<ResolvedAddrs> {
    if let Ok(first) = addr.parse::<SocketAddr>() {
        return Ok(ResolvedAddrs {
            first,
            rest: Vec::new(),
        });
    }

    let query = addr.to_owned();
    let mut addrs = pool()
        .run(Abandoned::Skip, move || {
            query
                .to_socket_addrs()
                .map(Iterator::collect::<Vec<SocketAddr>>)
        })
        .await??
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
