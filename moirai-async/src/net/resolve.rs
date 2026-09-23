//! Socket-address resolution that never blocks the polling thread.
//!
//! A literal `ip:port` parses in place. Anything else needs the system
//! resolver (`getaddrinfo`), which has no non-blocking interface, so the lookup
//! runs on a fixed pool of [`RESOLVER_WORKERS`] threads fed by a bounded
//! queue. A caller first awaits an admission permit, which it gets only while
//! fewer than [`RESOLVER_WORKERS`] + [`RESOLVER_QUEUE_DEPTH`] lookups are
//! running or queued. Submission therefore never spawns a thread and never
//! blocks: excess callers wait asynchronously. At most [`RESOLVER_WORKERS`]
//! resolver threads exist for the life of the process, whatever the caller
//! does.
//!
//! Dropping the awaiting future closes the reply channel. A lookup that is
//! still queued is then skipped without calling `getaddrinfo`. A lookup that is
//! already running cannot be interrupted and occupies its worker until
//! `getaddrinfo` returns, after which the worker and the admission permit
//! return to the pool.

use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use crate::sync::{Semaphore, SemaphorePermit, oneshot};

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
/// through the admission semaphore. Deeper queueing only moves waiting from
/// the async admission wait, which costs no thread, into the channel.
pub(super) const RESOLVER_QUEUE_DEPTH: usize = RESOLVER_WORKERS;

type LookupResult = io::Result<Vec<SocketAddr>>;

/// One admitted lookup. The permit returns to the pool when the lookup
/// finishes or is skipped, which bounds running plus queued lookups.
struct Lookup {
    query: String,
    reply: oneshot::Sender<LookupResult>,
    _admission: SemaphorePermit<'static>,
}

struct Resolver {
    admission: Semaphore,
    jobs: SyncSender<Lookup>,
    workers: usize,
}

fn resolver() -> &'static Resolver {
    static RESOLVER: OnceLock<Resolver> = OnceLock::new();
    RESOLVER.get_or_init(|| {
        // Sized for every admission permit: an admitted lookup sits in the
        // channel until a worker dequeues it, so all of them may be there at
        // once before any worker wakes.
        let (jobs, queue) = mpsc::sync_channel(RESOLVER_WORKERS + RESOLVER_QUEUE_DEPTH);
        let queue = Arc::new(Mutex::new(queue));
        let workers = (0..RESOLVER_WORKERS)
            .filter(|index| {
                let queue = Arc::clone(&queue);
                std::thread::Builder::new()
                    .name(format!("moirai-resolve-{index}"))
                    .spawn(move || run_worker(&queue))
                    .is_ok()
            })
            .count();
        Resolver {
            admission: Semaphore::new(RESOLVER_QUEUE_DEPTH + workers),
            jobs,
            workers,
        }
    })
}

fn run_worker(queue: &Mutex<Receiver<Lookup>>) {
    loop {
        // Holding the lock across `recv` makes idle workers queue on the
        // mutex; exactly one waits in the channel at a time.
        let next = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
        let Ok(lookup) = next else {
            // The static sender is never dropped; a closed channel means the
            // process is tearing down.
            return;
        };
        if lookup.reply.is_closed() {
            // The caller dropped its future while this lookup was queued.
            continue;
        }
        #[cfg(test)]
        let _running = test_hooks::Running::enter();
        let result = lookup
            .query
            .to_socket_addrs()
            .map(Iterator::collect::<Vec<SocketAddr>>);
        // `send` returns the result only when the awaiting future was dropped
        // after the lookup started; that cancelled caller was its sole consumer.
        drop(lookup.reply.send(result));
    }
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
/// address, and an error when no resolver worker thread could be started.
pub(super) async fn resolve(addr: &str) -> io::Result<ResolvedAddrs> {
    if let Ok(first) = addr.parse::<SocketAddr>() {
        return Ok(ResolvedAddrs {
            first,
            rest: Vec::new(),
        });
    }

    let resolver = resolver();
    if resolver.workers == 0 {
        return Err(io::Error::other(format!(
            "no resolver worker thread could be started to resolve {addr:?}"
        )));
    }
    let admission = resolver.admission.acquire().await;
    let (reply, mut receiver) = oneshot::channel();
    let lookup = Lookup {
        query: addr.to_owned(),
        reply,
        _admission: admission,
    };
    match resolver.jobs.try_send(lookup) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => unreachable!(
            "invariant: the channel holds one slot per admission permit, so an \
             admitted lookup always finds a queue slot"
        ),
        Err(TrySendError::Disconnected(_)) => {
            return Err(io::Error::other(format!(
                "resolver workers exited before resolving {addr:?}"
            )));
        }
    }

    let mut addrs = receiver
        .recv()
        .await
        .map_err(|()| {
            io::Error::other(format!(
                "resolver worker for {addr:?} exited without a result"
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

/// Test-only instrumentation: live and peak concurrent lookups, and a gate
/// that holds workers inside a lookup so a test can observe saturation.
#[cfg(test)]
pub(super) mod test_hooks {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex, PoisonError};
    use std::time::Duration;

    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static GATE: Mutex<GateState> = Mutex::new(GateState {
        closed: false,
        live: 0,
    });
    static CHANGED: Condvar = Condvar::new();

    struct GateState {
        closed: bool,
        live: usize,
    }

    /// Marks one worker as inside a lookup for its lifetime.
    pub(in crate::net) struct Running;

    impl Running {
        pub(in crate::net) fn enter() -> Self {
            let mut gate = GATE.lock().unwrap_or_else(PoisonError::into_inner);
            gate.live += 1;
            PEAK.fetch_max(gate.live, Ordering::SeqCst);
            CHANGED.notify_all();
            while gate.closed {
                gate = CHANGED.wait(gate).unwrap_or_else(PoisonError::into_inner);
            }
            Self
        }
    }

    impl Drop for Running {
        fn drop(&mut self) {
            GATE.lock().unwrap_or_else(PoisonError::into_inner).live -= 1;
            CHANGED.notify_all();
        }
    }

    /// Highest number of lookups observed running at once.
    pub(in crate::net) fn peak() -> usize {
        PEAK.load(Ordering::SeqCst)
    }

    /// Hold (`true`) or release (`false`) workers once they enter a lookup.
    pub(in crate::net) fn set_gate_closed(closed: bool) {
        GATE.lock().unwrap_or_else(PoisonError::into_inner).closed = closed;
        CHANGED.notify_all();
    }

    /// Wait until exactly `target` lookups are running, or `limit` passes;
    /// returns the running count observed last.
    pub(in crate::net) fn wait_for_live(target: usize, limit: Duration) -> usize {
        let gate = GATE.lock().unwrap_or_else(PoisonError::into_inner);
        let (gate, _) = CHANGED
            .wait_timeout_while(gate, limit, |gate| gate.live != target)
            .unwrap_or_else(PoisonError::into_inner);
        gate.live
    }

    /// Resolver threads started in this process.
    pub(in crate::net) fn workers() -> usize {
        super::resolver().workers
    }

    /// Admission permits currently free.
    pub(in crate::net) fn free_admissions() -> usize {
        super::resolver().admission.available_permits()
    }
}
