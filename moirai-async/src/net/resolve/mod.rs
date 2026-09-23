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
//! Worker threads start on the first hostname lookup. A spawn failure is
//! reported as the lookup's error only when no worker runs at all, and the
//! next lookup retries the spawn.
//!
//! Dropping the awaiting future closes the reply channel. A lookup that is
//! still queued is then skipped without calling `getaddrinfo`. A lookup that is
//! already running cannot be interrupted and occupies its worker until
//! `getaddrinfo` returns, after which the worker and the admission permit
//! return to the pool.

use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use moirai_pal::thread::ThreadStartError;

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

/// Lookups admitted at once, running or queued.
pub(super) const RESOLVER_ADMISSIONS: usize = RESOLVER_WORKERS + RESOLVER_QUEUE_DEPTH;

type LookupResult = io::Result<Vec<SocketAddr>>;

/// One admitted lookup. The permit returns to the pool when the lookup
/// finishes or is skipped, which bounds running plus queued lookups.
struct Lookup {
    query: String,
    reply: oneshot::Sender<LookupResult>,
    admission: SemaphorePermit<'static>,
}

struct Resolver {
    admission: Semaphore,
    jobs: SyncSender<Lookup>,
    queue: Arc<Mutex<Receiver<Lookup>>>,
    /// Workers started so far. A failed spawn is retried by the next lookup,
    /// so one failure never disables resolution for the process.
    workers: Mutex<usize>,
}

fn resolver() -> &'static Resolver {
    static RESOLVER: OnceLock<Resolver> = OnceLock::new();
    RESOLVER.get_or_init(|| {
        // One slot per admission permit: an admitted lookup sits in the
        // channel until a worker dequeues it, so all of them may be there at
        // once before any worker wakes.
        let (jobs, queue) = mpsc::sync_channel(RESOLVER_ADMISSIONS);
        Resolver {
            admission: Semaphore::new(RESOLVER_ADMISSIONS),
            jobs,
            queue: Arc::new(Mutex::new(queue)),
            workers: Mutex::new(0),
        }
    })
}

impl Resolver {
    /// Start any of the [`RESOLVER_WORKERS`] threads not yet running.
    ///
    /// # Errors
    /// Returns the spawn error only when no worker is running at all. With at
    /// least one worker, lookups still progress, and the missing workers are
    /// retried on the next call.
    fn ensure_workers(&self) -> io::Result<()> {
        let mut workers = self.workers.lock().unwrap_or_else(PoisonError::into_inner);
        while *workers < RESOLVER_WORKERS {
            let queue = Arc::clone(&self.queue);
            let spawned = std::thread::Builder::new()
                .name(format!("moirai-resolve-{}", *workers))
                .spawn(move || run_worker(&queue));
            match spawned {
                Ok(_) => *workers += 1,
                Err(source) if *workers == 0 => {
                    return Err(ThreadStartError::new("resolver worker", source).into());
                }
                Err(_) => break,
            }
        }
        Ok(())
    }
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
        // A panic in one job is contained here, so a worker never exits and
        // the pool never shrinks below the workers started. The panic can come
        // from `getaddrinfo` or from a caller's waker, which `send` runs
        // inline. The unwind drops the job's reply sender, which fails that
        // lookup, and its admission permit, which returns to the pool. The
        // panic hook has already reported the panic.
        let _contained = catch_unwind(AssertUnwindSafe(|| run_lookup(lookup)));
        #[cfg(test)]
        test_hooks::disposed();
    }
}

fn run_lookup(
    Lookup {
        query,
        reply,
        admission,
    }: Lookup,
) {
    // A caller that dropped its future while the lookup was queued gets no
    // `getaddrinfo` call.
    if !reply.is_closed() {
        #[cfg(test)]
        let _running = test_hooks::Running::enter();
        let result = query
            .to_socket_addrs()
            .map(Iterator::collect::<Vec<SocketAddr>>);
        // `send` returns the result only when the awaiting future was dropped
        // after the lookup started; that cancelled caller was its sole
        // consumer.
        drop(reply.send(result));
    }
    drop(admission);
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
    resolver.ensure_workers()?;
    let admission = resolver.admission.acquire().await;
    let (reply, mut receiver) = oneshot::channel();
    let lookup = Lookup {
        query: addr.to_owned(),
        reply,
        admission,
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
                "the resolver job for {addr:?} panicked before replying"
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

#[cfg(test)]
pub(super) mod test_hooks;
