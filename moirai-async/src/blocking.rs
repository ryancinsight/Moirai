//! Bounded pools of threads for work with no non-blocking interface.
//!
//! Hostname resolution (`getaddrinfo`) and file-system syscalls block the
//! calling thread on every platform this crate serves. Running them inside
//! `poll` would stall the executor and make the future uncancellable, so they
//! run on a [`BlockingPool`] instead: a fixed number of worker threads fed by a
//! bounded queue.
//!
//! A caller first awaits an [`Admission`], which it gets only while fewer than
//! workers + queue depth jobs are running or queued. Submission never spawns
//! past the bound and never blocks: excess callers wait asynchronously.
//! Workers start on the first admission. A spawn failure is reported only when
//! no worker runs at all, and the next admission retries the spawn.
//!
//! Dropping a [`Completion`] abandons its job. A queued [`Abandoned::Skip`]
//! job is then dropped without running. An [`Abandoned::Run`] job, whose side
//! effect must happen once submitted, runs regardless. A running job cannot be
//! interrupted and holds its worker until it returns. Either way the admission
//! slot returns to the pool when the worker is done with the job.
//!
//! In unwind builds, a panic inside a job fails only that job. That includes a
//! panic in a caller's waker, which the reply runs inline. The worker keeps
//! serving and the slot returns through the unwind. Under `panic = "abort"`,
//! which the workspace release profile sets, such a panic ends the process.

use std::future::Future;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use moirai_pal::thread::ThreadStartError;

use crate::sync::{Semaphore, SemaphorePermit, oneshot};

#[cfg(test)]
pub(crate) mod job_lifecycle;
#[cfg(test)]
pub(crate) mod test_hooks;

/// What a worker does with a queued job whose [`Completion`] was dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Abandoned {
    /// Drop the job without running it: its only effect is its result.
    Skip,
    /// Run the job anyway: its side effect must happen once submitted.
    Run,
}

/// A type-erased job. One queue serves callers of every result type, which a
/// generic parameter cannot express (dyn exception: type erasure unknown at
/// compile time, off the hot path).
type Job = Box<dyn FnOnce() + Send>;

/// A fixed set of worker threads fed by a bounded queue.
pub(crate) struct BlockingPool {
    /// Thread-name stem, also the pool's name in errors.
    name: &'static str,
    workers_bound: usize,
    #[cfg(test)]
    admissions: usize,
    admission: Semaphore,
    jobs: SyncSender<Job>,
    queue: Arc<Mutex<Receiver<Job>>>,
    /// Workers started so far. A failed spawn is retried by the next
    /// admission, so one failure never disables the pool.
    workers: Mutex<usize>,
    #[cfg(test)]
    hooks: test_hooks::Hooks,
}

impl BlockingPool {
    /// A pool of at most `workers` threads, with `queue_depth` further
    /// admitted jobs waiting behind them.
    pub(crate) fn new(name: &'static str, workers: usize, queue_depth: usize) -> Self {
        let admissions = workers + queue_depth;
        // One slot per admission permit: an admitted job sits in the channel
        // until a worker dequeues it, so all of them may be there at once
        // before any worker wakes.
        let (jobs, queue) = mpsc::sync_channel(admissions);
        Self {
            name,
            workers_bound: workers,
            #[cfg(test)]
            admissions,
            admission: Semaphore::new(admissions),
            jobs,
            queue: Arc::new(Mutex::new(queue)),
            workers: Mutex::new(0),
            #[cfg(test)]
            hooks: test_hooks::Hooks::new(),
        }
    }

    /// Wait for a slot, starting workers first if any are missing.
    ///
    /// # Errors
    /// Returns [`ThreadStartError`] when no worker is running and none can be
    /// started.
    pub(crate) async fn admit(&'static self) -> io::Result<Admission> {
        self.ensure_workers()?;
        let permit = self.admission.acquire().await;
        Ok(Admission { pool: self, permit })
    }

    /// Admit, submit `work`, and await its result.
    ///
    /// # Errors
    /// Returns the admission error, or an error when the job panicked.
    pub(crate) async fn run<T, F>(&'static self, abandoned: Abandoned, work: F) -> io::Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        self.admit().await?.submit(abandoned, work)?.await
    }

    fn ensure_workers(&'static self) -> io::Result<()> {
        let mut workers = self.workers.lock().unwrap_or_else(PoisonError::into_inner);
        while *workers < self.workers_bound {
            let spawned = std::thread::Builder::new()
                .name(format!("{}-{}", self.name, *workers))
                .spawn(move || self.serve());
            match spawned {
                Ok(_) => *workers += 1,
                Err(source) if *workers == 0 => {
                    return Err(ThreadStartError::new(self.name, source).into());
                }
                Err(_) => break,
            }
        }
        Ok(())
    }

    fn serve(&self) {
        loop {
            // Holding the lock across `recv` makes idle workers queue on the
            // mutex; exactly one waits in the channel at a time.
            let next = self
                .queue
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .recv();
            let Ok(job) = next else {
                // The pool owns its sender for the process lifetime; a closed
                // channel means the process is tearing down.
                return;
            };
            // A panic in one job is contained here, so a worker never exits
            // and the pool never shrinks. The unwind drops the job's reply
            // sender, which fails that job, and its admission permit, which
            // returns to the pool. The panic hook has already reported it.
            let _contained = catch_unwind(AssertUnwindSafe(job));
            #[cfg(test)]
            self.hooks.disposed();
        }
    }
}

/// A claimed slot in a [`BlockingPool`], spent by [`Admission::submit`].
pub(crate) struct Admission {
    pool: &'static BlockingPool,
    permit: SemaphorePermit<'static>,
}

impl Admission {
    /// Queue `work` on a worker. The slot returns to the pool when the worker
    /// is done with the job.
    ///
    /// # Errors
    /// Returns an error when the queue is closed, which happens only while the
    /// process tears down.
    pub(crate) fn submit<T, F>(self, abandoned: Abandoned, work: F) -> io::Result<Completion<T>>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let Self { pool, permit } = self;
        let (reply, receiver) = oneshot::channel();
        let job: Job = Box::new(move || {
            if abandoned == Abandoned::Run || !reply.is_closed() {
                #[cfg(test)]
                let _running = pool.hooks.enter();
                let result = work();
                // `send` returns the result only when the completion was
                // dropped after the job started; that caller was its sole
                // consumer.
                drop(reply.send(result));
            }
            drop(permit);
        });
        match pool.jobs.try_send(job) {
            Ok(()) => Ok(Completion {
                pool: pool.name,
                receiver,
            }),
            Err(TrySendError::Full(_)) => unreachable!(
                "invariant: the channel holds one slot per admission permit, so an \
                 admitted job always finds a queue slot"
            ),
            Err(TrySendError::Disconnected(_)) => Err(io::Error::other(format!(
                "the {} queue closed before the job was submitted",
                pool.name
            ))),
        }
    }
}

/// The pending result of a submitted job.
pub(crate) struct Completion<T> {
    pool: &'static str,
    receiver: oneshot::Receiver<T>,
}

impl<T> Future for Completion<T> {
    type Output = io::Result<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let pool = self.pool;
        self.receiver.poll_recv(cx).map(|received| {
            received
                .map_err(|()| io::Error::other(format!("a {pool} job panicked before replying")))
        })
    }
}

#[cfg(test)]
impl BlockingPool {
    pub(crate) fn hooks(&self) -> &test_hooks::Hooks {
        &self.hooks
    }

    /// Admission slots, running plus queued.
    pub(crate) fn admissions(&self) -> usize {
        self.admissions
    }

    /// The configured worker bound.
    pub(crate) fn workers_bound(&self) -> usize {
        self.workers_bound
    }

    /// Worker threads started.
    pub(crate) fn workers(&self) -> usize {
        *self.workers.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Admission slots currently free.
    pub(crate) fn free_admissions(&self) -> usize {
        self.admission.available_permits()
    }
}
