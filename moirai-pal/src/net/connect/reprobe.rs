//! Periodic re-poll for pending Windows connects.
//!
//! The reactor learns about sockets only from `WSAPoll`, and before Windows 10
//! version 2004 `WSAPoll` never reports a failed connect. A refused or
//! unreachable connect would then sit pending forever. Every pending connect
//! therefore also registers its waker here. One shared thread wakes all
//! registered wakers every [`CONNECT_REPROBE_INTERVAL`], and each woken future
//! re-runs the `select` probe, which does report failure. The thread count is
//! one per process, and the waker list holds at most one entry per distinct
//! pending task.
//!
//! Unix needs no counterpart: epoll and kqueue report a failed connect as
//! error or hangup readiness, and the reactor consumes those as writable.

use std::sync::{Condvar, Mutex, OnceLock, PoisonError};
use std::task::Waker;
use std::time::{Duration, Instant};
use std::{io, mem};

/// Re-poll period for a pending connect.
///
/// It bounds how long a failure `WSAPoll` missed can go unobserved. The
/// fastest connect failure Windows reports is a refused loopback connect after
/// its SYN retries: 2.06 to 2.48 s measured on the development host. A 100 ms
/// period adds at most 5% to that latency. Each wake costs one zero-timeout
/// `select` and a reactor re-registration per pending connect, which is ten
/// per second.
pub(super) const CONNECT_REPROBE_INTERVAL: Duration = Duration::from_millis(100);

struct Reprobe {
    pending: Mutex<Vec<Waker>>,
    registered: Condvar,
}

static REPROBE: Reprobe = Reprobe {
    pending: Mutex::new(Vec::new()),
    registered: Condvar::new(),
};

fn reprobe() -> io::Result<&'static Reprobe> {
    static STARTED: OnceLock<Result<(), io::ErrorKind>> = OnceLock::new();
    let started = *STARTED.get_or_init(|| {
        std::thread::Builder::new()
            .name("moirai-connect-reprobe".to_owned())
            .spawn(|| REPROBE.run())
            .map(drop)
            .map_err(|error| error.kind())
    });
    started
        .map_err(|kind| io::Error::new(kind, "the connect re-probe thread could not be started"))?;
    Ok(&REPROBE)
}

impl Reprobe {
    fn run(&self) {
        loop {
            let mut pending = self
                .registered
                .wait_while(
                    self.pending.lock().unwrap_or_else(PoisonError::into_inner),
                    |pending| pending.is_empty(),
                )
                .unwrap_or_else(PoisonError::into_inner);
            let due = Instant::now() + CONNECT_REPROBE_INTERVAL;
            while let Some(remaining) = due.checked_duration_since(Instant::now())
                && !remaining.is_zero()
            {
                pending = self
                    .registered
                    .wait_timeout(pending, remaining)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            let wakers = mem::take(&mut *pending);
            drop(pending);
            for waker in wakers {
                waker.wake();
            }
        }
    }
}

/// Wake `waker` within one [`CONNECT_REPROBE_INTERVAL`].
///
/// # Errors
/// Returns an error when the re-probe thread could not be started.
pub(super) fn schedule(waker: &Waker) -> io::Result<()> {
    let shared = reprobe()?;
    let mut pending = shared
        .pending
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if !pending.iter().any(|queued| queued.will_wake(waker)) {
        pending.push(waker.clone());
        shared.registered.notify_one();
    }
    Ok(())
}
