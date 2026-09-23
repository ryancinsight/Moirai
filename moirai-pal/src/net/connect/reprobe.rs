//! Periodic re-poll for pending Windows connects.
//!
//! The reactor learns about sockets only from `WSAPoll`, and before Windows 10
//! version 2004 `WSAPoll` never reports a failed connect. A refused or
//! unreachable connect would then sit pending forever. Every pending connect
//! therefore holds a [`Registration`] here. One shared thread wakes each
//! registered connect every [`CONNECT_REPROBE_INTERVAL`], and the woken future
//! re-runs the `select` probe, which does report failure. A registration is
//! removed when its connect completes or its future is dropped, so a settled
//! connect receives no further wakes.
//!
//! The re-probe runs on every Windows build, not only below 2004. Telling
//! builds apart needs `RtlGetVersion`, from a WDK feature this crate does not
//! enable. On fixed builds the thread starts at the first connect that stays
//! pending past its first poll. It sleeps on a condition variable while no
//! connect is registered, and each pending connect costs ten zero-timeout
//! `select` calls per second.
//!
//! Unix needs no counterpart: epoll and kqueue report a failed connect as
//! error or hangup readiness, and the reactor consumes those as writable.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::task::Waker;
use std::time::{Duration, Instant};
use std::{fmt, io};

/// Re-poll period for a pending connect.
///
/// It bounds how long a failure `WSAPoll` missed can go unobserved. The
/// fastest connect failure Windows reports is a refused loopback connect after
/// its SYN retries: 2.06 to 2.48 s measured on the development host. A 100 ms
/// period adds at most 5% to that latency. Each wake costs one zero-timeout
/// `select` and a reactor re-registration per pending connect, which is ten
/// per second.
pub(super) const CONNECT_REPROBE_INTERVAL: Duration = Duration::from_millis(100);

struct Registry {
    /// Whether the re-probe thread is running. A failed spawn leaves this
    /// `false`, so the next registration retries it.
    started: bool,
    next_key: u64,
    wakers: Vec<(u64, Waker)>,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    started: false,
    next_key: 0,
    wakers: Vec::new(),
});
static REGISTERED: Condvar = Condvar::new();

fn registry() -> MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One pending connect's claim on periodic wakes; removed on drop.
pub(super) struct Registration {
    key: Option<u64>,
}

impl Registration {
    pub(super) const fn new() -> Self {
        Self { key: None }
    }

    /// Wake `waker` every [`CONNECT_REPROBE_INTERVAL`] until this registration
    /// drops. A later call replaces the waker.
    ///
    /// # Errors
    /// Returns [`ThreadStartError`] when the re-probe thread is not running
    /// and cannot be started; the next call retries the spawn.
    pub(super) fn arm(&mut self, waker: &Waker) -> io::Result<()> {
        let mut registry = registry();
        if !registry.started {
            std::thread::Builder::new()
                .name("moirai-connect-reprobe".to_owned())
                .spawn(run)
                .map_err(|source| ThreadStartError { source })?;
            registry.started = true;
        }
        if let Some(key) = self.key
            && let Some((_, current)) = registry.wakers.iter_mut().find(|(k, _)| *k == key)
        {
            if !current.will_wake(waker) {
                current.clone_from(waker);
            }
            return Ok(());
        }
        let key = registry.next_key;
        registry.next_key += 1;
        registry.wakers.push((key, waker.clone()));
        self.key = Some(key);
        REGISTERED.notify_one();
        Ok(())
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(key) = self.key {
            registry().wakers.retain(|(k, _)| *k != key);
        }
    }
}

fn run() {
    let mut registry = registry();
    loop {
        registry = REGISTERED
            .wait_while(registry, |registry| registry.wakers.is_empty())
            .unwrap_or_else(PoisonError::into_inner);
        let due = Instant::now() + CONNECT_REPROBE_INTERVAL;
        while let Some(remaining) = due.checked_duration_since(Instant::now())
            && !remaining.is_zero()
        {
            registry = REGISTERED
                .wait_timeout(registry, remaining)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        let wakers: Vec<Waker> = registry.wakers.iter().map(|(_, w)| w.clone()).collect();
        // Wake outside the lock: a wake may poll the connect inline, and that
        // poll re-arms its registration.
        drop(registry);
        wakers.into_iter().for_each(Waker::wake);
        registry = self::registry();
    }
}

/// The re-probe thread failed to start; `source` is the spawn error.
#[derive(Debug)]
pub(super) struct ThreadStartError {
    source: io::Error,
}

impl fmt::Display for ThreadStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("could not start the connect re-probe thread")
    }
}

impl std::error::Error for ThreadStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl From<ThreadStartError> for io::Error {
    fn from(error: ThreadStartError) -> Self {
        let kind = error.source.kind();
        Self::new(kind, error)
    }
}

/// Connects currently registered for re-probe wakes.
#[cfg(test)]
pub(super) fn registered() -> usize {
    registry().wakers.len()
}
