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
//! The re-probe runs on every Windows build, not only below 2004. A path gated
//! on the build number would ship code that no CI host runs: every hosted
//! runner is on 2004 or later. The ungated path is the one the tests exercise.
//! On fixed builds it is also cheap. The thread starts at the first connect
//! that stays pending past its first poll, and it sleeps on a condition
//! variable while no connect is registered. Each pending connect costs ten
//! zero-timeout `select` calls per second.
//!
//! A registration lives exactly as long as its connect future. An executor
//! that leaks a pending task instead of dropping it therefore keeps that
//! task's registration, and the task is woken every interval for the life of
//! the process. `Waker` gives no signal that a task is gone, so the re-probe
//! cannot detect this; dropping tasks on executor shutdown avoids it.
//!
//! Unix needs no counterpart: epoll and kqueue report a failed connect as
//! error or hangup readiness, and the reactor consumes those as writable.

use std::io;
use std::mem;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::task::Waker;
use std::time::{Duration, Instant};

use crate::thread::ThreadStartError;

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
                .map_err(|source| ThreadStartError::new("connect re-probe", source))?;
            registry.started = true;
        }
        if let Some(key) = self.key
            && let Some((_, current)) = registry.wakers.iter_mut().find(|(k, _)| *k == key)
        {
            if !current.will_wake(waker) {
                let replaced = mem::replace(current, waker.clone());
                // Dropped after the lock is released; see `Drop`.
                drop(registry);
                drop(replaced);
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
        let Some(key) = self.key else {
            return;
        };
        let removed = {
            let mut registry = registry();
            let position = registry.wakers.iter().position(|(k, _)| *k == key);
            position.map(|index| registry.wakers.swap_remove(index))
        };
        // A waker's drop can run arbitrary code, including another
        // registration's drop; running it under the registry lock would
        // deadlock that re-entry.
        drop(removed);
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
        // One panicking waker, in its clone or its wake, must not end
        // re-probing for every other pending connect. A registration whose
        // clone panics is skipped this tick; the panic hook has already
        // reported it. The unwind stops inside the lock scope, so the guard is
        // not dropped while panicking and the registry is not poisoned.
        let wakers: Vec<Waker> = registry
            .wakers
            .iter()
            .filter_map(|(_, waker)| catch_unwind(AssertUnwindSafe(|| waker.clone())).ok())
            .collect();
        // Wake outside the lock: a wake may poll the connect inline, and that
        // poll re-arms its registration.
        drop(registry);
        for waker in wakers {
            let _contained = catch_unwind(AssertUnwindSafe(|| waker.wake()));
        }
        #[cfg(test)]
        ticks::completed();
        registry = self::registry();
    }
}

/// Connects currently registered for re-probe wakes.
#[cfg(test)]
pub(super) fn registered() -> usize {
    registry().wakers.len()
}

/// Test-only count of completed re-probe ticks.
#[cfg(test)]
pub(super) mod ticks {
    use std::sync::{Condvar, Mutex, PoisonError};
    use std::time::Duration;

    static COMPLETED: Mutex<u64> = Mutex::new(0);
    static ADVANCED: Condvar = Condvar::new();

    pub(in crate::net::connect) fn completed() {
        *COMPLETED.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        ADVANCED.notify_all();
    }

    /// Ticks completed so far.
    pub(in crate::net::connect) fn count() -> u64 {
        *COMPLETED.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Wait until at least `target` ticks have completed, or `limit` passes;
    /// returns the count observed last.
    pub(in crate::net::connect) fn wait_for(target: u64, limit: Duration) -> u64 {
        let completed = COMPLETED.lock().unwrap_or_else(PoisonError::into_inner);
        *ADVANCED
            .wait_timeout_while(completed, limit, |completed| *completed < target)
            .unwrap_or_else(PoisonError::into_inner)
            .0
    }
}
