//! Test-only pool instrumentation. A gate holds workers inside a job, so a
//! test can pin every stage of a job's life. Counters record jobs run and
//! jobs disposed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// Upper wait for a pool stage to be reached. Every job the tests gate
/// finishes in microseconds once released; the bound only catches a hang.
pub(crate) const STAGE_LIMIT: Duration = Duration::from_secs(10);

/// A snapshot of a pool's progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Progress {
    /// Jobs running right now.
    pub(crate) live: usize,
    /// Jobs that started running so far.
    pub(crate) started: usize,
    /// Jobs a worker finished with, run or skipped, after releasing the
    /// job's admission slot.
    pub(crate) disposed: usize,
}

struct HookState {
    /// Jobs still to fail with an injected panic.
    panics: usize,
    closed: bool,
    live: usize,
    started: usize,
    disposed: usize,
}

impl HookState {
    fn progress(&self) -> Progress {
        Progress {
            live: self.live,
            started: self.started,
            disposed: self.disposed,
        }
    }
}

/// One pool's gate and counters.
pub(crate) struct Hooks {
    state: Mutex<HookState>,
    changed: Condvar,
    peak: AtomicUsize,
}

/// Marks one worker as inside a job for its lifetime.
pub(crate) struct Running<'a>(&'a Hooks);

impl Drop for Running<'_> {
    fn drop(&mut self) {
        self.0.state().live -= 1;
        self.0.changed.notify_all();
    }
}

impl Hooks {
    pub(super) const fn new() -> Self {
        Self {
            state: Mutex::new(HookState {
                panics: 0,
                closed: false,
                live: 0,
                started: 0,
                disposed: 0,
            }),
            changed: Condvar::new(),
            peak: AtomicUsize::new(0),
        }
    }

    fn state(&self) -> MutexGuard<'_, HookState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Enter a job: count it, panic if one is injected, and wait while the
    /// gate is closed.
    pub(super) fn enter(&self) -> Running<'_> {
        let mut state = self.state();
        if state.panics > 0 {
            state.panics -= 1;
            drop(state);
            panic!("injected blocking-pool job panic");
        }
        state.live += 1;
        state.started += 1;
        self.peak.fetch_max(state.live, Ordering::SeqCst);
        self.changed.notify_all();
        while state.closed {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        Running(self)
    }

    /// Record that a worker finished with a job and released its slot.
    pub(super) fn disposed(&self) {
        self.state().disposed += 1;
        self.changed.notify_all();
    }

    /// Make the next `count` jobs panic inside the worker.
    pub(crate) fn inject_panics(&self, count: usize) {
        self.state().panics = count;
    }

    /// Highest number of jobs observed running at once.
    pub(crate) fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    /// Hold (`true`) or release (`false`) workers once they enter a job.
    pub(crate) fn set_gate_closed(&self, closed: bool) {
        self.state().closed = closed;
        self.changed.notify_all();
    }

    /// Current progress, without waiting.
    pub(crate) fn progress(&self) -> Progress {
        self.state().progress()
    }

    /// Wait until `reached` holds for the pool's progress, or [`STAGE_LIMIT`]
    /// passes. Returns the last progress observed.
    pub(crate) fn wait_until(&self, reached: impl Fn(Progress) -> bool) -> Progress {
        let (state, _) = self
            .changed
            .wait_timeout_while(self.state(), STAGE_LIMIT, |state| {
                !reached(state.progress())
            })
            .unwrap_or_else(PoisonError::into_inner);
        state.progress()
    }
}

/// Serialize tests that close a gate or read a pool's counters. Under
/// `cargo test` they share one process, and therefore every pool.
pub(crate) fn exclusive() -> MutexGuard<'static, ()> {
    static EXCLUSIVE: Mutex<()> = Mutex::new(());
    EXCLUSIVE.lock().unwrap_or_else(PoisonError::into_inner)
}
