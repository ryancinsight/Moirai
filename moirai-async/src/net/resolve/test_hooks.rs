//! Test-only resolver instrumentation. A gate holds workers inside a lookup,
//! so a test can pin every stage of a lookup's life. Counters record
//! `getaddrinfo` calls and disposed jobs.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

static PEAK: AtomicUsize = AtomicUsize::new(0);
static STATE: Mutex<HookState> = Mutex::new(HookState {
    panics: 0,
    closed: false,
    live: 0,
    started: 0,
    disposed: 0,
});
static CHANGED: Condvar = Condvar::new();

/// Upper wait for a resolver stage to be reached. `localhost` resolves from
/// the hosts file in microseconds; the bound only catches a hang.
pub(in crate::net) const STAGE_LIMIT: Duration = Duration::from_secs(10);

/// A snapshot of the resolver's progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::net) struct Progress {
    /// Lookups inside `getaddrinfo` right now.
    pub(in crate::net) live: usize,
    /// `getaddrinfo` calls made so far.
    pub(in crate::net) started: usize,
    /// Jobs a worker finished with, run or skipped, after releasing the
    /// job's admission permit.
    pub(in crate::net) disposed: usize,
}

struct HookState {
    /// Lookups still to fail with an injected panic.
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

fn state() -> MutexGuard<'static, HookState> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Marks one worker as inside `getaddrinfo` for its lifetime.
pub(super) struct Running;

impl Running {
    pub(super) fn enter() -> Self {
        let mut state = state();
        if state.panics > 0 {
            state.panics -= 1;
            drop(state);
            panic!("injected resolver lookup panic");
        }
        state.live += 1;
        state.started += 1;
        PEAK.fetch_max(state.live, Ordering::SeqCst);
        CHANGED.notify_all();
        while state.closed {
            state = CHANGED.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        Self
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        state().live -= 1;
        CHANGED.notify_all();
    }
}

/// Record that a worker finished with a job and released its permit.
pub(super) fn disposed() {
    state().disposed += 1;
    CHANGED.notify_all();
}

/// Make the next `count` lookups panic inside the worker.
pub(in crate::net) fn inject_panics(count: usize) {
    state().panics = count;
}

/// Highest number of lookups observed running at once.
pub(in crate::net) fn peak() -> usize {
    PEAK.load(Ordering::SeqCst)
}

/// Hold (`true`) or release (`false`) workers once they enter a lookup.
pub(in crate::net) fn set_gate_closed(closed: bool) {
    state().closed = closed;
    CHANGED.notify_all();
}

/// Wait until `reached` holds for the resolver's progress, or
/// [`STAGE_LIMIT`] passes. Returns the last progress observed.
pub(in crate::net) fn wait_until(reached: impl Fn(Progress) -> bool) -> Progress {
    let (state, _) = CHANGED
        .wait_timeout_while(state(), STAGE_LIMIT, |state| !reached(state.progress()))
        .unwrap_or_else(PoisonError::into_inner);
    state.progress()
}

/// Resolver threads started in this process.
pub(in crate::net) fn workers() -> usize {
    *super::resolver()
        .workers
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Admission permits currently free.
pub(in crate::net) fn free_admissions() -> usize {
    super::resolver().admission.available_permits()
}

/// Serialize tests that close the gate or read the counters. Under
/// `cargo test` they share one process, and therefore one resolver.
pub(in crate::net) fn exclusive() -> MutexGuard<'static, ()> {
    static EXCLUSIVE: Mutex<()> = Mutex::new(());
    EXCLUSIVE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Current progress, without waiting.
pub(in crate::net) fn progress() -> Progress {
    state().progress()
}
