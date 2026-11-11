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
    disposal_closed: bool,
    disposal_waiting: bool,
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

/// One accepted test job that must settle before another exclusive case starts.
pub(super) struct AcceptedJob;

struct AcceptedState {
    jobs: usize,
    waiters: usize,
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        self.0.state().live -= 1;
        self.0.changed.notify_all();
    }
}

impl Drop for AcceptedJob {
    fn drop(&mut self) {
        let mut accepted = ACCEPTED.0.lock().unwrap_or_else(PoisonError::into_inner);
        accepted.jobs = accepted
            .jobs
            .checked_sub(1)
            .expect("invariant: every completed test job was accepted");
        ACCEPTED.1.notify_all();
    }
}

static ACCEPTED: (Mutex<AcceptedState>, Condvar) = (
    Mutex::new(AcceptedState {
        jobs: 0,
        waiters: 0,
    }),
    Condvar::new(),
);

/// Record a job before it enters a test pool's queue.
pub(super) fn submitted() {
    let mut accepted = ACCEPTED.0.lock().unwrap_or_else(PoisonError::into_inner);
    accepted.jobs = accepted
        .jobs
        .checked_add(1)
        .expect("invariant: bounded test pools cannot exhaust usize jobs");
}

/// Bind completion accounting to a worker-owned admission permit.
pub(super) const fn accepted_job() -> AcceptedJob {
    AcceptedJob
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
                disposal_closed: false,
                disposal_waiting: false,
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
        // Relaxed: every writer holds `state`'s mutex, which totally orders
        // the updates, and an atomic read-modify-write reads the latest
        // value in modification order, so the running maximum is exact.
        // Observers gain the happens-before edge through the same mutex, the
        // condvar, or a join — the load below relies on one of those.
        self.peak.fetch_max(state.live, Ordering::Relaxed);
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
        let mut state = self.state();
        state.disposal_waiting = true;
        self.changed.notify_all();
        while state.disposal_closed {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.disposed += 1;
        state.disposal_waiting = false;
        self.changed.notify_all();
    }

    /// Hold (`true`) or release (`false`) the next disposal publication.
    pub(crate) fn set_disposal_closed(&self, closed: bool) {
        self.state().disposal_closed = closed;
        self.changed.notify_all();
    }

    /// Wait until a worker reaches the disposal-publication boundary.
    pub(crate) fn wait_until_disposal_waiting(&self) {
        let (state, timeout) = self
            .changed
            .wait_timeout_while(self.state(), STAGE_LIMIT, |state| !state.disposal_waiting)
            .unwrap_or_else(PoisonError::into_inner);
        assert!(
            state.disposal_waiting,
            "a worker did not reach disposal publication; timed out: {}",
            timeout.timed_out()
        );
    }

    /// Make the next `count` jobs panic inside the worker.
    pub(crate) fn inject_panics(&self, count: usize) {
        self.state().panics = count;
    }

    /// Highest number of jobs observed running at once.
    ///
    /// Relaxed: a correct reader holds a happens-before edge from the pool —
    /// the state mutex, the condvar, or a join — which is what makes the
    /// latest maximum visible; the load itself needs no ordering edge.
    pub(crate) fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
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
    let exclusive = EXCLUSIVE.lock().unwrap_or_else(PoisonError::into_inner);
    wait_until_quiescent();
    exclusive
}

/// Wait until every accepted test job has published its disposal.
pub(crate) fn wait_until_quiescent() {
    let mut accepted = ACCEPTED.0.lock().unwrap_or_else(PoisonError::into_inner);
    accepted.waiters += 1;
    ACCEPTED.1.notify_all();
    let (mut accepted, timeout) = ACCEPTED
        .1
        .wait_timeout_while(accepted, STAGE_LIMIT, |accepted| accepted.jobs != 0)
        .unwrap_or_else(PoisonError::into_inner);
    let jobs = accepted.jobs;
    accepted.waiters = accepted
        .waiters
        .checked_sub(1)
        .expect("invariant: this quiescence observer registered its wait");
    ACCEPTED.1.notify_all();
    assert_eq!(
        jobs,
        0,
        "accepted blocking-pool jobs did not settle before the exclusive test; timed out: {}",
        timeout.timed_out()
    );
}

/// Wait until a quiescence observer is blocked behind an accepted job.
pub(crate) fn wait_until_quiescence_waiting() {
    let (accepted, timeout) = ACCEPTED
        .1
        .wait_timeout_while(
            ACCEPTED.0.lock().unwrap_or_else(PoisonError::into_inner),
            STAGE_LIMIT,
            |accepted| accepted.waiters == 0,
        )
        .unwrap_or_else(PoisonError::into_inner);
    assert!(
        accepted.waiters != 0,
        "no quiescence observer reached the accepted-job wait; timed out: {}",
        timeout.timed_out()
    );
}
