//! Bounded common start for concurrent host threads.

use std::{
    panic::resume_unwind,
    sync::{Condvar, Mutex, MutexGuard, PoisonError},
    thread,
    time::{Duration, Instant},
};

use crate::webview_limits::MAX_WEBVIEW_WAIT_MILLISECONDS;

/// Why a party left the start line without every party arriving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StartLineError {
    /// A party ended before arriving, so the line can never complete.
    PeerFailed,
    /// Not every party arrived within the arrival wait.
    TimedOut,
}

#[derive(Default)]
struct LineState {
    arrived: usize,
    first_failure: Option<usize>,
}

/// Rendezvous of a fixed number of parties that ends early, with a typed
/// error, when a party fails before arriving or does not arrive in time.
pub(super) struct StartLine {
    parties: usize,
    state: Mutex<LineState>,
    changed: Condvar,
}

impl StartLine {
    pub(super) fn new(parties: usize) -> Self {
        Self {
            parties,
            state: Mutex::new(LineState::default()),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, LineState> {
        // The state is two counters that no panic can leave inconsistent.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn fail(&self, party: usize) {
        self.lock().first_failure.get_or_insert(party);
        self.changed.notify_all();
    }

    fn first_failure(&self) -> Option<usize> {
        self.lock().first_failure
    }

    fn arrive(&self, wait: Duration) -> Result<(), StartLineError> {
        let deadline = Instant::now()
            .checked_add(wait)
            .ok_or(StartLineError::TimedOut)?;
        let mut state = self.lock();
        state.arrived += 1;
        if state.arrived == self.parties {
            self.changed.notify_all();
        }
        loop {
            if state.first_failure.is_some() {
                return Err(StartLineError::PeerFailed);
            }
            if state.arrived == self.parties {
                return Ok(());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(StartLineError::TimedOut);
            }
            state = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

/// One party's handle; dropping it before [`Self::arrive`] fails the line.
pub(super) struct Party<'line> {
    index: usize,
    line: &'line StartLine,
    wait: Duration,
    arrived: bool,
}

impl Party<'_> {
    /// Blocks until every party has arrived.
    ///
    /// # Errors
    ///
    /// Returns [`StartLineError::PeerFailed`] when a party ended before
    /// arriving and [`StartLineError::TimedOut`] when the arrival wait ends.
    pub(super) fn arrive(&mut self) -> Result<(), StartLineError> {
        self.arrived = true;
        self.line.arrive(self.wait)
    }
}

impl Drop for Party<'_> {
    fn drop(&mut self) {
        if !self.arrived {
            self.line.fail(self.index);
        }
    }
}

/// Runs `parties` threads of `work` and joins them all, resuming the panic of
/// the first party that ended before arriving, or else the first panic, so the
/// root cause surfaces instead of its peers' start-line errors.
pub(super) fn run_from_common_start<F>(parties: usize, work: F)
where
    F: Fn(usize, &mut Party<'_>) + Sync,
{
    // The committed nextest profile reports a test at 30 seconds and ends it
    // at 60, so a party still setting up after the provider's 30 second wait
    // limit has breached the slow mark rather than started.
    let wait = Duration::from_millis(u64::from(MAX_WEBVIEW_WAIT_MILLISECONDS));
    let line = StartLine::new(parties);
    let mut panics = thread::scope(|scope| {
        let threads = (0..parties)
            .map(|index| {
                let line = &line;
                let work = &work;
                scope.spawn(move || {
                    let mut party = Party {
                        index,
                        line,
                        wait,
                        arrived: false,
                    };
                    work(index, &mut party);
                })
            })
            .collect::<Vec<_>>();
        threads
            .into_iter()
            .map(|thread| thread.join().err())
            .collect::<Vec<_>>()
    });
    let root = line
        .first_failure()
        .and_then(|party| panics.get_mut(party)?.take())
        .or_else(|| panics.into_iter().flatten().next());
    if let Some(payload) = root {
        resume_unwind(payload);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::atomic::{AtomicUsize, Ordering},
        thread,
        time::Duration,
    };

    use super::{Party, StartLine, StartLineError, run_from_common_start};

    #[test]
    fn every_party_crosses_once_all_have_arrived() {
        let crossed = AtomicUsize::new(0);
        run_from_common_start(8, |_, party| {
            party.arrive().expect("all parties arrive");
            crossed.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(crossed.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn a_party_ending_before_the_line_fails_the_run_with_its_own_panic() {
        let observed_failure = AtomicUsize::new(0);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_from_common_start(4, |index, party| {
                assert!(index != 2, "setup failed on party 2");
                assert_eq!(party.arrive(), Err(StartLineError::PeerFailed));
                observed_failure.fetch_add(1, Ordering::SeqCst);
            });
        }));
        let payload = outcome.expect_err("the failed setup fails the run");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("setup failed on party 2")
        );
        assert_eq!(observed_failure.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn the_run_resumes_the_failing_partys_panic_not_a_peers_start_line_panic() {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run_from_common_start(5, |index, party| {
                assert!(index != 3, "setup failed on party 3");
                // Parties 0-2 precede the failing party in thread order and
                // panic here, so the root cause is not the first panic.
                party.arrive().expect("every party reached the start line");
            });
        }));
        let payload = outcome.expect_err("the failed setup fails the run");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("setup failed on party 3")
        );
    }

    #[test]
    fn a_party_that_never_arrives_ends_the_wait_with_timed_out() {
        let line = StartLine::new(2);
        let waiting = thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                Party {
                    index: 0,
                    line: &line,
                    wait: Duration::from_millis(50),
                    arrived: false,
                }
                .arrive()
            });
            // This party holds its handle without arriving while the waiter's
            // wait runs out, then drops it.
            let absent = Party {
                index: 1,
                line: &line,
                wait: Duration::from_millis(50),
                arrived: false,
            };
            let outcome = waiter.join().expect("waiting party returns");
            drop(absent);
            outcome
        });
        assert_eq!(waiting, Err(StartLineError::TimedOut));
    }
}
