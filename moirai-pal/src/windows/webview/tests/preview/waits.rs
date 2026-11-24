//! Per-host durations of the finite WebView2 waits.
//!
//! A re-show run that passes the slow mark names its slow wait: each host
//! thread records the lifetime of every span it closes, and a summary event
//! names the slowest host of each wait.

use std::{
    io,
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant},
};

use tracing::{Subscriber, span};
use tracing_subscriber::{
    Layer, Registry,
    filter::filter_fn,
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
};

/// One closed span: the wait `name` that `host` spent `elapsed` in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Wait {
    pub(super) host: usize,
    pub(super) name: &'static str,
    pub(super) elapsed: Duration,
}

/// The longest `name` wait among the `spans` closed on every host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Slowest {
    pub(super) name: &'static str,
    pub(super) spans: usize,
    pub(super) host: usize,
    pub(super) elapsed: Duration,
}

/// Waits closed on any host thread, shared by every recorder of one run.
#[derive(Clone, Default)]
pub(super) struct WaitLog(Arc<Mutex<Vec<Wait>>>);

struct Started(Instant);

struct Recorder {
    host: usize,
    log: WaitLog,
}

impl<S> Layer<S> for Recorder
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(
        &self,
        _attributes: &span::Attributes<'_>,
        id: &span::Id,
        context: Context<'_, S>,
    ) {
        if let Some(span) = context.span(id) {
            span.extensions_mut().insert(Started(Instant::now()));
        }
    }

    fn on_close(&self, id: span::Id, context: Context<'_, S>) {
        let Some(span) = context.span(&id) else {
            return;
        };
        let Some(Started(started)) = span.extensions_mut().remove::<Started>() else {
            return;
        };
        self.log.push(Wait {
            host: self.host,
            name: span.name(),
            elapsed: started.elapsed(),
        });
    }
}

impl WaitLog {
    fn push(&self, wait: Wait) {
        // The log is append-only, so no panic leaves it inconsistent.
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(wait);
    }

    /// Records every span the calling thread closes for `host` until the
    /// returned guard drops.
    ///
    /// The recorder enables spans only, so an event-level check such as
    /// `tracing::enabled!` on the thread keeps the answer it had without it.
    pub(super) fn recording(&self, host: usize) -> tracing::subscriber::DefaultGuard {
        let recorder = Recorder {
            host,
            log: self.clone(),
        }
        .with_filter(filter_fn(|metadata| metadata.is_span()));
        tracing::subscriber::set_default(Registry::default().with(recorder))
    }

    /// The recorded waits in the order their spans closed.
    pub(super) fn waits(&self) -> Vec<Wait> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// A guard that reports the slowest host of each wait when dropped, also
    /// while a failing host's panic unwinds.
    pub(super) fn report_on_drop(&self) -> Report<'_> {
        Report(self)
    }
}

/// Reports a [`WaitLog`] through `tracing` events when dropped.
pub(super) struct Report<'log>(&'log WaitLog);

impl Drop for Report<'_> {
    fn drop(&mut self) {
        for slowest in slowest_by_name(&self.0.waits()) {
            tracing::info!(
                wait = slowest.name,
                spans = slowest.spans,
                slowest_host = slowest.host,
                slowest_seconds = slowest.elapsed.as_secs_f64(),
                "slowest wait"
            );
        }
    }
}

/// The longest wait of each name, ordered by name; the lowest host index wins
/// a tie.
pub(super) fn slowest_by_name(waits: &[Wait]) -> Vec<Slowest> {
    let mut ordered: Vec<&Wait> = waits.iter().collect();
    ordered.sort_by_key(|wait| (wait.name, wait.host));
    let mut slowest: Vec<Slowest> = Vec::new();
    for wait in ordered {
        match slowest.last_mut() {
            Some(current) if current.name == wait.name => {
                current.spans += 1;
                if wait.elapsed > current.elapsed {
                    current.host = wait.host;
                    current.elapsed = wait.elapsed;
                }
            }
            _ => slowest.push(Slowest {
                name: wait.name,
                spans: 1,
                host: wait.host,
                elapsed: wait.elapsed,
            }),
        }
    }
    slowest
}

/// Installs a subscriber that writes events to the test's captured stderr, so
/// the report reaches the nextest log of a run that passes with
/// `--success-output final` or fails.
pub(super) fn stderr_trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_writer(io::stderr)
            .with_ansi(false)
            .log_internal_errors(false)
            .finish(),
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{Slowest, Wait, WaitLog, slowest_by_name};

    fn wait(host: usize, name: &'static str, milliseconds: u64) -> Wait {
        Wait {
            host,
            name,
            elapsed: Duration::from_millis(milliseconds),
        }
    }

    #[test]
    fn closed_spans_are_recorded_inner_first_with_their_host() {
        let log = WaitLog::default();
        {
            let _recording = log.recording(7);
            tracing::debug_span!("outer").in_scope(|| {
                tracing::debug_span!("inner").in_scope(|| ());
            });
        }
        tracing::debug_span!("unrecorded").in_scope(|| ());
        let waits = log.waits();
        assert_eq!(
            waits
                .iter()
                .map(|wait| (wait.host, wait.name))
                .collect::<Vec<_>>(),
            [(7, "inner"), (7, "outer")]
        );
        // A span encloses everything closed inside it.
        assert!(waits[1].elapsed >= waits[0].elapsed);
    }

    #[test]
    fn the_recorder_leaves_event_level_checks_unchanged() {
        let log = WaitLog::default();
        assert!(!tracing::enabled!(target: "moirai_wait_recorder", tracing::Level::DEBUG));
        let _recording = log.recording(0);
        assert!(!tracing::enabled!(target: "moirai_wait_recorder", tracing::Level::DEBUG));
    }

    #[test]
    fn each_host_thread_records_under_its_own_index() {
        let log = WaitLog::default();
        std::thread::scope(|scope| {
            for host in 0..4 {
                let log = &log;
                scope.spawn(move || {
                    let _recording = log.recording(host);
                    tracing::debug_span!("wait").in_scope(|| ());
                });
            }
        });
        let mut hosts: Vec<usize> = log.waits().iter().map(|wait| wait.host).collect();
        hosts.sort_unstable();
        assert_eq!(hosts, [0, 1, 2, 3]);
    }

    #[test]
    fn the_slowest_host_of_each_wait_is_named_with_the_lowest_index_on_ties() {
        let waits = [
            wait(3, "frame", 40),
            wait(1, "environment", 9),
            wait(2, "frame", 55),
            wait(0, "environment", 9),
            wait(5, "frame", 55),
            wait(4, "capture", 1),
        ];
        assert_eq!(
            slowest_by_name(&waits),
            [
                Slowest {
                    name: "capture",
                    spans: 1,
                    host: 4,
                    elapsed: Duration::from_millis(1),
                },
                Slowest {
                    name: "environment",
                    spans: 2,
                    host: 0,
                    elapsed: Duration::from_millis(9),
                },
                Slowest {
                    name: "frame",
                    spans: 3,
                    host: 2,
                    elapsed: Duration::from_millis(55),
                },
            ]
        );
        assert_eq!(slowest_by_name(&[]), []);
    }
}
