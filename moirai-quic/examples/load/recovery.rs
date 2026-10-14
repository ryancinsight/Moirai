//! Opt-in aggregation of the pinned engine's real recovery events.
//!
//! Counts measure event occurrences, not blocked durations. Timer lateness
//! measures shared-driver wake time minus requested deadline, including
//! non-QUIC timers. It excludes subsequent protocol scheduling delay.

use serde::Serialize;
use std::convert::Infallible;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::filter::{LevelFilter, filter_fn};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

#[derive(Clone)]
pub(crate) struct Recovery {
    enabled: bool,
    started: Instant,
    state: Arc<Mutex<State>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
struct Counts {
    congestion_blocked: u64,
    pacing_blocked: u64,
    anti_amplification_blocked: u64,
    data_blocked: u64,
    stream_data_blocked: u64,
    established: u64,
    pto_initial: u64,
    pto_handshake: u64,
    pto_data: u64,
    max_pto_backoff: u64,
    max_pto_in_flight_bytes: u64,
    timer_driver_wakes: u64,
    timer_lateness_ns: u128,
    max_timer_lateness_ns: u128,
}

#[derive(Clone, Serialize)]
struct Phase {
    elapsed_ns: u128,
    phase: String,
    counts: Counts,
}

#[derive(Clone, Serialize)]
struct ProbeTimeout {
    elapsed_ns: u128,
    phase: String,
    space: String,
    in_flight_bytes: u64,
    backoff: u64,
}

struct State {
    phase: String,
    counts: Counts,
    phases: Vec<Phase>,
    timeouts: Vec<ProbeTimeout>,
}

#[derive(Serialize)]
struct Snapshot {
    schema: u32,
    observed_unix_ns: u128,
    elapsed_ns: u128,
    observation: String,
    phase: String,
    counts: Counts,
    phases: Vec<Phase>,
    timeouts: Vec<ProbeTimeout>,
}

impl Recovery {
    pub(crate) fn install() -> super::ProbeResult<Self> {
        let enabled = match std::env::var("MOIRAI_QUIC_RECOVERY_TRACE") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "1" => true,
            _ => {
                return Err(
                    super::invalid("MOIRAI_QUIC_RECOVERY_TRACE must be absent or 1").into(),
                );
            }
        };
        let recovery = Self::new(enabled);
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stderr)
                    .with_ansi(false)
                    .with_filter(LevelFilter::INFO),
            )
            .with(recovery.clone().with_filter(filter_fn(move |metadata| {
                enabled && selected(metadata.target())
            })))
            .try_init()?;
        Ok(recovery)
    }

    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            started: Instant::now(),
            state: Arc::new(Mutex::new(State {
                phase: "handshake".into(),
                counts: Counts::default(),
                phases: Vec::new(),
                timeouts: Vec::new(),
            })),
        }
    }

    fn snapshot(&self, observation: &str) -> Snapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Snapshot {
            schema: 1,
            observed_unix_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("probe clock follows epoch")
                .as_nanos(),
            elapsed_ns: self.started.elapsed().as_nanos(),
            observation: observation.into(),
            phase: state.phase.clone(),
            counts: state.counts,
            phases: state.phases.clone(),
            timeouts: state.timeouts.clone(),
        }
    }

    pub(crate) fn emit(&self, observation: &str) -> super::ProbeResult<()> {
        if self.enabled {
            tracing::info!(protocol_recovery = %serde_json::to_string(&self.snapshot(observation))?,
                "protocol recovery observation");
        }
        Ok(())
    }

    pub(crate) async fn observe(&self) -> Infallible {
        if !self.enabled {
            return std::future::pending().await;
        }
        loop {
            moirai_async::timer::sleep(Duration::from_secs(1)).await;
            // Serialization errors invalidate evidence rather than dropping it.
            self.emit("sample").expect("recovery evidence serializes");
        }
    }
}

fn selected(target: &str) -> bool {
    matches!(
        target,
        "quinn_proto::connection"
            | "quinn_proto::connection::streams::state"
            | "moirai_async::timer::driver"
    ) || target.starts_with("quic_load::load")
}

#[derive(Default)]
struct Fields {
    message: String,
    phase: String,
    space: String,
    flight: Option<u64>,
    backoff: Option<u64>,
    lateness: Option<u128>,
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "phase" {
            self.phase = value.into();
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "in_flight" => self.flight = Some(value),
            "count" => self.backoff = Some(value),
            "lateness_ns" => self.lateness = Some(u128::from(value)),
            _ => {}
        }
    }

    fn record_u128(&mut self, field: &Field, value: u128) {
        if field.name() == "lateness_ns" {
            self.lateness = Some(value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        // Condition 1: tracing erases heterogeneous event fields in Visit's ABI.
        match field.name() {
            "message" => self.message = format!("{value:?}"),
            "space" => self.space = format!("{value:?}").trim_matches('"').into(),
            _ => {}
        }
    }
}

impl<S: Subscriber> Layer<S> for Recovery {
    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        let target = event.metadata().target();
        if !self.enabled || !selected(target) {
            return;
        }
        let mut fields = Fields::default();
        event.record(&mut fields);
        let protocol = target.starts_with("quinn_proto::connection");
        let timer = target == "moirai_async::timer::driver"
            && fields.message == "timer driver wakes registration";
        let phase = target.starts_with("quic_load::load") && fields.message == "load probe phase";
        if !protocol && !timer && !phase {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if phase {
            state.phase = fields.phase;
            let entry = Phase {
                elapsed_ns: self.started.elapsed().as_nanos(),
                phase: state.phase.clone(),
                counts: state.counts,
            };
            state.phases.push(entry);
        } else if timer {
            let lateness = fields
                .lateness
                .expect("timer trace records actual lateness");
            state.counts.timer_driver_wakes += 1;
            state.counts.timer_lateness_ns += lateness;
            state.counts.max_timer_lateness_ns = state.counts.max_timer_lateness_ns.max(lateness);
        } else {
            match fields.message.as_str() {
                "blocked by congestion control" => state.counts.congestion_blocked += 1,
                "blocked by pacing" => state.counts.pacing_blocked += 1,
                "blocked by anti-amplification" => state.counts.anti_amplification_blocked += 1,
                "DATA_BLOCKED" => state.counts.data_blocked += 1,
                "STREAM_DATA_BLOCKED" => state.counts.stream_data_blocked += 1,
                "established" => state.counts.established += 1,
                "PTO fired" => {
                    let flight = fields
                        .flight
                        .expect("pinned Quinn records PTO flight bytes");
                    let backoff = fields.backoff.expect("pinned Quinn records PTO backoff");
                    match fields.space.as_str() {
                        "Initial" => state.counts.pto_initial += 1,
                        "Handshake" => state.counts.pto_handshake += 1,
                        "Data" => state.counts.pto_data += 1,
                        _ => panic!("pinned Quinn emits an unknown PTO packet space"),
                    }
                    state.counts.max_pto_backoff = state.counts.max_pto_backoff.max(backoff);
                    state.counts.max_pto_in_flight_bytes =
                        state.counts.max_pto_in_flight_bytes.max(flight);
                    let entry = ProbeTimeout {
                        elapsed_ns: self.started.elapsed().as_nanos(),
                        phase: state.phase.clone(),
                        space: fields.space,
                        in_flight_bytes: flight,
                        backoff,
                    };
                    state.timeouts.push(entry);
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_tracing_events_preserve_counts_spaces_and_timer_values() {
        let recovery = Recovery::new(true);
        let subscriber = tracing_subscriber::registry().with(recovery.clone());
        tracing::subscriber::with_default(subscriber, || {
            let _arrange = tracing::info_span!("arrange").entered();
            tracing::info!(phase = "authenticated", "load probe phase");
            let _act = tracing::info_span!("act").entered();
            tracing::trace!(target: "quinn_proto::connection", "blocked by congestion control");
            tracing::trace!(target: "quinn_proto::connection", "blocked by pacing");
            tracing::trace!(target: "quinn_proto::connection", "blocked by pacing");
            tracing::trace!(target: "quinn_proto::connection", "blocked by anti-amplification");
            tracing::trace!(target: "quinn_proto::connection", in_flight = 12_000_u64, count = 2_u32, space = ?"Handshake", "PTO fired");
            tracing::trace!(target: "quinn_proto::connection", in_flight = 2_400_u64, count = 0_u32, space = ?"Data", "PTO fired");
            tracing::trace!(target: "moirai_async::timer::driver", lateness_ns = 17_u128, "timer driver wakes registration");
            tracing::trace!(target: "moirai_async::timer::driver", lateness_ns = 31_u128, "timer driver wakes registration");
            tracing::trace!(target: "quinn_proto::connection::streams::state", "DATA_BLOCKED");
            tracing::trace!(target: "quinn_proto::connection::streams::state", "STREAM_DATA_BLOCKED");
            tracing::trace!(target: "quinn_proto::connection", "established");
            // A message from a different component must not count as Quinn.
            tracing::trace!("blocked by pacing");
        });
        let snapshot = recovery.snapshot("test");
        assert_eq!(snapshot.phase, "authenticated");
        assert_eq!(snapshot.counts.congestion_blocked, 1);
        assert_eq!(snapshot.counts.pacing_blocked, 2);
        assert_eq!(snapshot.counts.anti_amplification_blocked, 1);
        assert_eq!(snapshot.counts.pto_handshake, 1);
        assert_eq!(snapshot.counts.pto_data, 1);
        assert_eq!(snapshot.counts.pto_initial, 0);
        assert_eq!(snapshot.counts.max_pto_backoff, 2);
        assert_eq!(snapshot.counts.max_pto_in_flight_bytes, 12_000);
        assert_eq!(snapshot.counts.timer_driver_wakes, 2);
        assert_eq!(snapshot.counts.timer_lateness_ns, 48);
        assert_eq!(snapshot.counts.max_timer_lateness_ns, 31);
        assert_eq!(snapshot.counts.data_blocked, 1);
        assert_eq!(snapshot.counts.stream_data_blocked, 1);
        assert_eq!(snapshot.counts.established, 1);
    }

    #[test]
    fn disabled_capture_and_nonselected_events_do_not_change_counts() {
        let recovery = Recovery::new(false);
        let subscriber = tracing_subscriber::registry().with(recovery.clone());
        tracing::subscriber::with_default(subscriber, || {
            let _act = tracing::info_span!("act").entered();
            tracing::trace!(target: "quinn_proto::connection", "blocked by pacing");
            tracing::trace!(target: "moirai_async::timer::driver", lateness_ns = 31_u128, "timer driver wakes registration");
        });
        assert_eq!(recovery.snapshot("test").counts, Counts::default());
        assert_eq!(recovery.snapshot("test").phase, "handshake");
    }
}
