//! Numeric engine observations from the separately built read-only patch.
use serde::Serialize;
use serde_json::Value;
use std::{collections::BTreeMap, fmt};
use tracing::Event;
use tracing::field::{Field, Visit};

pub(super) const TARGET: &str = "quinn_proto::observation";

// Whole-second bounds exceed 7*61 callback/outcome and 2*61 blocker buckets.
const CALLBACK_BUCKET_LIMIT: usize = 1_024;
const BLOCKER_BUCKET_LIMIT: usize = 256;

#[derive(Clone, Default, Serialize)]
pub(super) struct Capture {
    observed_states: u64,
    handshake: Vec<Point>,
    events: Vec<Point>,
    ptos: Vec<Point>,
    buckets: Vec<Bucket>,
    callback_observations: u64,
    callback_buckets_dropped: u64,
    callbacks: Vec<CallbackBucket>,
    blocker_observations: u64,
    blocker_buckets_dropped: u64,
    blockers: Vec<EventBucket>,
}

#[derive(Clone, PartialEq, Serialize)]
struct Point {
    elapsed_ns: u128,
    kind: String,
    fields: BTreeMap<String, Value>,
}

#[derive(Clone, Serialize)]
struct Bucket {
    second: u128,
    observations: u64,
    first: Point,
    last: Point,
    min_flight_bytes: u64,
    max_flight_bytes: u64,
    min_window_bytes: u64,
    max_window_bytes: u64,
    min_connection_credit: u64,
    max_connection_credit: u64,
    max_pacing_remaining_ns: u64,
    pacing_armed_observations: u64,
}

#[derive(Clone, Serialize)]
struct CallbackBucket {
    second: u128,
    callback: String,
    outcome: String,
    observations: u64,
    first: Point,
    last: Point,
}

#[derive(Clone, Serialize)]
struct EventBucket {
    second: u128,
    kind: String,
    observations: u64,
    first: Point,
    last: Point,
}

#[derive(Default)]
struct Fields {
    message: String,
    values: BTreeMap<String, Value>,
}

impl Visit for Fields {
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.values.insert(field.name().into(), value.into());
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.values.insert(field.name().into(), value.into());
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.values.insert(field.name().into(), value.into());
    }
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        // Condition 1: tracing erases heterogeneous fields at its Visit ABI.
        let raw = format!("{value:?}");
        match field.name() {
            "message" => self.message = raw,
            "space" => {
                self.values.insert("space".into(), raw.into());
            }
            "spaces" | "queued" | "sent" | "acked" | "streams" | "timers" | "pto" | "ranges"
            | "state_before" | "state_after" => {
                self.values.insert(
                    field.name().into(),
                    serde_json::from_str(&raw).expect("pinned numeric arrays are JSON"),
                );
            }
            _ => {}
        }
    }
}

fn number(point: &Point, name: &str) -> u64 {
    point.fields[name].as_u64().expect("engine integer field")
}

fn pacing(point: &Point) -> Option<u64> {
    point.fields["timers"]
        .as_array()
        .expect("timer rows")
        .iter()
        .find(|row| row[0] == 6)
        .map(|row| row[1].as_u64().expect("timer remaining ns"))
}

impl Capture {
    pub(super) fn record(&mut self, event: &Event<'_>, elapsed_ns: u128) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let point = Point {
            elapsed_ns,
            kind: fields.message,
            fields: fields.values,
        };
        if point.kind == "engine cubic callback" {
            self.record_callback(point);
            return;
        }
        if matches!(
            point.kind.as_str(),
            "engine congestion blocked" | "engine pacing blocked"
        ) {
            self.record_blocker(point);
            return;
        }
        if point.kind != "engine recovery state" {
            self.events.push(point);
            return;
        }
        self.observed_states += 1;
        let handshake_keys = point.fields["spaces"][1][1] == 1;
        if point.fields["engine_handshake"] == true || handshake_keys {
            self.handshake.push(point.clone());
        }
        if point.fields["reason"] == "pto" {
            self.ptos.push(point.clone());
        }
        let flight = number(&point, "flight_bytes");
        let window = number(&point, "window_bytes");
        let credit = number(&point, "connection_max_data")
            .checked_sub(number(&point, "connection_data_sent"))
            .expect("connection credit invariant");
        let delay = pacing(&point);
        let second = elapsed_ns / 1_000_000_000;
        if let Some(bucket) = self.buckets.last_mut().filter(|b| b.second == second) {
            bucket.observations += 1;
            bucket.min_flight_bytes = bucket.min_flight_bytes.min(flight);
            bucket.max_flight_bytes = bucket.max_flight_bytes.max(flight);
            bucket.min_window_bytes = bucket.min_window_bytes.min(window);
            bucket.max_window_bytes = bucket.max_window_bytes.max(window);
            bucket.min_connection_credit = bucket.min_connection_credit.min(credit);
            bucket.max_connection_credit = bucket.max_connection_credit.max(credit);
            bucket.max_pacing_remaining_ns = bucket.max_pacing_remaining_ns.max(delay.unwrap_or(0));
            bucket.pacing_armed_observations += u64::from(delay.is_some());
            bucket.last = point;
        } else {
            self.buckets.push(Bucket {
                second,
                observations: 1,
                first: point.clone(),
                last: point,
                min_flight_bytes: flight,
                max_flight_bytes: flight,
                min_window_bytes: window,
                max_window_bytes: window,
                min_connection_credit: credit,
                max_connection_credit: credit,
                max_pacing_remaining_ns: delay.unwrap_or(0),
                pacing_armed_observations: u64::from(delay.is_some()),
            });
        }
    }

    fn record_callback(&mut self, point: Point) {
        self.callback_observations += 1;
        let second = point.elapsed_ns / 1_000_000_000;
        let callback = point.fields["callback"]
            .as_str()
            .expect("cubic callback kind")
            .to_owned();
        let outcome = point.fields["outcome"]
            .as_str()
            .expect("cubic callback outcome")
            .to_owned();
        if let Some(bucket) = self.callbacks.iter_mut().rev().find(|bucket| {
            bucket.second == second && bucket.callback == callback && bucket.outcome == outcome
        }) {
            bucket.observations += 1;
            bucket.last = point;
        } else if self.callbacks.len() < CALLBACK_BUCKET_LIMIT {
            self.callbacks.push(CallbackBucket {
                second,
                callback,
                outcome,
                observations: 1,
                first: point.clone(),
                last: point,
            });
        } else {
            self.callback_buckets_dropped += 1;
        }
    }

    fn record_blocker(&mut self, point: Point) {
        self.blocker_observations += 1;
        let second = point.elapsed_ns / 1_000_000_000;
        if let Some(bucket) = self
            .blockers
            .iter_mut()
            .rev()
            .find(|bucket| bucket.second == second && bucket.kind == point.kind)
        {
            bucket.observations += 1;
            bucket.last = point;
        } else if self.blockers.len() < BLOCKER_BUCKET_LIMIT {
            self.blockers.push(EventBucket {
                second,
                kind: point.kind.clone(),
                observations: 1,
                first: point.clone(),
                last: point,
            });
        } else {
            self.blocker_buckets_dropped += 1;
        }
    }

    pub(super) fn snapshot(&self, observation: &str) -> Option<Self> {
        if self.observed_states == 0
            && self.events.is_empty()
            && self.callback_observations == 0
            && self.blocker_observations == 0
        {
            return None;
        }
        let mut snapshot = self.clone();
        if observation == "sample" {
            snapshot.buckets = self.buckets.last().cloned().into_iter().collect();
        }
        Some(snapshot)
    }
}

#[cfg(all(test, quic_observation))]
mod tests {
    use super::{super::Recovery, BLOCKER_BUCKET_LIMIT, CALLBACK_BUCKET_LIMIT, Capture, Point};
    use moirai_quic::{Identity, QuicConfig, QuicEndpoint};
    use serde_json::{Value, json};
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tracing_subscriber::prelude::*;

    fn recovery() -> (Recovery, tracing::subscriber::DefaultGuard) {
        let recovery = Recovery::new(true, false);
        let subscriber =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(recovery.clone()));
        (recovery, subscriber)
    }

    fn emit_callback(outcome: &str, before: [u64; 6], after: [u64; 6]) {
        tracing::trace!(
            target: super::TARGET,
            callback = "ack",
            outcome,
            state_before = ?before,
            state_after = ?after,
            "engine cubic callback"
        );
    }

    fn point(second: usize, kind: &str, callback: bool) -> Point {
        let fields = if callback {
            BTreeMap::from([
                ("callback".into(), Value::from("ack")),
                ("outcome".into(), Value::from("avoidance")),
            ])
        } else {
            BTreeMap::new()
        };
        Point {
            elapsed_ns: u128::try_from(second).expect("test second fits u128") * 1_000_000_000,
            kind: kind.into(),
            fields,
        }
    }

    #[test]
    fn collector_groups_real_events_and_serializes_endpoints() {
        let (recovery, _subscriber) = recovery();
        emit_callback(
            "avoidance",
            [10, 20, 1_200, 3, 4, 5],
            [11, 20, 1_200, 3, 4, 0],
        );
        emit_callback(
            "avoidance",
            [11, 20, 1_200, 3, 4, 0],
            [12, 20, 1_200, 3, 4, 1],
        );
        tracing::trace!(
            target: super::TARGET,
            flight_bytes = 10_800_u64,
            bytes_to_send = 1_200_u64,
            window_bytes = 12_000_u64,
            "engine congestion blocked"
        );
        tracing::trace!(
            target: super::TARGET,
            flight_bytes = 12_u64,
            bytes_to_send = 1_200_u64,
            window_bytes = 24_000_u64,
            smoothed_rtt_ns = 100_000_000_u64,
            delay_ns = 40_000_000_u64,
            "engine pacing blocked"
        );
        let engine = serde_json::to_value(
            recovery
                .snapshot("test")
                .engine
                .expect("real tracing events create an engine capture"),
        )
        .expect("engine capture serializes");
        assert_eq!(engine["callback_observations"], 2);
        let callbacks = engine["callbacks"]
            .as_array()
            .expect("callback buckets serialize as an array");
        let callback_count = callbacks.iter().map(|bucket| {
            bucket["observations"]
                .as_u64()
                .expect("callback counts serialize as integers")
        });
        assert_eq!(callback_count.sum::<u64>(), 2);
        assert!(
            callbacks
                .iter()
                .all(|bucket| { bucket["callback"] == "ack" && bucket["outcome"] == "avoidance" })
        );
        assert_eq!(
            callbacks[0]["first"]["fields"]["state_before"],
            json!([10, 20, 1_200, 3, 4, 5])
        );
        let last = callbacks
            .last()
            .expect("two callback observations create a bucket");
        assert_eq!(
            last["last"]["fields"]["state_after"],
            json!([12, 20, 1_200, 3, 4, 1])
        );
        assert_eq!(engine["blocker_observations"], 2);
        assert_eq!(engine["blockers"][0]["kind"], "engine congestion blocked");
        assert_eq!(
            engine["blockers"][0]["first"]["fields"],
            json!({
                "flight_bytes": 10_800,
                "bytes_to_send": 1_200,
                "window_bytes": 12_000,
            })
        );
        assert_eq!(engine["blockers"][1]["kind"], "engine pacing blocked");
        assert_eq!(
            engine["blockers"][1]["first"]["fields"]["delay_ns"],
            40_000_000
        );
    }

    #[test]
    fn collector_bounds_distinct_callback_and_blocker_buckets() {
        let mut grouped = Capture::default();
        grouped.record_callback(point(7, "engine cubic callback", true));
        grouped.record_callback(point(7, "engine cubic callback", true));
        assert_eq!(grouped.callbacks.len(), 1);
        assert_eq!(grouped.callbacks[0].observations, 2);
        assert_eq!(grouped.callbacks[0].first.elapsed_ns, 7_000_000_000);
        assert_eq!(grouped.callbacks[0].last.elapsed_ns, 7_000_000_000);
        let mut capture = Capture::default();
        for second in 0..=CALLBACK_BUCKET_LIMIT {
            capture.record_callback(point(second, "engine cubic callback", true));
        }
        for second in 0..=BLOCKER_BUCKET_LIMIT {
            capture.record_blocker(point(second, "engine congestion blocked", false));
        }

        assert_eq!(capture.callback_observations, 1_025);
        assert_eq!(capture.callbacks.len(), CALLBACK_BUCKET_LIMIT);
        assert_eq!(capture.callback_buckets_dropped, 1);
        assert_eq!(capture.callbacks[0].first.elapsed_ns, 0);
        assert_eq!(capture.callbacks[1_023].last.elapsed_ns, 1_023_000_000_000);
        assert_eq!(capture.blocker_observations, 257);
        assert_eq!(capture.blockers.len(), BLOCKER_BUCKET_LIMIT);
        assert_eq!(capture.blocker_buckets_dropped, 1);
        assert_eq!(capture.blockers[255].last.elapsed_ns, 255_000_000_000);
    }

    #[test]
    fn real_transport_reaches_observation_collector() {
        let (recovery, _subscriber) = recovery();
        let executor = Arc::new(
            moirai_executor::ExecutorBuilder::new()
                .worker_threads(2)
                .async_threads(1)
                .build()
                .expect("invariant: the test host supports the configured executor"),
        );
        let config = QuicConfig::new(
            b"moirai-cubic-observation/1",
            1_024 * 1_024,
            16,
            1,
            65_536,
            Duration::from_secs(5),
            Duration::from_secs(10),
        )
        .expect("invariant: test resource and deadline bounds are valid");
        let certificates = crate::load::identity::certificates()
            .expect("invariant: the controlled certificate fixture parses");
        let server = QuicEndpoint::server(
            Arc::clone(&executor),
            "127.0.0.1:0"
                .parse()
                .expect("invariant: the loopback literal is valid"),
            Identity::new(
                certificates,
                crate::load::identity::private_key()
                    .expect("invariant: the controlled private-key fixture parses"),
            ),
            None,
            config.clone(),
        )
        .expect("invariant: controlled server endpoint construction succeeds");
        let client = QuicEndpoint::client(
            Arc::clone(&executor),
            "127.0.0.1:0"
                .parse()
                .expect("invariant: the loopback literal is valid"),
            crate::load::identity::roots().expect("invariant: the controlled trust fixture parses"),
            None,
            config,
        )
        .expect("invariant: controlled client endpoint construction succeeds");
        let address = server
            .local_addr()
            .expect("invariant: the bound endpoint has a local address");
        moirai_executor::block_on(async {
            let (server_connection, client_connection) =
                futures::future::join(server.accept(), client.connect(address, "localhost")).await;
            let server_connection =
                server_connection.expect("invariant: controlled server authentication succeeds");
            let client_connection =
                client_connection.expect("invariant: controlled client authentication succeeds");
            let payload = vec![37_u8; 1_024 * 1_024];
            let sends = futures::future::join_all((0..16).map(|_| async {
                let mut stream = client_connection
                    .open_stream()
                    .await
                    .expect("invariant: the client opens a controlled stream");
                stream.send_message(&payload).await?;
                stream.finish()
            }));
            let receives = futures::future::join_all((0..16).map(|_| async {
                let mut stream = server_connection
                    .accept_stream()
                    .await
                    .expect("invariant: the server accepts a controlled stream");
                stream.recv_message().await
            }));
            let (sent, received) = futures::future::join(sends, receives).await;
            for result in sent {
                result.expect("invariant: each bounded loopback write completes");
            }
            for result in received {
                assert_eq!(
                    result
                        .expect("invariant: each bounded loopback read completes")
                        .as_deref(),
                    Some(payload.as_slice())
                );
            }
        });
        let snapshot = recovery.snapshot("test");
        let engine = snapshot
            .engine
            .expect("invariant: production Quinn emitted engine observations");
        assert_eq!(
            engine
                .buckets
                .iter()
                .map(|bucket| bucket.observations)
                .sum::<u64>(),
            engine.observed_states
        );
        assert!(engine.observed_states > 0);
    }
}
