//! Numeric engine observations from the separately built read-only patch.
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use tracing::Event;
use tracing::field::{Field, Visit};

pub(super) const TARGET: &str = "quinn_proto::observation";

#[derive(Clone, Default, Serialize)]
pub(super) struct Capture {
    observed_states: u64,
    handshake: Vec<Point>,
    events: Vec<Point>,
    ptos: Vec<Point>,
    buckets: Vec<Bucket>,
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
            "spaces" | "queued" | "sent" | "acked" | "streams" | "timers" | "pto" | "ranges" => {
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

    pub(super) fn snapshot(&self, observation: &str) -> Option<Self> {
        if self.observed_states == 0 && self.events.is_empty() {
            return None;
        }
        let mut snapshot = self.clone();
        if observation == "sample" {
            snapshot.buckets = self.buckets.last().cloned().into_iter().collect();
        }
        Some(snapshot)
    }
}
